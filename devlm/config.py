"""Configuration: typed, immutable, validated at load time."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

import yaml

TRAIN_MODES = ("full", "growth_only", "base_only")
CURIOSITY_STRATEGIES = ("surprise", "learning_progress", "random")
TEACHER_BACKENDS = ("scripted", "ollama", "anthropic")
GROWTH_ORDERS = ("top_down", "bottom_up")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_RULES_PER_CREATURE = 4
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ConfigError(ValueError):
    """Raised when a config file is malformed or has invalid values."""


@dataclass(frozen=True)
class ChildConfig:
    model_name: str = "roneneldan/TinyStories-33M"
    tokenizer_name: str = "EleutherAI/gpt-neo-125M"
    device: str = "auto"
    max_length: int = 256
    generation_tokens: int = 40
    temperature: float = 0.8


@dataclass(frozen=True)
class GrowthConfig:
    enabled: bool = True
    bottleneck: int = 64
    order: str = "top_down"
    grow_loss_threshold: float = 2.0
    min_days_between_growth: int = 3


@dataclass(frozen=True)
class SleepConfig:
    train_mode: str = "full"
    learning_rate: float = 3e-5
    growth_learning_rate: float = 1e-3
    epochs: int = 3
    batch_size: int = 8
    max_items: int = 160
    replay_ratio: float = 0.5
    rehearsal_ratio: float = 0.3
    rehearsal_pool: int = 200
    surprise_quantile: float = 0.5
    autobiographical: bool = True
    grad_clip: float = 1.0


@dataclass(frozen=True)
class WorldConfig:
    seed: int = 7
    starting_creatures: int = 3
    arrival_days: tuple[int, ...] = (8, 16, 24)
    rules_per_creature: int = 2
    observations_per_day: int = 10
    noise_ratio: float = 0.2


@dataclass(frozen=True)
class CuriosityConfig:
    strategy: str = "learning_progress"
    questions_per_day: int = 4
    window: int = 3


@dataclass(frozen=True)
class TeacherConfig:
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    anthropic_model: str = "claude-opus-5-5"
    ollama_url: str = "http://localhost:11434"
    max_words: int = 40
    lessons_per_day: int = 1
    timeout_seconds: float = 120.0
    effort: str = "low"


@dataclass(frozen=True)
class ProbeConfig:
    metacog_lags: tuple[int, ...] = (1, 3, 5)
    milestone_variants: int = 3


@dataclass(frozen=True)
class SnapshotConfig:
    keep_every: int = 1


@dataclass(frozen=True)
class Config:
    run_name: str = "first-child"
    runs_dir: str = "runs"
    seed: int = 1234
    child: ChildConfig = field(default_factory=ChildConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    sleep: SleepConfig = field(default_factory=SleepConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    probes: ProbeConfig = field(default_factory=ProbeConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)


def load_config(path: Path) -> Config:
    """Read a YAML file into a validated Config."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path}: {exc}") from exc
    return config_from_dict(raw or {})


def config_from_dict(data: Mapping[str, Any]) -> Config:
    return validate(_build(Config, data, "config"))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return _plain(dataclasses.asdict(cfg))


def save_config(cfg: Config, path: Path) -> None:
    Path(path).write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {unknown}")
    kwargs = {name: _coerce(hints[name], value, f"{where}.{name}") for name, value in data.items()}
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, where: str) -> Any:
    if is_dataclass(hint):
        return _build(hint, value if value is not None else {}, where)
    if get_origin(hint) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where} must be a list")
        item_type = get_args(hint)[0]
        return tuple(_coerce(item_type, v, f"{where}[{i}]") for i, v in enumerate(value))
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    raise ConfigError(f"Unsupported config type at {where}")


def validate(cfg: Config) -> Config:
    """Check value ranges and enums; raise one error listing every problem."""
    checks = [
        (RUN_NAME_PATTERN.match(cfg.run_name) is not None,
         "run_name may only contain letters, digits, '-' and '_' (max 64 chars)"),
        (cfg.sleep.train_mode in TRAIN_MODES, f"sleep.train_mode must be one of {TRAIN_MODES}"),
        (cfg.curiosity.strategy in CURIOSITY_STRATEGIES,
         f"curiosity.strategy must be one of {CURIOSITY_STRATEGIES}"),
        (cfg.teacher.backend in TEACHER_BACKENDS, f"teacher.backend must be one of {TEACHER_BACKENDS}"),
        (cfg.teacher.effort in EFFORT_LEVELS, f"teacher.effort must be one of {EFFORT_LEVELS}"),
        (cfg.growth.order in GROWTH_ORDERS, f"growth.order must be one of {GROWTH_ORDERS}"),
        (cfg.growth.bottleneck >= 1, "growth.bottleneck must be >= 1"),
        (cfg.growth.min_days_between_growth >= 0, "growth.min_days_between_growth must be >= 0"),
        (not (cfg.sleep.train_mode == "growth_only" and not cfg.growth.enabled),
         "sleep.train_mode 'growth_only' needs growth.enabled: true"),
        (cfg.sleep.learning_rate > 0 and cfg.sleep.growth_learning_rate > 0, "learning rates must be > 0"),
        (cfg.sleep.epochs >= 1, "sleep.epochs must be >= 1"),
        (cfg.sleep.batch_size >= 1, "sleep.batch_size must be >= 1"),
        (cfg.sleep.max_items >= 1, "sleep.max_items must be >= 1"),
        (cfg.sleep.rehearsal_pool >= 0, "sleep.rehearsal_pool must be >= 0"),
        (cfg.sleep.grad_clip > 0, "sleep.grad_clip must be > 0"),
        (0.0 <= cfg.sleep.replay_ratio <= 4.0, "sleep.replay_ratio must be between 0 and 4"),
        (0.0 <= cfg.sleep.rehearsal_ratio <= 4.0, "sleep.rehearsal_ratio must be between 0 and 4"),
        (0.0 <= cfg.sleep.surprise_quantile <= 1.0, "sleep.surprise_quantile must be between 0 and 1"),
        (cfg.world.starting_creatures >= 1, "world.starting_creatures must be >= 1"),
        (1 <= cfg.world.rules_per_creature <= MAX_RULES_PER_CREATURE,
         f"world.rules_per_creature must be between 1 and {MAX_RULES_PER_CREATURE}"),
        (cfg.world.observations_per_day >= 1, "world.observations_per_day must be >= 1"),
        (0.0 <= cfg.world.noise_ratio < 1.0, "world.noise_ratio must be in [0, 1)"),
        (all(d >= 2 for d in cfg.world.arrival_days), "world.arrival_days must all be >= 2"),
        (cfg.curiosity.questions_per_day >= 0, "curiosity.questions_per_day must be >= 0"),
        (cfg.curiosity.window >= 1, "curiosity.window must be >= 1"),
        (cfg.teacher.max_words >= 5, "teacher.max_words must be >= 5"),
        (cfg.teacher.lessons_per_day >= 0, "teacher.lessons_per_day must be >= 0"),
        (cfg.teacher.timeout_seconds > 0, "teacher.timeout_seconds must be > 0"),
        (len(cfg.probes.metacog_lags) > 0 and all(lag >= 1 for lag in cfg.probes.metacog_lags),
         "probes.metacog_lags must be a non-empty list of integers >= 1"),
        (cfg.probes.milestone_variants >= 1, "probes.milestone_variants must be >= 1"),
        (cfg.snapshots.keep_every >= 1, "snapshots.keep_every must be >= 1"),
        (cfg.child.max_length >= 16, "child.max_length must be >= 16"),
        (cfg.child.generation_tokens >= 1, "child.generation_tokens must be >= 1"),
        (cfg.child.temperature > 0, "child.temperature must be > 0"),
    ]
    errors = [message for ok, message in checks if not ok]
    if errors:
        raise ConfigError("Invalid config:\n- " + "\n- ".join(errors))
    return cfg
