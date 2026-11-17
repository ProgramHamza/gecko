"""Torch-free interfaces, so the simulation can be tested with a fake learner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from devlm.config import SleepConfig


@dataclass(frozen=True)
class TrainSpec:
    mode: str                     # full | growth_only | base_only
    learning_rate: float
    growth_learning_rate: float
    epochs: int
    batch_size: int
    grad_clip: float

    @staticmethod
    def from_sleep(cfg: SleepConfig, mode: str | None = None) -> "TrainSpec":
        return TrainSpec(
            mode=mode or cfg.train_mode,
            learning_rate=cfg.learning_rate,
            growth_learning_rate=cfg.growth_learning_rate,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            grad_clip=cfg.grad_clip,
        )


@dataclass(frozen=True)
class TrainResult:
    steps: int
    first_epoch_loss: float | None
    last_epoch_loss: float | None


class Learner(Protocol):
    """What the simulation needs from the child model."""

    def surprise(self, texts: Sequence[str]) -> list[float]:
        """Mean per-token negative log-likelihood of each text (higher = more surprising)."""

    def score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        """Mean log-probability of each continuation given the prompt."""

    def generate(self, prompt: str, max_new_tokens: int | None = None, stop_at_newline: bool = True) -> str:
        """Sample a continuation."""

    def train(self, texts: Sequence[str], spec: TrainSpec, seed: int = 0) -> TrainResult:
        """Update weights on the texts."""

    def grow(self) -> bool:
        """Activate one more reserved module. False if none are left."""

    def active_modules(self) -> int:
        """How many growth modules are active."""

    def save(self, path: Path) -> None:
        """Write the full weights (including growth modules) to path."""

    def load(self, path: Path) -> None:
        """Restore weights written by save()."""
