"""Shared fixtures. FakeLearner stands in for the neural child so the loop can be tested without torch."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import pytest

from devlm.config import Config, config_from_dict
from devlm.protocols import TrainResult, TrainSpec

_QA = re.compile(r"What does the (\w+) do (.+)\?")


class FakeLearner:
    """Remembers trained texts verbatim. A fact is 'known' once its statement was trained on."""

    def __init__(self, capacity: int = 2) -> None:
        self.memory: Counter[str] = Counter()
        self.modules = 0
        self.capacity = capacity
        self.train_calls: list[tuple[int, str]] = []

    def surprise(self, texts: Sequence[str]) -> list[float]:
        return [5.0 / (1 + self.memory[t]) for t in texts]

    def score_choices(self, prompt: str, choices: Sequence[str]) -> list[float]:
        match = _QA.search(prompt)
        stem = f"{match.group(2)[0].upper()}{match.group(2)[1:]}, the {match.group(1)}" if match else prompt
        return [float(sum(n for t, n in self.memory.items() if (stem + c) in t)) - i * 1e-6
                for i, c in enumerate(choices)]

    def generate(self, prompt: str, max_new_tokens: int | None = None, stop_at_newline: bool = True) -> str:
        return "I do not know."

    def train(self, texts: Sequence[str], spec: TrainSpec, seed: int = 0) -> TrainResult:
        if spec.mode == "growth_only" and self.modules == 0:
            return TrainResult(0, None, None)
        self.memory.update(texts)
        self.train_calls.append((len(texts), spec.mode))
        return TrainResult(len(texts), 1.0, 0.5)

    def grow(self) -> bool:
        if self.modules >= self.capacity:
            return False
        self.modules += 1
        return True

    def active_modules(self) -> int:
        return self.modules

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"memory": dict(self.memory), "modules": self.modules}), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.memory = Counter(data["memory"])
        self.modules = data["modules"]


def small_config_dict(tmp_path: Path) -> dict:
    return {
        "run_name": "test-run",
        "runs_dir": str(tmp_path / "runs"),
        "seed": 3,
        "world": {"seed": 5, "starting_creatures": 2, "arrival_days": [3], "observations_per_day": 6},
        "sleep": {"rehearsal_pool": 4, "max_items": 60},
        "teacher": {"backend": "scripted", "lessons_per_day": 1},
        "curiosity": {"questions_per_day": 3},
        "probes": {"metacog_lags": [1, 2], "milestone_variants": 1},
        "growth": {"grow_loss_threshold": 0.1, "min_days_between_growth": 2},
    }


@pytest.fixture
def fake_learner() -> FakeLearner:
    return FakeLearner()


@pytest.fixture
def small_config(tmp_path: Path) -> Config:
    return config_from_dict(small_config_dict(tmp_path))
