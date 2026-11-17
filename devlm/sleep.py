"""Sleep: copy selected memories into the weights.

Each night the child replays:
* new salient memories (teacher answers, lessons, autobiographical notes,
  and the more surprising observations) that are not consolidated yet,
* a sample of older consolidated memories (replay, against forgetting),
* stories its own day-0 self wrote (pseudo-rehearsal, to keep its language).

After training, if new memories are still badly predicted, it may recruit a
new growth module and train that module on them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from devlm.config import GrowthConfig, SleepConfig
from devlm.memory import Episode
from devlm.protocols import Learner, TrainSpec

SALIENT_KINDS = ("teacher", "lesson", "autobio", "introduction")


@dataclass(frozen=True)
class SleepPlan:
    new_texts: tuple[str, ...]
    replay_texts: tuple[str, ...]
    rehearsal_texts: tuple[str, ...]
    episode_ids: tuple[str, ...]

    @property
    def all_texts(self) -> tuple[str, ...]:
        return self.new_texts + self.replay_texts + self.rehearsal_texts


def training_texts(e: Episode) -> list[str]:
    """How one memory is rehearsed. Q/A pairs teach the format the probes use."""
    if e.kind == "teacher":
        question = e.meta.get("question")
        return [f"Q: {question}\nA: {e.text}"] if question else [e.text]
    if e.kind == "autobio":
        when = e.meta.get("when_question")
        return [e.text] + ([f"{when} {e.meta.get('day_learned')}."] if when else [])
    if e.kind in ("lesson", "introduction", "observation"):
        return [e.text]
    return []


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("inf")
    index = min(len(ordered) - 1, max(0, math.floor(q * (len(ordered) - 1) + 0.5)))
    return ordered[index]


def plan_sleep(episodes: Sequence[Episode], consolidated: Iterable[str], cfg: SleepConfig,
               rehearsal_pool: Sequence[str], rng: random.Random, day: int | None = None) -> SleepPlan:
    """If `day` is given, only that day's observations are candidates (old unpicked ones are not revisited)."""
    done = set(consolidated)
    fresh = [e for e in episodes if e.id not in done]
    salient = [e for e in fresh if e.kind in SALIENT_KINDS]
    observations = [e for e in fresh if e.kind == "observation" and e.surprise is not None
                    and (day is None or e.day == day)]
    threshold = quantile([e.surprise for e in observations], cfg.surprise_quantile)
    chosen = salient + [e for e in observations if e.surprise >= threshold]

    pairs = [(e.id, t) for e in chosen for t in training_texts(e)][: cfg.max_items]
    new_texts = tuple(t for _, t in pairs)
    ids = tuple(dict.fromkeys(i for i, _ in pairs))

    budget = cfg.max_items - len(new_texts)
    old = [t for e in episodes if e.id in done and e.kind in SALIENT_KINDS for t in training_texts(e)]
    replay = _sample(old, min(budget, round(len(new_texts) * cfg.replay_ratio)), rng)
    budget -= len(replay)
    rehearsal = _sample(list(rehearsal_pool), min(budget, round(len(new_texts) * cfg.rehearsal_ratio)), rng)
    return SleepPlan(new_texts, tuple(replay), tuple(rehearsal), ids)


def _sample(items: list[str], k: int, rng: random.Random) -> list[str]:
    k = max(0, min(k, len(items)))
    return rng.sample(items, k)


def should_grow(loss_after: float, cfg: GrowthConfig, day: int, last_growth_day: int | None) -> bool:
    if not cfg.enabled or loss_after <= cfg.grow_loss_threshold:
        return False
    return last_growth_day is None or day - last_growth_day >= cfg.min_days_between_growth


def _mean_surprise(learner: Learner, texts: Sequence[str]) -> float:
    return fmean(learner.surprise(texts))


def run_sleep(learner: Learner, plan: SleepPlan, sleep_cfg: SleepConfig, growth_cfg: GrowthConfig,
              day: int, last_growth_day: int | None, seed: int) -> dict:
    record = {"day": day, "n_new": len(plan.new_texts), "n_replay": len(plan.replay_texts),
              "n_rehearsal": len(plan.rehearsal_texts), "episode_ids": list(plan.episode_ids),
              "grew": False, "steps": 0, "loss_before": None, "loss_after": None}
    if not plan.new_texts:
        return {**record, "modules_active": learner.active_modules(), "skipped": True}

    spec = TrainSpec.from_sleep(sleep_cfg)
    grew = False
    if spec.mode == "growth_only" and learner.active_modules() == 0:
        grew = learner.grow()
        last_growth_day = day
    loss_before = _mean_surprise(learner, plan.new_texts)
    result = learner.train(plan.all_texts, spec, seed=seed)
    loss_after = _mean_surprise(learner, plan.new_texts)
    steps = result.steps
    if should_grow(loss_after, growth_cfg, day, last_growth_day) and learner.grow():
        grew = True
        extra = learner.train(plan.new_texts + plan.replay_texts,
                              TrainSpec.from_sleep(sleep_cfg, mode="growth_only"), seed=seed + 1)
        steps += extra.steps
        loss_after = _mean_surprise(learner, plan.new_texts)
    return {**record, "grew": grew, "steps": steps, "loss_before": round(loss_before, 4),
            "loss_after": round(loss_after, 4), "modules_active": learner.active_modules(), "skipped": False}
