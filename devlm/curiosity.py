"""Curiosity: the child picks what to ask about, from its own prediction error.

The child does not need to "reason" to be curious. Every observation gets a
surprise score (how badly the model predicted that sentence). The strategies:

* surprise          - ask about the most surprising things today.
* learning_progress - ask first about things never seen before, then about the
                      things whose surprise has been dropping fastest (what it
                      is currently getting better at). This is the
                      "learning progress" idea from intrinsic-motivation research
                      and avoids getting stuck on pure noise.
* random            - control condition.
"""

from __future__ import annotations

import random
from collections import defaultdict
from statistics import fmean
from typing import Iterable, Sequence

from devlm.config import CuriosityConfig
from devlm.memory import Episode


def topic_key(episode: Episode) -> str:
    if episode.fact_id:
        return episode.fact_id
    return f"event:{episode.meta.get('subject')}:{episode.meta.get('effect')}:{episode.meta.get('context')}"


def surprise_history(episodes: Iterable[Episode]) -> dict[str, list[tuple[int, float]]]:
    history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for e in episodes:
        if e.kind == "observation" and e.surprise is not None:
            history[topic_key(e)].append((e.day, float(e.surprise)))
    return dict(history)


def learning_progress(points: Sequence[tuple[int, float]], day: int, window: int) -> float | None:
    """Drop in mean surprise between the previous window and the latest window (positive = improving)."""
    recent = [s for d, s in points if day - window < d <= day]
    older = [s for d, s in points if day - 2 * window < d <= day - window]
    if not recent or not older:
        return None
    return fmean(older) - fmean(recent)


def select_observations(today: Sequence[Episode], past: Iterable[Episode], day: int,
                        cfg: CuriosityConfig, rng: random.Random) -> list[Episode]:
    """Choose up to cfg.questions_per_day of today's observations to ask the teacher about."""
    candidates: dict[str, Episode] = {}
    for e in today:
        if e.kind != "observation" or e.surprise is None:
            continue
        key = topic_key(e)
        if key not in candidates or e.surprise > candidates[key].surprise:
            candidates[key] = e
    pool = list(candidates.values())
    k = min(cfg.questions_per_day, len(pool))
    if k == 0:
        return []
    if cfg.strategy == "random":
        return rng.sample(pool, k)
    if cfg.strategy == "surprise":
        return sorted(pool, key=lambda e: (-e.surprise, e.id))[:k]
    history = surprise_history(past)
    unseen = sorted((e for e in pool if topic_key(e) not in history), key=lambda e: (-e.surprise, e.id))
    seen = [e for e in pool if topic_key(e) in history]

    def progress(e: Episode) -> float:
        points = history[topic_key(e)] + [(day, float(e.surprise))]
        value = learning_progress(points, day, cfg.window)
        return value if value is not None else 0.0

    ranked_seen = sorted(seen, key=lambda e: (-progress(e), -e.surprise, e.id))
    return (unseen + ranked_seen)[:k]


def why_question(episode: Episode) -> str:
    return f"Why did the {episode.meta.get('subject')} {episode.meta.get('effect')}?"
