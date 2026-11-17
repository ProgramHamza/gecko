"""Probe battery: multiple-choice tests scored by likelihood, so even a model
that cannot follow instructions can be tested. Run on every day's weights.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from devlm.protocols import Learner
from devlm.world import EFFECTS, World

DISTRACTORS = 3

# (template, choices, index of the correct choice). {a} and {b} are creature names.
MILESTONES: dict[str, tuple[str, tuple[str, ...], int]] = {
    "object_permanence": (
        "The {a} put a shiny stone in the box and closed the lid. Later, the stone was",
        (" in the box.", " gone forever.", " up in the sky."), 0),
    "false_belief": (
        "The {a} put a berry in the basket and went away. Then the {b} moved the berry to the cup. "
        "When the {a} came back, the {a} looked for the berry in the",
        (" basket.", " cup."), 0),
    "conservation": (
        "There were three pebbles in a row. The {a} spread them far apart. Now there were",
        (" three pebbles.", " more pebbles.", " fewer pebbles."), 0),
    "causal_induction": (
        "Every time it was windy, the {a} jumped. Today it was windy, so the {a}",
        (" jumped.", " slept.", " cried."), 0),
}


@dataclass(frozen=True)
class ProbeItem:
    id: str
    category: str              # fact | milestone:<name>
    prompt: str
    choices: tuple[str, ...]
    answer: int
    fact_id: str | None = None


@dataclass(frozen=True)
class ProbeResult:
    item_id: str
    category: str
    fact_id: str | None
    correct: bool
    margin: float
    chosen: int

    def to_record(self, day: int) -> dict:
        return {"day": day, "item_id": self.item_id, "category": self.category, "fact_id": self.fact_id,
                "correct": self.correct, "margin": round(self.margin, 4), "chosen": self.chosen}


def stable_rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def fact_probes(world: World) -> tuple[ProbeItem, ...]:
    """Two phrasings per fact; a fact counts as known only if both are answered correctly."""
    items = []
    for fact in world.facts:
        rng = stable_rng(fact.id)
        others = [e for e in EFFECTS if e.base != fact.effect_base]
        options = [fact.effect] + rng.sample(others, DISTRACTORS)
        rng.shuffle(options)
        choices = tuple(f" {e.present}." for e in options)
        answer = options.index(fact.effect)
        items.append(ProbeItem(f"{fact.id}/qa", "fact", f"Q: {fact.question()}\nA: It", choices, answer, fact.id))
        items.append(ProbeItem(f"{fact.id}/cloze", "fact", fact.cloze_prompt(), choices, answer, fact.id))
    return tuple(items)


def milestone_probes(world: World, variants: int) -> tuple[ProbeItem, ...]:
    names = world.names()
    items = []
    for name, (template, choices, answer) in MILESTONES.items():
        for v in range(variants):
            a, b = names[v % len(names)], names[(v + 1) % len(names)]
            prompt = template.format(a=a, b=b)
            items.append(ProbeItem(f"{name}/{v}", f"milestone:{name}", prompt, choices, answer))
    return tuple(items)


def score_item(learner: Learner, item: ProbeItem) -> ProbeResult:
    scores = learner.score_choices(item.prompt, item.choices)
    chosen = max(range(len(scores)), key=lambda i: scores[i])
    rival = max(s for i, s in enumerate(scores) if i != item.answer)
    return ProbeResult(item.id, item.category, item.fact_id, chosen == item.answer,
                       scores[item.answer] - rival, chosen)


def run_probes(learner: Learner, items: Sequence[ProbeItem]) -> list[ProbeResult]:
    return [score_item(learner, item) for item in items]


def known_facts(results: Iterable[ProbeResult | Mapping]) -> dict[str, bool]:
    """fact_id -> True if every probe for that fact was answered correctly."""
    grouped: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        fact_id = r.fact_id if isinstance(r, ProbeResult) else r.get("fact_id")
        correct = r.correct if isinstance(r, ProbeResult) else bool(r.get("correct"))
        if fact_id:
            grouped[fact_id].append(correct)
    return {f: all(v) for f, v in grouped.items()}


def milestone_scores(results: Iterable[ProbeResult]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        if r.category.startswith("milestone:"):
            grouped[r.category.split(":", 1)[1]].append(r.correct)
    return {name: sum(v) / len(v) for name, v in grouped.items()}
