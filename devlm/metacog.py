"""Metacognition: "would I have known this N days ago?"

Ground truth comes from the probe history: on every day the real weights
were tested, so we know exactly which facts were known when. The child is
then asked about its own past, in two ways:

* weights-only  - a likelihood-scored question to the model itself.
* log-based     - read the episodic log: "was I taught this before that day?"
                  (Being taught is not the same as knowing, so this can be wrong too.)

It is also asked "when did I learn X?", which it can only answer from
autobiographical memories consolidated during sleep.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from devlm.memory import Episode, first_taught_day
from devlm.probes import stable_rng
from devlm.protocols import Learner
from devlm.world import Fact, World

YES, NO = " Yes.", " No."
DAY_CHOICES = 4

KnowledgeTable = dict[str, dict[int, bool]]


def knowledge_table(probe_records: Iterable[Mapping]) -> KnowledgeTable:
    """fact_id -> {day: known}. Known = every probe for the fact correct that day."""
    grouped: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for r in probe_records:
        if r.get("fact_id"):
            grouped[(r["fact_id"], int(r["day"]))].append(bool(r["correct"]))
    table: KnowledgeTable = defaultdict(dict)
    for (fact_id, day), values in grouped.items():
        table[fact_id][day] = all(values)
    return dict(table)


def first_known_day(table: KnowledgeTable, fact_id: str, after: int = 0) -> int | None:
    days = sorted(d for d, known in table.get(fact_id, {}).items() if known and d >= after)
    return days[0] if days else None


def newly_known(table: KnowledgeTable, day: int) -> list[str]:
    return sorted(f for f, days in table.items() if days.get(day) and not days.get(day - 1, False))


def newly_forgotten(table: KnowledgeTable, day: int) -> list[str]:
    return sorted(f for f, days in table.items() if days.get(day - 1) and days.get(day) is False)


def autobiographical_text(fact: Fact, day: int) -> str:
    return f"On day {day}, I learned that {fact.statement_clause()}."


def when_question(fact: Fact) -> str:
    return f"Q: When did I learn that {fact.statement_clause()}?\nA: On day"


def lag_question(fact: Fact, day: int, lag: int) -> str:
    return f"Today is day {day}. Q: {lag} days ago, did I know that {fact.statement_clause()}?\nA:"


@dataclass(frozen=True)
class LagJudgement:
    fact_id: str
    lag: int
    truth: bool
    weights_answer: bool
    log_answer: bool


def day_choices(true_day: int, today: int, key: str) -> tuple[tuple[str, ...], int]:
    rng = stable_rng(f"{key}:{today}")
    pool = [d for d in range(1, today + 1) if d != true_day]
    days = sorted([true_day] + rng.sample(pool, min(DAY_CHOICES - 1, len(pool))))
    return tuple(f" {d}." for d in days), days.index(true_day)


def judge_lags(learner: Learner, world: World, table: KnowledgeTable, episodes: Sequence[Episode],
               day: int, lags: Sequence[int]) -> list[LagJudgement]:
    judgements = []
    for fact in world.introduced_facts(day):
        for lag in lags:
            past = day - lag
            if past < fact.introduced_day:
                continue
            truth = table.get(fact.id, {}).get(past, False)
            yes, no = learner.score_choices(lag_question(fact, day, lag), (YES, NO))
            taught = first_taught_day(episodes, fact.id)
            judgements.append(LagJudgement(fact.id, lag, truth, yes > no, taught is not None and taught <= past))
    return judgements


def judge_when_learned(learner: Learner, world: World, table: KnowledgeTable, day: int) -> list[bool]:
    """For each known fact with a learned day, did the model pick the right day?"""
    outcomes = []
    for fact in world.introduced_facts(day):
        learned = first_known_day(table, fact.id, after=fact.introduced_day)
        if learned is None or learned >= day or day < 2:
            continue
        choices, answer = day_choices(learned, day, fact.id)
        scores = learner.score_choices(when_question(fact), choices)
        outcomes.append(max(range(len(scores)), key=lambda i: scores[i]) == answer)
    return outcomes


def balanced_accuracy(pairs: Iterable[tuple[bool, bool]]) -> float | None:
    """Mean of per-class recall, so 'always say no' does not look clever. pairs = (prediction, truth)."""
    pairs = list(pairs)
    positives = [p for p, t in pairs if t]
    negatives = [p for p, t in pairs if not t]
    recalls = []
    if positives:
        recalls.append(sum(positives) / len(positives))
    if negatives:
        recalls.append(sum(1 for p in negatives if not p) / len(negatives))
    return sum(recalls) / len(recalls) if recalls else None


def evaluate(learner: Learner, world: World, table: KnowledgeTable, episodes: Sequence[Episode],
             day: int, lags: Sequence[int]) -> dict:
    lag_results = judge_lags(learner, world, table, episodes, day, lags)
    when_results = judge_when_learned(learner, world, table, day)
    return {
        "day": day,
        "n_lag_questions": len(lag_results),
        "weights_balanced_accuracy": balanced_accuracy((j.weights_answer, j.truth) for j in lag_results),
        "log_balanced_accuracy": balanced_accuracy((j.log_answer, j.truth) for j in lag_results),
        "n_when_questions": len(when_results),
        "when_accuracy": (sum(when_results) / len(when_results)) if when_results else None,
        "per_lag": {
            str(lag): balanced_accuracy((j.weights_answer, j.truth) for j in lag_results if j.lag == lag)
            for lag in lags
        },
    }
