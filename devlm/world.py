"""A small persistent meadow world with invented creatures and hidden rules.

Everything is invented (names, rules) so the child model cannot already know
it from pretraining. The world is a pure function of its seed and the day
number: replaying a day always produces the same events.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from devlm.config import WorldConfig


@dataclass(frozen=True)
class Condition:
    key: str
    when: str          # "when it is rainy" / "at night"
    observed: str      # sentence describing the context; may contain {name}
    kind: str          # weather | time | interaction


@dataclass(frozen=True)
class Effect:
    base: str          # "turn blue"
    present: str       # "turns blue"
    past: str          # "turned blue"


WEATHERS = ("rainy", "sunny", "windy", "snowy")

CONDITIONS: tuple[Condition, ...] = (
    Condition("rainy", "when it is rainy", "It was rainy.", "weather"),
    Condition("sunny", "when it is sunny", "It was sunny.", "weather"),
    Condition("windy", "when it is windy", "It was windy.", "weather"),
    Condition("snowy", "when it is snowy", "It was snowy.", "weather"),
    Condition("night", "at night", "It was night.", "time"),
    Condition("fed", "when it is fed", "Someone fed the {name}.", "interaction"),
    Condition("touched", "when someone touches it", "Someone touched the {name}.", "interaction"),
)

EFFECTS: tuple[Effect, ...] = (
    Effect("glow", "glows", "glowed"),
    Effect("hum", "hums", "hummed"),
    Effect("turn blue", "turns blue", "turned blue"),
    Effect("float", "floats", "floated"),
    Effect("hide under a rock", "hides under a rock", "hid under a rock"),
    Effect("spin around", "spins around", "spun around"),
    Effect("fall asleep", "falls asleep", "fell asleep"),
    Effect("grow bigger", "grows bigger", "grew bigger"),
    Effect("sing", "sings", "sang"),
    Effect("jump", "jumps", "jumped"),
)

STILL = Effect("sit still", "sits still", "sat still")

CONDITION_BY_KEY = {c.key: c for c in CONDITIONS}
EFFECT_BY_BASE = {e.base: e for e in EFFECTS + (STILL,)}

_CONSONANTS = "bdfgkmnpstvz"
_VOWELS = "aeiou"
_ENDINGS = ("", "", "k", "m", "n", "z")
_REAL_WORDS = frozenset({
    "satin", "basin", "bison", "demon", "sumo", "tuna", "pasta", "bingo", "mango", "tango",
    "panda", "puma", "soda", "kimono", "tofu", "menu", "venom", "token", "seven", "given",
    "women", "dozen", "gamut", "nomad", "bonus", "sinus", "fetus", "fakir", "zebu", "dodo",
    "kudos", "begin", "bikini", "denim", "mini", "taxi", "vodka", "dime", "game", "time",
})


@dataclass(frozen=True)
class Creature:
    name: str
    introduced_day: int


@dataclass(frozen=True)
class Fact:
    """One hidden rule: under `condition_key`, `subject` does `effect_base`."""

    id: str
    subject: str
    condition_key: str
    effect_base: str
    introduced_day: int

    @property
    def condition(self) -> Condition:
        return CONDITION_BY_KEY[self.condition_key]

    @property
    def effect(self) -> Effect:
        return EFFECT_BY_BASE[self.effect_base]

    def statement(self) -> str:
        when = self.condition.when
        return f"{when[0].upper()}{when[1:]}, the {self.subject} {self.effect.present}."

    def statement_clause(self) -> str:
        """Lower-case clause without the final period, for embedding in sentences."""
        return self.statement()[0].lower() + self.statement()[1:-1]

    def question(self) -> str:
        return f"What does the {self.subject} do {self.condition.when}?"

    def cloze_prompt(self) -> str:
        when = self.condition.when
        return f"{when[0].upper()}{when[1:]}, the {self.subject}"


@dataclass(frozen=True)
class WorldEvent:
    text: str
    kind: str                  # observation | introduction
    subject: str
    effect_base: str | None
    context_key: str | None
    fact_id: str | None


@dataclass(frozen=True)
class DayEvents:
    day: int
    weather: str
    events: tuple[WorldEvent, ...]


@dataclass(frozen=True)
class World:
    seed: int
    creatures: tuple[Creature, ...]
    facts: tuple[Fact, ...]
    observations_per_day: int
    noise_ratio: float

    def present(self, day: int) -> tuple[Creature, ...]:
        return tuple(c for c in self.creatures if c.introduced_day <= day)

    def facts_for(self, subject: str) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.subject == subject)

    def introduced_facts(self, day: int) -> tuple[Fact, ...]:
        return tuple(f for f in self.facts if f.introduced_day <= day)

    def fact(self, fact_id: str) -> Fact:
        for f in self.facts:
            if f.id == fact_id:
                return f
        raise KeyError(f"Unknown fact id: {fact_id}")

    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.creatures)

    def day_events(self, day: int) -> DayEvents:
        rng = random.Random(self.seed * 100_003 + day)
        weather = rng.choice(WEATHERS)
        present = self.present(day)
        events = [_introduction(c) for c in present if c.introduced_day == day]
        if present:
            events.extend(self._moment(rng, weather, present) for _ in range(self.observations_per_day))
        return DayEvents(day=day, weather=weather, events=tuple(events))

    def _moment(self, rng: random.Random, weather: str, present: tuple[Creature, ...]) -> WorldEvent:
        creature = rng.choice(present)
        context = _pick_context(rng, weather)
        rules = {f.condition_key: f for f in self.facts_for(creature.name)}
        opener = CONDITION_BY_KEY[context].observed.format(name=creature.name)
        if context in rules:
            fact = rules[context]
            return _observation(opener, creature.name, fact.effect, context, fact.id)
        if rng.random() < self.noise_ratio:
            own = {f.effect_base for f in rules.values()}
            effect = rng.choice([e for e in EFFECTS if e.base not in own])
            return _observation(opener, creature.name, effect, context, None)
        return _observation(opener, creature.name, STILL, context, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "observations_per_day": self.observations_per_day,
            "noise_ratio": self.noise_ratio,
            "creatures": [vars(c) for c in self.creatures],
            "facts": [vars(f) for f in self.facts],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "World":
        try:
            return World(
                seed=int(data["seed"]),
                creatures=tuple(Creature(**c) for c in data["creatures"]),
                facts=tuple(Fact(**f) for f in data["facts"]),
                observations_per_day=int(data["observations_per_day"]),
                noise_ratio=float(data["noise_ratio"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"world.json is malformed: {exc}") from exc


def _pick_context(rng: random.Random, weather: str) -> str:
    roll = rng.random()
    if roll < 0.5:
        return weather
    if roll < 0.7:
        return "night"
    return rng.choice(("fed", "touched"))


def _observation(opener: str, name: str, effect: Effect, context: str, fact_id: str | None) -> WorldEvent:
    return WorldEvent(
        text=f"{opener} The {name} {effect.past}.",
        kind="observation",
        subject=name,
        effect_base=effect.base,
        context_key=context,
        fact_id=fact_id,
    )


def _introduction(creature: Creature) -> WorldEvent:
    return WorldEvent(
        text=f"A new creature came to the meadow. It was called the {creature.name}.",
        kind="introduction",
        subject=creature.name,
        effect_base=None,
        context_key=None,
        fact_id=None,
    )


def _invent_name(rng: random.Random, taken: set[str]) -> str:
    for _ in range(10_000):
        name = (rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)
                + rng.choice(_VOWELS) + rng.choice(_ENDINGS))
        if name not in taken and name not in _REAL_WORDS:
            return name
    raise RuntimeError("Could not invent a unique creature name")


def build_world(cfg: WorldConfig) -> World:
    """Create the world's creatures and hidden rules from the seed."""
    rng = random.Random(cfg.seed)
    arrival = [1] * cfg.starting_creatures + sorted(cfg.arrival_days)
    taken: set[str] = set()
    creatures: list[Creature] = []
    facts: list[Fact] = []
    for day in arrival:
        name = _invent_name(rng, taken)
        taken.add(name)
        creatures.append(Creature(name=name, introduced_day=day))
        conditions = rng.sample(CONDITIONS, cfg.rules_per_creature)
        effects = rng.sample(EFFECTS, cfg.rules_per_creature)
        facts.extend(
            Fact(id=f"{name}-{c.key}", subject=name, condition_key=c.key,
                 effect_base=e.base, introduced_day=day)
            for c, e in zip(conditions, effects)
        )
    return World(
        seed=cfg.seed,
        creatures=tuple(creatures),
        facts=tuple(facts),
        observations_per_day=cfg.observations_per_day,
        noise_ratio=cfg.noise_ratio,
    )
