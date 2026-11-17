from devlm.config import WorldConfig
from devlm.world import World, _REAL_WORDS, build_world


def make(**kw):
    return build_world(WorldConfig(**kw))


def test_world_is_deterministic():
    assert make(seed=1) == make(seed=1)
    assert make(seed=1).names() != make(seed=2).names()


def test_creatures_and_facts():
    world = make(starting_creatures=3, arrival_days=(5, 9), rules_per_creature=2)
    assert len(world.creatures) == 5
    assert len(world.facts) == 10
    assert len(set(world.names())) == 5
    assert not set(world.names()) & _REAL_WORDS
    assert [c.introduced_day for c in world.creatures] == [1, 1, 1, 5, 9]
    for creature in world.creatures:
        effects = [f.effect_base for f in world.facts_for(creature.name)]
        assert len(effects) == len(set(effects))


def test_introduced_facts_follow_arrivals():
    world = make(starting_creatures=1, arrival_days=(4,))
    assert len(world.introduced_facts(3)) == 2
    assert len(world.introduced_facts(4)) == 4


def test_day_events_are_reproducible_and_consistent():
    world = make()
    assert world.day_events(3) == world.day_events(3)
    for day in range(1, 12):
        for ev in world.day_events(day).events:
            if ev.fact_id:
                fact = world.fact(ev.fact_id)
                assert fact.subject == ev.subject
                assert fact.effect_base == ev.effect_base
                assert fact.condition_key == ev.context_key
                assert f"The {ev.subject} {fact.effect.past}." in ev.text


def test_arrival_day_has_an_introduction():
    world = make(starting_creatures=1, arrival_days=(4,))
    intros = [e for e in world.day_events(4).events if e.kind == "introduction"]
    assert [e.subject for e in intros] == [world.creatures[1].name]
    assert all(e.kind != "introduction" for e in world.day_events(5).events)


def test_absent_creatures_never_appear():
    world = make(starting_creatures=1, arrival_days=(6,))
    late = world.creatures[1].name
    for day in range(1, 6):
        assert all(e.subject != late for e in world.day_events(day).events)


def test_round_trip():
    world = make()
    assert World.from_dict(world.to_dict()) == world


def test_sentences():
    fact = make().facts[0]
    assert fact.statement().endswith(".") and fact.subject in fact.statement()
    assert fact.question().startswith("What does the ")
    assert fact.statement().startswith(fact.cloze_prompt())
    assert fact.statement_clause()[0].islower() and not fact.statement_clause().endswith(".")
