from devlm.config import WorldConfig
from devlm.probes import (MILESTONES, ProbeItem, ProbeResult, fact_probes, known_facts, milestone_probes,
                          milestone_scores, run_probes, score_item)
from devlm.world import build_world

WORLD = build_world(WorldConfig(seed=11))


class FixedScores:
    def __init__(self, scores):
        self.scores = scores

    def score_choices(self, prompt, choices):
        return self.scores[: len(choices)]


def test_two_probes_per_fact_with_the_right_answer():
    items = fact_probes(WORLD)
    assert len(items) == 2 * len(WORLD.facts)
    for item in items:
        fact = WORLD.fact(item.fact_id)
        assert item.choices[item.answer] == f" {fact.effect.present}."
        assert len(set(item.choices)) == len(item.choices) == 4
    assert fact_probes(WORLD) == items


def test_qa_and_cloze_prompts():
    qa, cloze = fact_probes(WORLD)[:2]
    fact = WORLD.facts[0]
    assert qa.prompt == f"Q: {fact.question()}\nA: It"
    assert cloze.prompt == fact.cloze_prompt()


def test_milestones_cover_every_template():
    items = milestone_probes(WORLD, variants=2)
    assert len(items) == 2 * len(MILESTONES)
    assert all("{" not in i.prompt for i in items)


def test_score_item_marks_correct_and_margin():
    item = ProbeItem("x", "fact", "p", (" a.", " b.", " c."), answer=1, fact_id="f")
    good = score_item(FixedScores([0.1, 0.9, 0.3]), item)
    assert good.correct and abs(good.margin - 0.6) < 1e-9
    bad = score_item(FixedScores([0.9, 0.1, 0.3]), item)
    assert not bad.correct and bad.chosen == 0
    assert len(run_probes(FixedScores([1, 0, 0]), [item, item])) == 2


def test_fact_known_only_if_every_probe_is_right():
    results = [ProbeResult("a/qa", "fact", "a", True, 1, 0), ProbeResult("a/cloze", "fact", "a", True, 1, 0),
               ProbeResult("b/qa", "fact", "b", True, 1, 0), ProbeResult("b/cloze", "fact", "b", False, -1, 1)]
    assert known_facts(results) == {"a": True, "b": False}
    assert known_facts([r.to_record(3) for r in results]) == {"a": True, "b": False}


def test_milestone_scores():
    results = [ProbeResult("o/0", "milestone:object_permanence", None, True, 1, 0),
               ProbeResult("o/1", "milestone:object_permanence", None, False, -1, 1)]
    assert milestone_scores(results) == {"object_permanence": 0.5}
