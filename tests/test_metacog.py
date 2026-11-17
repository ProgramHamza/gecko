from devlm import metacog
from devlm.config import WorldConfig
from devlm.memory import EpisodeFactory
from devlm.world import build_world

WORLD = build_world(WorldConfig(seed=11, starting_creatures=1, arrival_days=(5,)))


def records(fact_id, by_day):
    return [{"day": d, "fact_id": fact_id, "correct": c, "item_id": f"{fact_id}/{i}"}
            for d, values in by_day.items() for i, c in enumerate(values)]


def test_knowledge_table_requires_all_probes():
    table = metacog.knowledge_table(records("a", {1: [True, False], 2: [True, True]}))
    assert table == {"a": {1: False, 2: True}}


def test_first_known_newly_known_and_forgotten():
    table = {"a": {1: False, 2: True, 3: True, 4: False}, "b": {1: False, 2: False, 3: True, 4: True}}
    assert metacog.first_known_day(table, "a") == 2
    assert metacog.first_known_day(table, "a", after=3) == 3
    assert metacog.first_known_day(table, "zzz") is None
    assert metacog.newly_known(table, 3) == ["b"]
    assert metacog.newly_forgotten(table, 4) == ["a"]


def test_balanced_accuracy_punishes_always_no():
    pairs = [(False, False)] * 9 + [(False, True)]
    assert metacog.balanced_accuracy(pairs) == 0.5
    assert metacog.balanced_accuracy([(True, True), (False, False)]) == 1.0
    assert metacog.balanced_accuracy([]) is None


def test_day_choices_contain_truth():
    choices, answer = metacog.day_choices(3, 9, "k")
    assert choices[answer] == " 3." and len(choices) == 4
    assert metacog.day_choices(1, 1, "k") == ((" 1.",), 0)


class AlwaysYes:
    def score_choices(self, prompt, choices):
        return [1.0] + [0.0] * (len(choices) - 1)


def test_judge_lags_skips_days_before_arrival_and_uses_log():
    early, late = WORLD.facts[0], WORLD.facts[2]
    table = {early.id: {1: False, 2: True, 3: True, 4: True, 5: True, 6: True}}
    episodes = [EpisodeFactory(2).make("teacher", "t", "scripted", fact_id=early.id)]
    judgements = metacog.judge_lags(AlwaysYes(), WORLD, table, episodes, day=6, lags=[1, 5])
    assert all(j.fact_id != late.id or 6 - j.lag >= 5 for j in judgements)
    by_key = {(j.fact_id, j.lag): j for j in judgements}
    assert by_key[(early.id, 5)].truth is False and by_key[(early.id, 5)].log_answer is False
    assert by_key[(early.id, 1)].truth is True and by_key[(early.id, 1)].log_answer is True


def test_evaluate_summary_shape():
    fact = WORLD.facts[0]
    table = {fact.id: {1: False, 2: True, 3: True}}
    summary = metacog.evaluate(AlwaysYes(), WORLD, table, [], day=3, lags=[1, 2])
    assert summary["day"] == 3 and summary["n_lag_questions"] > 0
    assert summary["n_when_questions"] == 1
    assert set(summary["per_lag"]) == {"1", "2"}


def test_texts():
    fact = WORLD.facts[0]
    assert metacog.autobiographical_text(fact, 4).startswith("On day 4, I learned that ")
    assert metacog.when_question(fact).endswith("A: On day")
    assert "3 days ago" in metacog.lag_question(fact, 9, 3)
