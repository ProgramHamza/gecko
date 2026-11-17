import random

from devlm.config import GrowthConfig, SleepConfig
from devlm.memory import EpisodeFactory
from devlm.sleep import plan_sleep, quantile, run_sleep, should_grow, training_texts

from conftest import FakeLearner


def test_training_text_formats():
    f = EpisodeFactory(1)
    teacher = f.make("teacher", "When it is rainy, the zib hums.", "scripted", "zib-rainy",
                     meta={"question": "Why did the zib hum?"})
    assert training_texts(teacher) == ["Q: Why did the zib hum?\nA: When it is rainy, the zib hums."]
    autobio = f.make("autobio", "On day 1, I learned that x.", "self", "zib-rainy",
                     meta={"when_question": "Q: When did I learn that x?\nA: On day", "day_learned": 1})
    assert training_texts(autobio)[1] == "Q: When did I learn that x?\nA: On day 1."
    assert training_texts(f.make("attempt", "dunno", "child")) == []


def test_quantile():
    assert quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert quantile([], 0.5) == float("inf")


def test_plan_takes_unconsolidated_salient_and_surprising_only():
    old, new = EpisodeFactory(1), EpisodeFactory(2)
    consolidated = old.make("lesson", "old lesson", "scripted")
    episodes = [
        consolidated,
        new.make("teacher", "answer", "scripted", meta={"question": "q?"}),
        new.make("observation", "boring", "world", surprise=1.0),
        new.make("observation", "wow", "world", surprise=9.0),
        new.make("attempt", "guess", "child"),
    ]
    cfg = SleepConfig(replay_ratio=1.0, rehearsal_ratio=1.0, surprise_quantile=0.5)
    plan = plan_sleep(episodes, [consolidated.id], cfg, ["story one", "story two"], random.Random(0))
    assert "Q: q?\nA: answer" in plan.new_texts and "wow" in plan.new_texts
    assert "boring" not in plan.new_texts and "guess" not in plan.new_texts
    assert plan.replay_texts == ("old lesson",)
    assert len(plan.rehearsal_texts) == 2
    assert consolidated.id not in plan.episode_ids


def test_plan_respects_max_items():
    f = EpisodeFactory(1)
    episodes = [f.make("lesson", f"lesson {i}", "scripted") for i in range(30)]
    plan = plan_sleep(episodes, [], SleepConfig(max_items=10), ["s"] * 50, random.Random(0))
    assert len(plan.all_texts) <= 10
    assert len(plan.episode_ids) == 10


def test_should_grow():
    cfg = GrowthConfig(grow_loss_threshold=2.0, min_days_between_growth=3)
    assert should_grow(3.0, cfg, day=5, last_growth_day=None)
    assert not should_grow(1.0, cfg, day=5, last_growth_day=None)
    assert not should_grow(3.0, cfg, day=5, last_growth_day=4)
    assert not should_grow(3.0, GrowthConfig(enabled=False), day=5, last_growth_day=None)


def _plan():
    f = EpisodeFactory(1)
    return plan_sleep([f.make("lesson", "new thing", "scripted")], [], SleepConfig(), [], random.Random(0))


def test_run_sleep_trains_and_grows_when_still_surprised():
    learner = FakeLearner()
    record = run_sleep(learner, _plan(), SleepConfig(), GrowthConfig(grow_loss_threshold=0.1), 1, None, seed=0)
    assert record["grew"] and record["modules_active"] == 1
    assert record["loss_after"] < record["loss_before"]
    assert [mode for _, mode in learner.train_calls] == ["full", "growth_only"]


def test_run_sleep_skips_when_nothing_new():
    plan = plan_sleep([], [], SleepConfig(), [], random.Random(0))
    record = run_sleep(FakeLearner(), plan, SleepConfig(), GrowthConfig(), 1, None, seed=0)
    assert record["skipped"] and record["steps"] == 0


def test_growth_only_mode_grows_first_module_automatically():
    learner = FakeLearner()
    record = run_sleep(learner, _plan(), SleepConfig(train_mode="growth_only"),
                       GrowthConfig(grow_loss_threshold=100.0), 1, None, seed=0)
    assert learner.modules == 1 and record["grew"] and record["steps"] > 0


def test_plan_with_day_ignores_old_unpicked_observations():
    old, new = EpisodeFactory(1), EpisodeFactory(2)
    episodes = [old.make("observation", "stale", "world", surprise=9.0),
                new.make("observation", "today", "world", surprise=5.0)]
    plan = plan_sleep(episodes, [], SleepConfig(surprise_quantile=0.0), [], random.Random(0), day=2)
    assert plan.new_texts == ("today",)
