import random

from devlm.config import CuriosityConfig
from devlm.curiosity import learning_progress, select_observations, surprise_history, topic_key, why_question
from devlm.memory import EpisodeFactory


def obs(factory, fact_id, surprise, subject="zib", effect="hum"):
    return factory.make("observation", f"{subject} {effect}", "world", fact_id=fact_id, surprise=surprise,
                        meta={"subject": subject, "effect": effect, "context": "rainy"})


def test_surprise_strategy_picks_most_surprising_and_dedupes():
    f = EpisodeFactory(1)
    today = [obs(f, "a", 1.0), obs(f, "b", 5.0), obs(f, "b", 6.0), obs(f, "c", 3.0)]
    picked = select_observations(today, [], 1, CuriosityConfig(strategy="surprise", questions_per_day=2),
                                 random.Random(0))
    assert [e.fact_id for e in picked] == ["b", "c"]
    assert picked[0].surprise == 6.0


def test_learning_progress_prefers_unseen_then_improving():
    past = []
    for day, (sa, sb) in enumerate([(5, 5), (5, 5), (5, 5), (5, 2), (5, 2), (5, 2)], start=1):
        f = EpisodeFactory(day)
        past += [obs(f, "a", sa), obs(f, "b", sb)]
    f = EpisodeFactory(7)
    today = [obs(f, "a", 5.0), obs(f, "b", 2.0), obs(f, "new", 1.0)]
    cfg = CuriosityConfig(strategy="learning_progress", questions_per_day=3, window=3)
    picked = select_observations(today, past, 7, cfg, random.Random(0))
    assert [e.fact_id for e in picked] == ["new", "b", "a"]


def test_learning_progress_value():
    points = [(1, 6.0), (2, 6.0), (3, 2.0), (4, 2.0)]
    assert learning_progress(points, day=4, window=2) == 4.0
    assert learning_progress([(4, 1.0)], day=4, window=2) is None


def test_random_strategy_and_empty_pool():
    f = EpisodeFactory(1)
    today = [obs(f, str(i), float(i)) for i in range(5)]
    cfg = CuriosityConfig(strategy="random", questions_per_day=3)
    assert len(select_observations(today, [], 1, cfg, random.Random(1))) == 3
    assert select_observations([], [], 1, cfg, random.Random(1)) == []


def test_noise_events_have_their_own_topic_and_question():
    f = EpisodeFactory(1)
    noise = obs(f, None, 2.0, subject="mup", effect="spin around")
    assert topic_key(noise).startswith("event:mup")
    assert why_question(noise) == "Why did the mup spin around?"
    assert topic_key(noise) in surprise_history([noise])
