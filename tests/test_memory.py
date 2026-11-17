import pytest

from devlm.memory import Episode, EpisodeFactory, MemoryStore, first_taught_day


def test_factory_ids_are_unique_and_dated():
    factory = EpisodeFactory(day=4)
    a = factory.make("observation", "x", "world")
    b = factory.make("teacher", "y", "scripted", fact_id="f1")
    assert (a.id, b.id) == ("d004-0001", "d004-0002")
    assert b.day == 4 and b.fact_id == "f1"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        Episode(id="x", day=1, kind="dream", text="", source="")


def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(path)
    factory = EpisodeFactory(1)
    store.extend([factory.make("observation", "It was rainy.", "world", surprise=3.2, meta={"subject": "zib"})])
    reloaded = MemoryStore(path)
    assert reloaded.episodes == store.episodes
    assert reloaded.episodes[0].meta["subject"] == "zib"
    assert reloaded.by_day(1) and not reloaded.by_day(2)


def test_duplicate_ids_are_refused(tmp_path):
    store = MemoryStore(tmp_path / "m.jsonl")
    episode = EpisodeFactory(1).make("observation", "a", "world")
    store.extend([episode])
    with pytest.raises(ValueError, match="already stored"):
        store.extend([episode])


def test_first_taught_day():
    f2, f5 = EpisodeFactory(2), EpisodeFactory(5)
    episodes = [f5.make("teacher", "t", "s", fact_id="a"), f2.make("lesson", "l", "s", fact_id="a"),
                f2.make("observation", "o", "world", fact_id="b")]
    assert first_taught_day(episodes, "a") == 2
    assert first_taught_day(episodes, "b") is None
