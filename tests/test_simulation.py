import dataclasses

import pytest

from devlm.config import SnapshotConfig
from devlm.memory import MemoryStore
from devlm.report import build_report, learning_speed, write_report
from devlm.run_files import append_jsonl, read_json, read_jsonl, run_paths
from devlm.simulation import RunExistsError, Simulation, initialise_run
from devlm.teacher import ScriptedTeacher

from conftest import FakeLearner


def start(cfg, learner=None):
    paths = run_paths(cfg.runs_dir, cfg.run_name)
    learner = learner or FakeLearner()
    initialise_run(cfg, paths, learner)
    return paths, Simulation.open(cfg, paths, learner, ScriptedTeacher())


def test_init_writes_day_zero(small_config):
    paths, _ = start(small_config)
    assert read_json(paths.state) == {"day": 0}
    assert paths.snapshot(0).exists() and paths.world.exists() and paths.config.exists()
    assert len(read_json(paths.rehearsal)) == small_config.sleep.rehearsal_pool
    metrics = read_jsonl(paths.metrics)
    assert metrics[0]["day"] == 0 and metrics[0]["facts_known"] == 0
    with pytest.raises(RunExistsError):
        initialise_run(small_config, paths, FakeLearner())


def test_days_run_end_to_end_and_child_learns(small_config):
    paths, sim = start(small_config)
    assert sim.run(6) == 6
    assert read_json(paths.state) == {"day": 6}
    metrics = read_jsonl(paths.metrics)
    assert [m["day"] for m in metrics] == list(range(7))
    assert metrics[-1]["facts_known"] > 0
    assert metrics[-1]["facts_introduced"] == len(sim.world.facts)
    kinds = {e.kind for e in MemoryStore(paths.memory).episodes}
    assert {"observation", "question", "attempt", "teacher", "lesson", "autobio", "introduction"} <= kinds
    assert len(read_jsonl(paths.sleep)) == 6
    assert any(r["grew"] for r in read_jsonl(paths.sleep))
    assert read_jsonl(paths.metacog)[-1]["day"] == 6
    assert "## Sleep" in paths.diary(6).read_text(encoding="utf-8")


def test_autobiography_is_written_for_newly_known_facts(small_config):
    paths, sim = start(small_config)
    sim.run(4)
    autobio = [e for e in MemoryStore(paths.memory).episodes if e.kind == "autobio"]
    assert autobio
    assert all(e.text.startswith(f"On day {e.day}, I learned that") for e in autobio)


def test_resume_continues_from_saved_day(small_config):
    paths, sim = start(small_config)
    sim.run(2)
    again = Simulation.open(small_config, paths, FakeLearner(), ScriptedTeacher())
    assert again.run(1) == 3
    assert [m["day"] for m in read_jsonl(paths.metrics)] == [0, 1, 2, 3]


def test_unfinished_day_is_discarded_on_resume(small_config):
    paths, sim = start(small_config)
    sim.run(1)
    append_jsonl(paths.metrics, [{"day": 2, "facts_known": 99}])
    append_jsonl(paths.memory, [{"id": "d002-0001", "day": 2, "kind": "observation", "text": "x", "source": "w"}])
    again = Simulation.open(small_config, paths, FakeLearner(), ScriptedTeacher())
    again.run(1)
    assert [m["day"] for m in read_jsonl(paths.metrics)] == [0, 1, 2]
    assert all(m["facts_known"] != 99 for m in read_jsonl(paths.metrics))


def test_snapshots_are_pruned(small_config):
    cfg = dataclasses.replace(small_config, snapshots=SnapshotConfig(keep_every=2))
    paths, sim = start(cfg)
    sim.run(5)
    assert paths.snapshot_days() == [0, 2, 4, 5]


def test_report_has_every_section(small_config):
    paths, sim = start(small_config)
    sim.run(4)
    page = build_report(paths)
    for heading in ("Learning curve", "Knowledge map", "Learning to learn", "Developmental milestones",
                    "Self-knowledge", "Growth", "Diary"):
        assert heading in page
    write_report(paths)
    assert paths.report.exists()
    rows = learning_speed(sim.world, {})
    assert len(rows) == len(sim.world.creatures) and rows[0]["mean_days"] is None


def test_run_requires_teacher(small_config):
    paths = run_paths(small_config.runs_dir, small_config.run_name)
    initialise_run(small_config, paths, FakeLearner())
    with pytest.raises(RuntimeError, match="teacher"):
        Simulation.open(small_config, paths, FakeLearner(), None).run(1)
