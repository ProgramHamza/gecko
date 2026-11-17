import pytest

from devlm.run_files import RunPaths, append_jsonl, drop_records_after, read_json, read_jsonl, write_json


def test_jsonl_round_trip_and_missing_file(tmp_path):
    path = tmp_path / "x.jsonl"
    assert read_jsonl(path) == []
    append_jsonl(path, [{"day": 1}, {"day": 2}])
    append_jsonl(path, [])
    assert read_jsonl(path) == [{"day": 1}, {"day": 2}]


def test_corrupt_middle_line_is_an_error(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"day": 1}\n{broken\n{"day": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2 is not valid JSON"):
        read_jsonl(path)


def test_half_written_last_line_is_repaired_on_resume(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"day": 1}\n{"day": 2}\n{"day": 3, "te', encoding="utf-8")
    with pytest.raises(ValueError, match="half-written"):
        read_jsonl(path)
    assert drop_records_after(path, 1) == 2
    assert read_jsonl(path) == [{"day": 1}]
    assert drop_records_after(path, 1) == 0


def test_json_helpers_and_paths(tmp_path):
    path = tmp_path / "s.json"
    write_json(path, {"day": 3})
    assert read_json(path) == {"day": 3}
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "none.json")
    paths = RunPaths(tmp_path)
    assert paths.snapshot_days() == []
    paths.snapshots.mkdir()
    (paths.snapshots / "day_002.safetensors").write_text("x")
    (paths.snapshots / "notes.txt").write_text("x")
    assert paths.snapshot_days() == [2]
    with pytest.raises(ValueError):
        paths.log("nope")
