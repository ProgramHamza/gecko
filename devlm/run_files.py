"""Where a run keeps its files, plus small JSON / JSONL helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

JSONL_LOGS = ("memory", "sleep", "probes", "metacog", "metrics")
_SNAPSHOT_RE = re.compile(r"^day_(\d{3,})\.safetensors$")


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def world(self) -> Path:
        return self.root / "world.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def rehearsal(self) -> Path:
        return self.root / "rehearsal.json"

    @property
    def memory(self) -> Path:
        return self.root / "memory.jsonl"

    @property
    def sleep(self) -> Path:
        return self.root / "sleep.jsonl"

    @property
    def probes(self) -> Path:
        return self.root / "probes.jsonl"

    @property
    def metacog(self) -> Path:
        return self.root / "metacog.jsonl"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.jsonl"

    @property
    def diary_dir(self) -> Path:
        return self.root / "diary"

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def report(self) -> Path:
        return self.root / "report.html"

    def diary(self, day: int) -> Path:
        return self.diary_dir / f"day_{day:03d}.md"

    def snapshot(self, day: int) -> Path:
        return self.snapshots / f"day_{day:03d}.safetensors"

    def log(self, name: str) -> Path:
        if name not in JSONL_LOGS:
            raise ValueError(f"Unknown log: {name}")
        return self.root / f"{name}.jsonl"

    def snapshot_days(self) -> list[int]:
        if not self.snapshots.exists():
            return []
        days = []
        for entry in self.snapshots.iterdir():
            match = _SNAPSHOT_RE.match(entry.name)
            if match:
                days.append(int(match.group(1)))
        return sorted(days)


def run_paths(runs_dir: str | Path, run_name: str) -> RunPaths:
    return RunPaths(Path(runs_dir) / run_name)


def append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(r), ensure_ascii=False) for r in records]
    if not lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _parse_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Returns (records, torn_tail). A broken *last* line (crash mid-write) is reported, not raised."""
    if not path.exists():
        return [], False
    lines = [(n, line) for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
             if line.strip()]
    records = []
    for index, (number, line) in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                return records, True
            raise ValueError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return records, False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records, torn = _parse_jsonl(path)
    if torn:
        raise ValueError(f"{path} ends with a half-written line; resume the run to repair it")
    return records


def drop_records_after(path: Path, day: int) -> int:
    """Remove records with record['day'] > day and any half-written last line (left by a crashed day)."""
    records, torn = _parse_jsonl(path)
    kept = [r for r in records if int(r.get("day", 0)) <= day]
    removed = len(records) - len(kept) + int(torn)
    if removed:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        os.replace(tmp, path)
    return removed


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing run file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
