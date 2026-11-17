"""Episodic memory: an append-only, timestamped log of everything the child experienced.

This is the "hippocampus". The weights are the "neocortex": sleep copies a
selection of these episodes into the weights, but the log itself keeps the
day and source of every memory, which weights cannot do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from devlm.run_files import append_jsonl, read_jsonl

EPISODE_KINDS = ("observation", "introduction", "question", "attempt", "teacher", "lesson", "autobio")


@dataclass(frozen=True)
class Episode:
    id: str
    day: int
    kind: str
    text: str
    source: str
    fact_id: str | None = None
    surprise: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"Unknown episode kind: {self.kind}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["meta"] = dict(self.meta)
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Episode":
        return Episode(
            id=str(data["id"]),
            day=int(data["day"]),
            kind=str(data["kind"]),
            text=str(data["text"]),
            source=str(data["source"]),
            fact_id=data.get("fact_id"),
            surprise=data.get("surprise"),
            meta=dict(data.get("meta") or {}),
        )


class EpisodeFactory:
    """Hands out unique, deterministic ids for one day's new episodes."""

    def __init__(self, day: int, start: int = 0) -> None:
        self._day = day
        self._count = start

    def make(self, kind: str, text: str, source: str, fact_id: str | None = None,
             surprise: float | None = None, meta: Mapping[str, Any] | None = None) -> Episode:
        self._count += 1
        return Episode(
            id=f"d{self._day:03d}-{self._count:04d}",
            day=self._day,
            kind=kind,
            text=text,
            source=source,
            fact_id=fact_id,
            surprise=surprise,
            meta=dict(meta or {}),
        )


class MemoryStore:
    """Append-only episode log backed by a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._episodes: tuple[Episode, ...] = tuple(Episode.from_dict(d) for d in read_jsonl(path))

    @property
    def episodes(self) -> tuple[Episode, ...]:
        return self._episodes

    def extend(self, episodes: Iterable[Episode]) -> None:
        new = tuple(episodes)
        existing = {e.id for e in self._episodes}
        clashes = [e.id for e in new if e.id in existing]
        if clashes:
            raise ValueError(f"Episode ids already stored: {clashes[:5]}")
        append_jsonl(self._path, (e.to_dict() for e in new))
        self._episodes = self._episodes + new

    def by_day(self, day: int) -> tuple[Episode, ...]:
        return tuple(e for e in self._episodes if e.day == day)


def first_taught_day(episodes: Iterable[Episode], fact_id: str) -> int | None:
    """The day the teacher first explained this fact (what the log says, not what the weights know)."""
    days = [e.day for e in episodes if e.fact_id == fact_id and e.kind in ("teacher", "lesson")]
    return min(days) if days else None
