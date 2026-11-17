"""Human-readable diary: one markdown page per simulated day."""

from __future__ import annotations

from typing import Mapping, Sequence

from devlm.memory import Episode
from devlm.world import World

TOP_OBSERVATIONS = 5


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _observations(episodes: Sequence[Episode]) -> list[str]:
    seen = [e for e in episodes if e.kind in ("observation", "introduction")]
    lines = [f"- {e.text}" for e in seen if e.kind == "introduction"]
    ranked = sorted((e for e in seen if e.kind == "observation"), key=lambda e: -(e.surprise or 0.0))
    lines += [f"- {e.text} _(surprise {_fmt(e.surprise)})_" for e in ranked[:TOP_OBSERVATIONS]]
    return lines or ["- (nothing)"]


def _dialogue(episodes: Sequence[Episode]) -> list[str]:
    lines = []
    attempts = {e.meta.get("question"): e.text for e in episodes if e.kind == "attempt"}
    for e in episodes:
        if e.kind == "teacher":
            question = e.meta.get("question", "")
            lines += [f"- **I asked:** {question}",
                      f"  **My guess:** {attempts.get(question) or '...'}",
                      f"  **Teacher ({e.source}):** {e.text}"]
        elif e.kind == "lesson":
            lines.append(f"- **Lesson ({e.source}):** {e.text}")
    return lines or ["- (no questions today)"]


def _sleep(record: Mapping | None) -> list[str]:
    if not record:
        return ["- (no sleep)"]
    if record.get("skipped"):
        return ["- Nothing new to consolidate."]
    lines = [
        f"- Replayed {record['n_new']} new memories, {record['n_replay']} old ones and "
        f"{record['n_rehearsal']} of my own old stories ({record['steps']} training steps).",
        f"- Surprise on the new memories: {_fmt(record['loss_before'])} -> {_fmt(record['loss_after'])}",
    ]
    if record.get("grew"):
        lines.append(f"- I grew a new module ({record['modules_active']} active now).")
    return lines


def render_day(day: int, weather: str | None, episodes: Sequence[Episode], sleep_record: Mapping | None,
               metrics: Mapping, world: World) -> str:
    known = [world.fact(f).statement() for f in metrics.get("newly_known", [])]
    forgotten = [world.fact(f).statement() for f in metrics.get("forgotten", [])]
    milestones = ", ".join(f"{k} {_fmt(v)}" for k, v in (metrics.get("milestones") or {}).items())
    parts = [
        f"# Day {day}",
        f"Weather: {weather or '-'}",
        "",
        "## What I noticed",
        *_observations(episodes),
        "",
        "## What I asked",
        *_dialogue(episodes),
        "",
        "## Sleep",
        *_sleep(sleep_record),
        "",
        "## What I know now",
        f"- Facts known: {metrics.get('facts_known')}/{metrics.get('facts_introduced')}",
        f"- Newly learned: {'; '.join(known) if known else 'nothing'}",
        f"- Forgotten: {'; '.join(forgotten) if forgotten else 'nothing'}",
        f"- Milestones: {milestones or '-'}",
        f"- Self-knowledge (balanced accuracy, weights / log): "
        f"{_fmt(metrics.get('metacog_weights'))} / {_fmt(metrics.get('metacog_log'))}",
        "",
    ]
    return "\n".join(parts)
