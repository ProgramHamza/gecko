"""Static HTML report for a run: growth charts, a knowledge map, and the diary."""

from __future__ import annotations

import html
from collections import defaultdict
from statistics import fmean
from typing import Mapping, Sequence

from devlm import metacog
from devlm.run_files import RunPaths, read_json, read_jsonl
from devlm.world import World

W, H, PAD = 640, 220, 36
COLORS = ("var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)")
DIARY_DAYS = 14

CSS = """
:root { --bg:#fbfaf7; --fg:#1f2328; --muted:#6b7280; --grid:#e5e2da; --card:#ffffff;
  --known:#2f9e6b; --unknown:#e9e5dc; --absent:transparent;
  --c1:#3b6fd8; --c2:#d9822b; --c3:#2f9e6b; --c4:#a855c7; --c5:#c2410c; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181c; --fg:#e7e7e4; --muted:#9aa0a6; --grid:#2c3036;
  --card:#1d2025; --known:#3fbf85; --unknown:#2c3036; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 32px 0 8px; }
.muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--grid); border-radius: 10px; padding: 14px; overflow-x: auto; }
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.stat b { display:block; font-size: 22px; }
svg { width: 100%; height: auto; max-width: 640px; }
svg text { fill: var(--muted); font-size: 11px; }
.legend span { display:inline-block; margin-right: 14px; font-size: 13px; }
.legend i { display:inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; }
table.map { border-collapse: collapse; font-size: 12px; }
table.map td { padding: 0; } table.map td.label { padding-right: 10px; white-space: nowrap; }
table.map td.cell { width: 10px; height: 14px; border: 1px solid var(--bg); }
.k1 { background: var(--known); } .k0 { background: var(--unknown); } .kx { background: var(--absent); }
table.plain { border-collapse: collapse; } table.plain td, table.plain th { padding: 4px 12px 4px 0; text-align: left; }
details { margin: 6px 0; } pre { white-space: pre-wrap; font-size: 13px; }
"""


def _series_points(metrics: Sequence[Mapping], key: str) -> list[tuple[int, float]]:
    return [(int(m["day"]), float(m[key])) for m in metrics if m.get(key) is not None]


def line_chart(series: Mapping[str, Sequence[tuple[int, float]]], y_max: float = 1.0) -> str:
    all_days = [d for pts in series.values() for d, _ in pts]
    if not all_days:
        return "<p class='muted'>No data yet.</p>"
    x_max = max(max(all_days), 1)
    sx = lambda d: PAD + (W - 2 * PAD) * d / x_max          # noqa: E731
    sy = lambda v: H - PAD - (H - 2 * PAD) * min(v, y_max) / y_max  # noqa: E731
    grid = "".join(
        f"<line x1='{PAD}' x2='{W - PAD}' y1='{sy(t * y_max):.1f}' y2='{sy(t * y_max):.1f}' stroke='var(--grid)'/>"
        f"<text x='4' y='{sy(t * y_max) + 4:.1f}'>{t * y_max:g}</text>" for t in (0, 0.5, 1))
    lines, legend = [], []
    for (name, pts), color in zip(series.items(), COLORS):
        if pts:
            path = " ".join(f"{sx(d):.1f},{sy(v):.1f}" for d, v in pts)
            lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{path}'/>")
        legend.append(f"<span><i style='background:{color}'></i>{html.escape(name)}</span>")
    axis = f"<text x='{W - PAD}' y='{H - 10}' text-anchor='end'>day {x_max}</text><text x='{PAD}' y='{H - 10}'>day 0</text>"
    return (f"<svg viewBox='0 0 {W} {H}' role='img'>{grid}{''.join(lines)}{axis}</svg>"
            f"<div class='legend'>{''.join(legend)}</div>")


def knowledge_map(world: World, table: metacog.KnowledgeTable, last_day: int) -> str:
    rows = []
    for fact in world.facts:
        cells = []
        for day in range(last_day + 1):
            if day < fact.introduced_day:
                cls = "kx"
            else:
                cls = "k1" if table.get(fact.id, {}).get(day) else "k0"
            cells.append(f"<td class='cell {cls}' title='day {day}'></td>")
        rows.append(f"<tr><td class='label'>{html.escape(fact.statement())}</td>{''.join(cells)}</tr>")
    return f"<table class='map'>{''.join(rows)}</table>"


def learning_speed(world: World, table: metacog.KnowledgeTable) -> list[dict]:
    """Days from a creature's arrival until each of its facts was first known. Smaller later = learning to learn."""
    rows = []
    for creature in world.creatures:
        delays = []
        for fact in world.facts_for(creature.name):
            learned = metacog.first_known_day(table, fact.id, after=creature.introduced_day)
            if learned is not None:
                delays.append(learned - creature.introduced_day)
        rows.append({"name": creature.name, "arrived": creature.introduced_day,
                     "learned": len(delays), "total": len(world.facts_for(creature.name)),
                     "mean_days": round(fmean(delays), 1) if delays else None})
    return rows


def _speed_table(rows: Sequence[Mapping]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['arrived']}</td><td>{r['learned']}/{r['total']}</td>"
        f"<td>{'-' if r['mean_days'] is None else r['mean_days']}</td></tr>" for r in rows)
    return ("<table class='plain'><tr><th>Creature</th><th>Arrived</th><th>Facts learned</th>"
            f"<th>Mean days to learn</th></tr>{body}</table>")


def _milestone_series(metrics: Sequence[Mapping]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for m in metrics:
        for name, value in (m.get("milestones") or {}).items():
            series[name].append((int(m["day"]), float(value)))
    return dict(series)


def _stats(metrics: Sequence[Mapping]) -> str:
    last = metrics[-1]
    items = [("Day", last["day"]), ("Facts known", f"{last['facts_known']}/{last['facts_introduced']}"),
             ("Growth modules", last["modules_active"]),
             ("Teacher fallbacks", sum(m.get("teacher_fallbacks", 0) for m in metrics))]
    return "<div class='stats'>" + "".join(
        f"<div class='card stat'><span class='muted'>{k}</span><b>{html.escape(str(v))}</b></div>"
        for k, v in items) + "</div>"


def _diary(paths: RunPaths, last_day: int) -> str:
    blocks = []
    for day in range(last_day, max(-1, last_day - DIARY_DAYS), -1):
        page = paths.diary(day)
        if page.exists():
            text = html.escape(page.read_text(encoding="utf-8"))
            blocks.append(f"<details{' open' if day == last_day else ''}><summary>Day {day}</summary><pre>{text}</pre></details>")
    return "".join(blocks) or "<p class='muted'>No diary yet.</p>"


def build_report(paths: RunPaths) -> str:
    metrics = read_jsonl(paths.metrics)
    if not metrics:
        raise ValueError(f"No metrics in {paths.root}; run `init` first")
    world = World.from_dict(read_json(paths.world))
    table = metacog.knowledge_table(read_jsonl(paths.probes))
    last_day = int(metrics[-1]["day"])
    growth = [f"day {r['day']} (now {r['modules_active']} active)" for r in read_jsonl(paths.sleep) if r.get("grew")]
    sections = [
        f"<h1>{html.escape(paths.root.name)}</h1><p class='muted'>Developmental LM run report</p>",
        _stats(metrics),
        "<h2>Learning curve</h2><div class='card'>" + line_chart({
            "accuracy on introduced facts": _series_points(metrics, "fact_accuracy"),
            "control: facts not met yet (chance level)": _series_points(metrics, "control_accuracy"),
        }) + "</div>",
        "<h2>Knowledge map</h2><p class='muted'>Green = known that day (both probes right), "
        "grey = not known, blank = creature had not arrived yet.</p>"
        f"<div class='card'>{knowledge_map(world, table, last_day)}</div>",
        "<h2>Learning to learn</h2><p class='muted'>If later arrivals are learned in fewer days, "
        "learning itself is getting better, not just the store of facts.</p>"
        f"<div class='card'>{_speed_table(learning_speed(world, table))}</div>",
        "<h2>Developmental milestones</h2><div class='card'>" + line_chart(_milestone_series(metrics)) + "</div>",
        "<h2>Self-knowledge</h2><p class='muted'>Balanced accuracy at \"did I know this N days ago?\" "
        "(0.5 = guessing), and accuracy at \"when did I learn this?\".</p><div class='card'>" + line_chart({
            "from weights only": _series_points(metrics, "metacog_weights"),
            "from the memory log": _series_points(metrics, "metacog_log"),
            "when did I learn it": _series_points(metrics, "when_accuracy"),
        }) + "</div>",
        "<h2>Growth</h2><div class='card'>" + (", ".join(growth) or "No modules grown yet.") + "</div>",
        f"<h2>Diary</h2>{_diary(paths, last_day)}",
    ]
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Run {html.escape(paths.root.name)}</title><style>{CSS}</style></head>"
            f"<body><main>{''.join(sections)}</main></body></html>")


def write_report(paths: RunPaths) -> None:
    paths.report.write_text(build_report(paths), encoding="utf-8")
