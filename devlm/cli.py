"""Command line: python -m devlm <command>."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
from pathlib import Path
from typing import Sequence

from devlm.config import Config, ConfigError, load_config, validate
from devlm.protocols import Learner
from devlm.run_files import RunPaths, read_jsonl, run_paths

log = logging.getLogger("devlm")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines, does not override variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def make_learner(cfg: Config) -> Learner:
    from devlm.child import TinyChild  # torch is imported only when a model is needed
    return TinyChild.from_pretrained(cfg.child, cfg.growth, seed=cfg.seed)


def _base(args) -> tuple[Config, RunPaths]:
    cfg = load_config(Path(args.config))
    if args.run:
        cfg = validate(dataclasses.replace(cfg, run_name=args.run))
    return cfg, run_paths(cfg.runs_dir, cfg.run_name)


def _existing(args) -> tuple[Config, RunPaths]:
    _, paths = _base(args)
    if not paths.state.exists():
        raise FileNotFoundError(f"No run at {paths.root}. Create one with: ./run.sh start")
    return load_config(paths.config), paths


def _prompt(text: str) -> tuple[str, bool]:
    """'!text' = raw continuation; anything else is asked as a question."""
    if text.startswith("!"):
        return text[1:], False
    return f"Q: {text}\nA:", True


def cmd_download(args) -> None:
    cfg, _ = _base(args)
    learner = make_learner(cfg)
    print(f"Model ready: {cfg.child.model_name} ({learner.parameter_count() / 1e6:.1f}M parameters) "
          f"on {learner.device}")


def cmd_init(args) -> None:
    from devlm.simulation import initialise_run
    cfg, paths = _base(args)
    world = initialise_run(cfg, paths, make_learner(cfg))
    print(f"Created run at {paths.root}")
    for creature in world.creatures:
        print(f"  {creature.name} arrives on day {creature.introduced_day}")


def cmd_run(args) -> None:
    from devlm.report import write_report
    from devlm.simulation import Simulation
    from devlm.teacher import make_teacher
    cfg, paths = _existing(args)
    sim = Simulation.open(cfg, paths, make_learner(cfg), make_teacher(cfg.teacher))
    last = sim.run(args.days)
    write_report(paths)
    print(f"Finished day {last}. Report: {paths.report}")


def cmd_start(args) -> None:
    """Create the run if it does not exist yet, then live N more days."""
    _, paths = _base(args)
    if not paths.state.exists():
        cmd_init(args)
    cmd_run(args)


def cmd_chat(args) -> None:
    cfg, paths = _existing(args)
    day = args.day if args.day is not None else int(read_jsonl(paths.metrics)[-1]["day"])
    learner = make_learner(cfg)
    learner.load(paths.snapshot(day))
    print(f"Talking to the day-{day} child. Type a question, '!text' to let it continue text, 'quit' to stop.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit"):
            break
        if text:
            prompt, is_question = _prompt(text)
            print("child>", learner.generate(prompt, stop_at_newline=is_question))


def _softmax(scores: Sequence[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


def cmd_compare(args) -> None:
    cfg, paths = _existing(args)
    days = [int(d) for d in args.days.split(",")] if args.days else paths.snapshot_days()
    available = set(paths.snapshot_days())
    missing = [d for d in days if d not in available]
    if missing:
        raise FileNotFoundError(f"No snapshot for day(s) {missing}. Available: {sorted(available)}")
    choices = [c if c.startswith(" ") else " " + c for c in args.choices.split("|")] if args.choices else None
    prompt, is_question = _prompt(args.question)
    learner = make_learner(cfg)
    for day in days:
        learner.load(paths.snapshot(day))
        if choices:
            probs = _softmax(learner.score_choices(prompt, choices))
            answer = "  ".join(f"{c.strip()} {p:.0%}" for c, p in zip(choices, probs))
        else:
            answer = learner.generate(prompt, stop_at_newline=is_question)
        print(f"day {day:>3}: {answer}")


def cmd_report(args) -> None:
    from devlm.report import write_report
    _, paths = _existing(args)
    write_report(paths)
    print(f"Report written: {paths.report}")


def cmd_status(args) -> None:
    _, paths = _existing(args)
    last = read_jsonl(paths.metrics)[-1]
    print(f"Run {paths.root.name}: day {last['day']}, knows {last['facts_known']}/{last['facts_introduced']} facts, "
          f"{last['modules_active']} growth modules, snapshots for days {paths.snapshot_days()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="devlm", description="A developmental language model sandbox.")
    parser.add_argument("--config", default="config.yaml", help="config file (default: config.yaml)")
    parser.add_argument("--run", default=None, help="run name (overrides run_name in the config)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download the child model and tokenizer").set_defaults(func=cmd_download)
    sub.add_parser("init", help="create a new run (world, day-0 snapshot, day-0 tests)").set_defaults(func=cmd_init)
    run = sub.add_parser("run", help="live N more days")
    run.add_argument("--days", type=int, default=10)
    run.set_defaults(func=cmd_run)
    start = sub.add_parser("start", help="init the run if needed, then live N more days")
    start.add_argument("--days", type=int, default=10)
    start.set_defaults(func=cmd_start)
    chat = sub.add_parser("chat", help="talk to the child as it was on a given day")
    chat.add_argument("--day", type=int, default=None)
    chat.set_defaults(func=cmd_chat)
    compare = sub.add_parser("compare", help="ask every saved day the same thing")
    compare.add_argument("question")
    compare.add_argument("--choices", default=None, help="score fixed answers, e.g. 'hums.|glows.|floats.'")
    compare.add_argument("--days", default=None, help="comma-separated days (default: all snapshots)")
    compare.set_defaults(func=cmd_compare)
    sub.add_parser("report", help="rebuild report.html").set_defaults(func=cmd_report)
    sub.add_parser("status", help="print where the run is").set_defaults(func=cmd_status)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    args = build_parser().parse_args(argv)
    if getattr(args, "days", None) is not None and isinstance(args.days, int) and args.days < 1:
        print("error: --days must be >= 1", file=sys.stderr)
        return 2
    try:
        args.func(args)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
