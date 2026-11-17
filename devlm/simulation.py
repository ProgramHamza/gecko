"""The daily loop: perceive -> wonder -> ask -> sleep -> get tested -> remember learning it."""

from __future__ import annotations

import logging
import random
from statistics import fmean
from typing import Sequence

from devlm import metacog
from devlm.config import Config, save_config
from devlm.curiosity import select_observations, why_question
from devlm.diary import render_day
from devlm.memory import Episode, EpisodeFactory, MemoryStore
from devlm.probes import fact_probes, known_facts, milestone_probes, milestone_scores, run_probes
from devlm.protocols import Learner
from devlm.run_files import JSONL_LOGS, RunPaths, append_jsonl, drop_records_after, read_json, read_jsonl, write_json
from devlm.sleep import plan_sleep, run_sleep
from devlm.teacher import Question, Teacher, pick_lesson_facts
from devlm.world import DayEvents, World, build_world

log = logging.getLogger(__name__)

REHEARSAL_PROMPTS = ("Once upon a time,", "One day, a little", "Lily and Tom were", "The little dog",
                     "There was a big", "Mom said,")


class RunExistsError(RuntimeError):
    pass


def make_rehearsal(learner: Learner, count: int, max_new_tokens: int) -> list[str]:
    """Stories written by the untouched day-0 model, replayed later so it keeps its language."""
    stories = []
    for i in range(count):
        prompt = REHEARSAL_PROMPTS[i % len(REHEARSAL_PROMPTS)]
        story = learner.generate(prompt, max_new_tokens=max_new_tokens, stop_at_newline=False)
        if story:
            stories.append(f"{prompt} {story}")
    return stories


def initialise_run(cfg: Config, paths: RunPaths, learner: Learner) -> World:
    """Create the run folder, the world, the rehearsal set, and the day-0 snapshot and tests."""
    if paths.state.exists():
        raise RunExistsError(f"A run already exists at {paths.root}. Pick another run_name or delete that folder.")
    paths.root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, paths.config)
    world = build_world(cfg.world)
    write_json(paths.world, world.to_dict())
    log.info("Writing %d rehearsal stories with the day-0 model...", cfg.sleep.rehearsal_pool)
    write_json(paths.rehearsal, make_rehearsal(learner, cfg.sleep.rehearsal_pool, cfg.child.generation_tokens * 2))
    Simulation(cfg, paths, world, learner, teacher=None).close_day(0, None, [], EpisodeFactory(0), None)
    return world


class Simulation:
    def __init__(self, cfg: Config, paths: RunPaths, world: World, learner: Learner,
                 teacher: Teacher | None) -> None:
        self.cfg = cfg
        self.paths = paths
        self.world = world
        self.learner = learner
        self.teacher = teacher
        self.memory = MemoryStore(paths.memory)
        self.probe_items = fact_probes(world) + milestone_probes(world, cfg.probes.milestone_variants)
        self.rehearsal: list[str] = read_json(paths.rehearsal) if paths.rehearsal.exists() else []

    @staticmethod
    def open(cfg: Config, paths: RunPaths, learner: Learner, teacher: Teacher) -> "Simulation":
        world = World.from_dict(read_json(paths.world))
        return Simulation(cfg, paths, world, learner, teacher)

    def current_day(self) -> int:
        return int(read_json(self.paths.state)["day"])

    # ---- driver -----------------------------------------------------------

    def run(self, days: int) -> int:
        if self.teacher is None:
            raise RuntimeError("A teacher is required to run days")
        start = self.current_day()
        removed = sum(drop_records_after(self.paths.log(name), start) for name in JSONL_LOGS)
        if removed:
            log.warning("Discarded %d records from an unfinished day %d", removed, start + 1)
            self.memory = MemoryStore(self.paths.memory)
        self.learner.load(self.paths.snapshot(start))
        self.teacher.check()
        for day in range(start + 1, start + days + 1):
            self.run_day(day)
        return start + days

    def run_day(self, day: int) -> dict:
        rng = random.Random(self.cfg.seed * 7919 + day)
        factory = EpisodeFactory(day)
        events = self.world.day_events(day)
        observed = self._perceive(events, factory)
        new = observed + self._ask(observed, day, factory, rng) + self._lessons(day, factory, rng)
        sleep_records = read_jsonl(self.paths.sleep)
        consolidated = [i for r in sleep_records for i in r.get("episode_ids", [])]
        grown_days = [int(r["day"]) for r in sleep_records if r.get("grew")]
        plan = plan_sleep(self.memory.episodes + tuple(new), consolidated, self.cfg.sleep, self.rehearsal, rng,
                          day=day)
        sleep_record = run_sleep(self.learner, plan, self.cfg.sleep, self.cfg.growth, day,
                                 max(grown_days) if grown_days else None, seed=self.cfg.seed + day)
        return self.close_day(day, events, new, factory, sleep_record)

    # ---- daytime ----------------------------------------------------------

    def _perceive(self, events: DayEvents, factory: EpisodeFactory) -> list[Episode]:
        texts = [ev.text for ev in events.events]
        surprises = self.learner.surprise(texts) if texts else []
        return [
            factory.make(ev.kind, ev.text, "world", fact_id=ev.fact_id, surprise=round(s, 4),
                         meta={"subject": ev.subject, "effect": ev.effect_base, "context": ev.context_key,
                               "weather": events.weather})
            for ev, s in zip(events.events, surprises)
        ]

    def _ask(self, observed: Sequence[Episode], day: int, factory: EpisodeFactory,
             rng: random.Random) -> list[Episode]:
        picked = select_observations(observed, self.memory.episodes, day, self.cfg.curiosity, rng)
        episodes = []
        for obs in picked:
            question = why_question(obs)
            attempt = self.learner.generate(f"Q: {question}\nA:")
            fact = self.world.fact(obs.fact_id) if obs.fact_id else None
            answer = self.teacher.explain(Question(question, attempt, str(obs.meta["subject"]), fact),
                                          self.world, day)
            episodes += [
                factory.make("question", question, "child", obs.fact_id, meta={"about": obs.id}),
                factory.make("attempt", attempt, "child", obs.fact_id, meta={"question": question}),
                factory.make("teacher", answer.text, answer.source, obs.fact_id, meta={"question": question}),
            ]
        return episodes

    def _lessons(self, day: int, factory: EpisodeFactory, rng: random.Random) -> list[Episode]:
        taught = [e.fact_id for e in self.memory.episodes if e.kind in ("teacher", "lesson") and e.fact_id]
        facts = pick_lesson_facts(self.world, day, taught, self.cfg.teacher.lessons_per_day, rng)
        episodes = []
        for fact in facts:
            answer = self.teacher.lesson(fact, self.world, day)
            episodes.append(factory.make("lesson", answer.text, answer.source, fact.id))
        return episodes

    # ---- evening: test, remember, record ----------------------------------

    def close_day(self, day: int, events: DayEvents | None, new: list[Episode], factory: EpisodeFactory,
                  sleep_record: dict | None) -> dict:
        self.learner.save(self.paths.snapshot(day))
        results = run_probes(self.learner, self.probe_items)
        probe_records = [r.to_record(day) for r in results]
        table = metacog.knowledge_table(read_jsonl(self.paths.probes) + probe_records)
        autobio = self._autobiography(day, table, factory) if day > 0 and self.cfg.sleep.autobiographical else []
        all_new = new + autobio
        self_knowledge = (metacog.evaluate(self.learner, self.world, table, self.memory.episodes + tuple(all_new),
                                           day, self.cfg.probes.metacog_lags) if day > 0 else None)
        metrics = self._metrics(day, results, table, sleep_record, self_knowledge, new)

        self.memory.extend(all_new)
        append_jsonl(self.paths.sleep, [sleep_record] if sleep_record else [])
        append_jsonl(self.paths.probes, probe_records)
        append_jsonl(self.paths.metacog, [self_knowledge] if self_knowledge else [])
        append_jsonl(self.paths.metrics, [metrics])
        self.paths.diary_dir.mkdir(parents=True, exist_ok=True)
        self.paths.diary(day).write_text(
            render_day(day, events.weather if events else None, all_new, sleep_record, metrics, self.world),
            encoding="utf-8")
        write_json(self.paths.state, {"day": day})
        self._prune_snapshots(day)
        log.info("Day %d: knows %s/%s facts, sleep loss %s -> %s, modules %s", day, metrics["facts_known"],
                 metrics["facts_introduced"], metrics["sleep_loss_before"], metrics["sleep_loss_after"],
                 metrics["modules_active"])
        return metrics

    def _autobiography(self, day: int, table: metacog.KnowledgeTable, factory: EpisodeFactory) -> list[Episode]:
        episodes = []
        for fact_id in metacog.newly_known(table, day):
            fact = self.world.fact(fact_id)
            if fact.introduced_day > day:
                continue
            episodes.append(factory.make(
                "autobio", metacog.autobiographical_text(fact, day), "self", fact_id,
                meta={"when_question": metacog.when_question(fact), "day_learned": day}))
        return episodes

    def _metrics(self, day: int, results, table: metacog.KnowledgeTable, sleep_record: dict | None,
                 self_knowledge: dict | None, new: Sequence[Episode]) -> dict:
        introduced = {f.id for f in self.world.introduced_facts(day)}
        known = known_facts(results)
        fact_results = [r for r in results if r.category == "fact"]
        seen = [r.correct for r in fact_results if r.fact_id in introduced]
        unseen = [r.correct for r in fact_results if r.fact_id not in introduced]
        surprises = [e.surprise for e in new if e.kind == "observation" and e.surprise is not None]
        return {
            "day": day,
            "facts_introduced": len(introduced),
            "facts_known": sum(1 for f in introduced if known.get(f)),
            "fact_accuracy": round(fmean(seen), 4) if seen else None,
            "control_accuracy": round(fmean(unseen), 4) if unseen else None,
            "milestones": milestone_scores(results),
            "mean_surprise": round(fmean(surprises), 4) if surprises else None,
            "sleep_loss_before": sleep_record.get("loss_before") if sleep_record else None,
            "sleep_loss_after": sleep_record.get("loss_after") if sleep_record else None,
            "grew": bool(sleep_record and sleep_record.get("grew")),
            "modules_active": self.learner.active_modules(),
            "teacher_fallbacks": sum(1 for e in new if e.source == "fallback"),
            "newly_known": [f for f in metacog.newly_known(table, day) if f in introduced] if day > 0 else [],
            "forgotten": metacog.newly_forgotten(table, day) if day > 0 else [],
            "metacog_weights": self_knowledge.get("weights_balanced_accuracy") if self_knowledge else None,
            "metacog_log": self_knowledge.get("log_balanced_accuracy") if self_knowledge else None,
            "when_accuracy": self_knowledge.get("when_accuracy") if self_knowledge else None,
        }

    def _prune_snapshots(self, today: int) -> None:
        every = self.cfg.snapshots.keep_every
        for day in self.paths.snapshot_days():
            if day in (0, today) or day % every == 0:
                continue
            self.paths.snapshot(day).unlink(missing_ok=True)
