"""The teacher: a bigger model (or a script) that answers the child's questions.

The teacher knows the world's true rules. Every answer from an LLM teacher is
checked against those rules; if it is empty, too long, or does not state the
rule, the scripted answer is used instead (and the fallback is logged).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import requests

from devlm.config import TeacherConfig
from devlm.world import Fact, World

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a warm, patient caregiver talking to a very young child who is learning \
about a small meadow. The meadow has invented creatures. These are the TRUE rules of the meadow:
{rules}

How to answer:
- Use 1 to 3 very short sentences with small, simple words, like a picture book.
- Always say the true rule plainly and use the creature's name.
- If the child's guess was wrong, gently say so.
- If something has no rule, say it just happens sometimes.
- Never invent rules that are not in the list. Do not use lists, markdown, or emojis."""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class TeacherError(RuntimeError):
    """The teacher backend failed (network, API error, refusal)."""


@dataclass(frozen=True)
class Answer:
    text: str
    source: str                # scripted | ollama | anthropic | fallback


@dataclass(frozen=True)
class Question:
    question: str
    child_attempt: str
    subject: str
    fact: Fact | None


class Teacher(Protocol):
    def check(self) -> None: ...
    def explain(self, q: Question, world: World, day: int) -> Answer: ...
    def lesson(self, fact: Fact, world: World, day: int) -> Answer: ...


def scripted_explanation(q: Question) -> str:
    if q.fact is None:
        return f"Good question! Sometimes the {q.subject} just does that. There is no rule for it."
    return f"Good question! {q.fact.statement()}"


def scripted_lesson(fact: Fact) -> str:
    return f"Let me tell you about the {fact.subject}. {fact.statement()}"


class ScriptedTeacher:
    """Rule-based teacher: free, offline, always correct. Also the safety net for LLM teachers."""

    def check(self) -> None:
        return None

    def explain(self, q: Question, world: World, day: int) -> Answer:
        return Answer(scripted_explanation(q), "scripted")

    def lesson(self, fact: Fact, world: World, day: int) -> Answer:
        return Answer(scripted_lesson(fact), "scripted")


def clean_reply(text: str, max_words: int) -> str:
    text = _THINK_RE.sub("", text).replace("*", "").strip()
    text = " ".join(text.split())
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words])
    end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    return clipped[:end + 1] if end > 0 else clipped


def states_fact(text: str, fact: Fact) -> bool:
    lowered = text.lower()
    effect_word = fact.effect.base.split(" ")[0]          # "turn blue" -> "turn"
    variants = {effect_word, fact.effect.present.split(" ")[0], fact.effect.past.split(" ")[0]}
    return fact.subject in lowered and any(v in lowered for v in variants)


class LLMTeacher:
    """Shared prompting + validation; subclasses implement _complete()."""

    source = "llm"

    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg
        self.fallbacks = 0

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def check(self) -> None:
        self._complete("Reply with the single word: ready", "Are you ready?")

    def _system(self, world: World, day: int) -> str:
        rules = "\n".join(f"- {f.statement()}" for f in world.introduced_facts(day))
        return SYSTEM_PROMPT.format(rules=rules or "- (no creatures yet)")

    def _ask(self, system: str, user: str, fact: Fact | None, fallback: str) -> Answer:
        try:
            text = clean_reply(self._complete(system, user), self.cfg.max_words)
        except TeacherError as exc:
            log.warning("Teacher failed, using scripted answer: %s", exc)
            self.fallbacks += 1
            return Answer(fallback, "fallback")
        if not text or (fact is not None and not states_fact(text, fact)):
            log.warning("Teacher answer did not state the rule, using scripted answer: %r", text)
            self.fallbacks += 1
            return Answer(fallback, "fallback")
        return Answer(text, self.source)

    def explain(self, q: Question, world: World, day: int) -> Answer:
        hint = (f"(The true rule here: {q.fact.statement()})" if q.fact
                else "(There is no rule for this event; it just happened.)")
        user = f'The child asked: "{q.question}"\nThe child guessed: "{q.child_attempt or "..."}"\n{hint}'
        return self._ask(self._system(world, day), user, q.fact, scripted_explanation(q))

    def lesson(self, fact: Fact, world: World, day: int) -> Answer:
        user = f"Tell the child a tiny story (2 or 3 sentences) that teaches this rule: {fact.statement()}"
        return self._ask(self._system(world, day), user, fact, scripted_lesson(fact))


class OllamaTeacher(LLMTeacher):
    """Local model served by Ollama (https://ollama.com). Free, runs on your GPU."""

    source = "ollama"

    def _complete(self, system: str, user: str) -> str:
        url = self.cfg.ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.7},
        }
        try:
            response = requests.post(url, json=payload, timeout=self.cfg.timeout_seconds)
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except requests.ConnectionError as exc:
            raise TeacherError(f"Cannot reach Ollama at {self.cfg.ollama_url}. Is `ollama serve` running?") from exc
        except requests.HTTPError as exc:
            body = exc.response.text[:200] if exc.response is not None else ""
            raise TeacherError(f"Ollama returned an error ({exc}). Did you `ollama pull {self.cfg.model}`? {body}") from exc
        except requests.RequestException as exc:
            raise TeacherError(f"Ollama request failed: {exc}") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise TeacherError(f"Unexpected Ollama response: {exc}") from exc


class AnthropicTeacher(LLMTeacher):
    """Claude via the Anthropic API. Needs ANTHROPIC_API_KEY (or `ant auth login`)."""

    source = "anthropic"

    def __init__(self, cfg: TeacherConfig) -> None:
        super().__init__(cfg)
        try:
            import anthropic
        except ImportError as exc:
            raise TeacherError("The anthropic package is not installed: pip install -r requirements-claude.txt") from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(timeout=cfg.timeout_seconds)

    def _complete(self, system: str, user: str) -> str:
        anthropic = self._anthropic
        try:
            response = self._client.beta.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.cfg.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise TeacherError("Anthropic API key is missing or invalid (set ANTHROPIC_API_KEY in .env)") from exc
        except anthropic.RateLimitError as exc:
            raise TeacherError("Anthropic rate limit hit; try again later") from exc
        except anthropic.APIStatusError as exc:
            raise TeacherError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise TeacherError("Cannot reach the Anthropic API (network problem)") from exc
        except TypeError as exc:
            raise TeacherError(f"The installed anthropic SDK is too old ({exc}); run: "
                               "pip install --upgrade -r requirements-claude.txt") from exc
        if response.stop_reason == "refusal":
            raise TeacherError("Claude declined to answer this request")
        return "".join(block.text for block in response.content if block.type == "text")


def make_teacher(cfg: TeacherConfig) -> Teacher:
    if cfg.backend == "scripted":
        return ScriptedTeacher()
    if cfg.backend == "ollama":
        return OllamaTeacher(cfg)
    if cfg.backend == "anthropic":
        return AnthropicTeacher(cfg)
    raise ValueError(f"Unknown teacher backend: {cfg.backend}")


def pick_lesson_facts(world: World, day: int, taught: Sequence[str], count: int, rng) -> list[Fact]:
    """Prefer facts the teacher has never explained yet."""
    available = list(world.introduced_facts(day))
    fresh = [f for f in available if f.id not in set(taught)]
    pool = fresh or available
    return rng.sample(pool, min(count, len(pool)))
