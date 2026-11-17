import pytest
import requests

from devlm.config import TeacherConfig, WorldConfig
from devlm.teacher import (LLMTeacher, OllamaTeacher, Question, ScriptedTeacher, TeacherError, clean_reply,
                           make_teacher, pick_lesson_facts, states_fact)
from devlm.world import build_world

WORLD = build_world(WorldConfig(seed=11))
FACT = WORLD.facts[0]
QUESTION = Question(f"Why did the {FACT.subject} {FACT.effect.base}?", "it was sad", FACT.subject, FACT)


class StubTeacher(LLMTeacher):
    source = "stub"

    def __init__(self, reply=None, error=None):
        super().__init__(TeacherConfig(backend="ollama"))
        self.reply, self.error, self.calls = reply, error, []

    def _complete(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return self.reply


def test_scripted_teacher_states_the_rule():
    answer = ScriptedTeacher().explain(QUESTION, WORLD, 1)
    assert FACT.statement() in answer.text and answer.source == "scripted"
    noise = Question("Why did the x jump?", "", "x", None)
    assert "no rule" in ScriptedTeacher().explain(noise, WORLD, 1).text
    assert FACT.statement() in ScriptedTeacher().lesson(FACT, WORLD, 1).text


def test_clean_reply_strips_thinking_markdown_and_length():
    assert clean_reply("<think>hmm</think> **Hi** there.", 10) == "Hi there."
    long = "One two three. Four five six seven eight nine ten."
    assert clean_reply(long, 5) == "One two three."


def test_states_fact_checks_subject_and_effect():
    assert states_fact(f"The {FACT.subject} {FACT.effect.present} {FACT.condition.when}.", FACT)
    assert not states_fact("Creatures do things.", FACT)


def test_llm_answer_is_used_when_it_states_the_rule():
    teacher = StubTeacher(reply=f"Yes! {FACT.statement()}")
    answer = teacher.explain(QUESTION, WORLD, 1)
    assert answer.source == "stub" and teacher.fallbacks == 0
    system, user = teacher.calls[0]
    assert FACT.statement() in system and "it was sad" in user


def test_wrong_llm_answer_falls_back_to_script():
    answer = StubTeacher(reply="Because the moon is cheese.").explain(QUESTION, WORLD, 1)
    assert answer.source == "fallback" and FACT.statement() in answer.text


def test_backend_error_falls_back_to_script():
    teacher = StubTeacher(error=TeacherError("down"))
    assert teacher.lesson(FACT, WORLD, 1).source == "fallback"
    assert teacher.fallbacks == 1


def test_only_introduced_rules_reach_the_prompt():
    late = WORLD.facts[-1]
    teacher = StubTeacher(reply=FACT.statement())
    teacher.explain(QUESTION, WORLD, 1)
    assert late.statement() not in teacher.calls[0][0]


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, "body"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("404", response=self)

    def json(self):
        return self.payload


def test_ollama_request_and_response(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, json=json, timeout=timeout)
        return FakeResponse({"message": {"content": FACT.statement()}})

    monkeypatch.setattr(requests, "post", fake_post)
    teacher = OllamaTeacher(TeacherConfig(model="tiny", ollama_url="http://host:1/", timeout_seconds=9))
    assert teacher.explain(QUESTION, WORLD, 1).source == "ollama"
    assert seen["url"] == "http://host:1/api/chat"
    assert seen["json"]["model"] == "tiny" and seen["json"]["stream"] is False
    assert seen["timeout"] == 9


def test_ollama_errors_become_teacher_errors(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", refuse)
    with pytest.raises(TeacherError, match="ollama serve"):
        OllamaTeacher(TeacherConfig()).check()
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({}, status=404))
    with pytest.raises(TeacherError, match="ollama pull"):
        OllamaTeacher(TeacherConfig()).check()
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse({"oops": 1}))
    with pytest.raises(TeacherError, match="Unexpected"):
        OllamaTeacher(TeacherConfig()).check()


def test_make_teacher():
    assert isinstance(make_teacher(TeacherConfig(backend="scripted")), ScriptedTeacher)
    assert isinstance(make_teacher(TeacherConfig(backend="ollama")), OllamaTeacher)


def test_lessons_prefer_untaught_facts():
    import random
    taught = [f.id for f in WORLD.introduced_facts(1)][:-1]
    picked = pick_lesson_facts(WORLD, 1, taught, 1, random.Random(0))
    assert picked == [WORLD.introduced_facts(1)[-1]]
