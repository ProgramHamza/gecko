import os

import pytest
import yaml

from devlm import cli
from devlm.run_files import read_jsonl, run_paths

from conftest import FakeLearner, small_config_dict


@pytest.fixture
def project(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(small_config_dict(tmp_path)), encoding="utf-8")
    monkeypatch.setattr(cli, "make_learner", lambda cfg: FakeLearner())
    monkeypatch.chdir(tmp_path)
    return config


def test_init_run_status_report_compare(project, tmp_path, capsys):
    assert cli.main(["--config", str(project), "init"]) == 0
    assert cli.main(["--config", str(project), "run", "--days", "2"]) == 0
    paths = run_paths(tmp_path / "runs", "test-run")
    assert [m["day"] for m in read_jsonl(paths.metrics)] == [0, 1, 2]
    assert paths.report.exists()
    assert cli.main(["--config", str(project), "status"]) == 0
    assert "day 2" in capsys.readouterr().out
    assert cli.main(["--config", str(project), "compare", "What is a zib?", "--choices", "yes|no",
                     "--days", "0,2"]) == 0
    out = capsys.readouterr().out
    assert "day   0" in out and "day   2" in out


def test_run_name_override(project, tmp_path):
    assert cli.main(["--config", str(project), "--run", "other", "init"]) == 0
    assert run_paths(tmp_path / "runs", "other").state.exists()


def test_errors_are_reported_not_raised(project, capsys):
    assert cli.main(["--config", str(project), "status"]) == 2
    assert "No run at" in capsys.readouterr().err
    assert cli.main(["--config", "missing.yaml", "status"]) == 2
    assert cli.main(["--config", str(project), "run", "--days", "0"]) == 2


def test_compare_rejects_missing_snapshot(project, capsys):
    cli.main(["--config", str(project), "init"])
    assert cli.main(["--config", str(project), "compare", "hi", "--days", "7"]) == 2
    assert "No snapshot" in capsys.readouterr().err


def test_prompt_modes():
    assert cli._prompt("Why?") == ("Q: Why?\nA:", True)
    assert cli._prompt("!Once upon") == ("Once upon", False)


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nDEVLM_A='one'\nDEVLM_B=two\nnot a pair\n", encoding="utf-8")
    monkeypatch.setenv("DEVLM_B", "kept")
    monkeypatch.delenv("DEVLM_A", raising=False)
    cli.load_dotenv(env)
    assert os.environ["DEVLM_A"] == "one" and os.environ["DEVLM_B"] == "kept"
    monkeypatch.delenv("DEVLM_A")


def test_softmax():
    probs = cli._softmax([0.0, 0.0])
    assert probs == [0.5, 0.5]


def test_start_creates_then_continues(project, tmp_path):
    assert cli.main(["--config", str(project), "start", "--days", "1"]) == 0
    assert cli.main(["--config", str(project), "start", "--days", "2"]) == 0
    paths = run_paths(tmp_path / "runs", "test-run")
    assert [m["day"] for m in read_jsonl(paths.metrics)] == [0, 1, 2, 3]
