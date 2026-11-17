from pathlib import Path

import pytest

from devlm.config import Config, ConfigError, config_from_dict, config_to_dict, load_config, save_config, validate

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def test_defaults_are_valid():
    assert validate(Config()).run_name == "first-child"


def test_repository_config_file_loads():
    cfg = load_config(REPO_CONFIG)
    assert cfg.child.model_name == "roneneldan/TinyStories-33M"
    assert cfg.world.arrival_days == (8, 16, 24)
    assert isinstance(cfg.sleep.learning_rate, float)


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"sleep": {"epochz": 3}})


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="must be an integer"):
        config_from_dict({"sleep": {"epochs": "three"}})


def test_bool_is_not_accepted_as_int():
    with pytest.raises(ConfigError):
        config_from_dict({"sleep": {"epochs": True}})


def test_int_is_coerced_to_float():
    assert config_from_dict({"sleep": {"learning_rate": 1}}).sleep.learning_rate == 1.0


def test_all_invalid_values_are_reported_together():
    with pytest.raises(ConfigError) as info:
        config_from_dict({"sleep": {"train_mode": "dream"}, "teacher": {"backend": "owl"}})
    assert "train_mode" in str(info.value) and "backend" in str(info.value)


@pytest.mark.parametrize("name", ["../escape", "a/b", "", "x" * 65])
def test_unsafe_run_names_are_rejected(name):
    with pytest.raises(ConfigError, match="run_name"):
        config_from_dict({"run_name": name})


def test_growth_only_requires_growth():
    with pytest.raises(ConfigError, match="growth_only"):
        config_from_dict({"sleep": {"train_mode": "growth_only"}, "growth": {"enabled": False}})


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_gives_clear_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("sleep: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_save_and_reload_round_trip(tmp_path):
    cfg = config_from_dict({"probes": {"metacog_lags": [2, 4]}})
    path = tmp_path / "c.yaml"
    save_config(cfg, path)
    assert load_config(path) == cfg
    assert config_to_dict(cfg)["probes"]["metacog_lags"] == [2, 4]
