"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from arrears_risk_model.config import Config, load_config


def _minimal_valid_yaml() -> dict:
    """Smallest YAML payload that satisfies the schema."""
    return {
        "paths": {
            "household_data": "x.xlsx",
            "imd_data": "y.xlsx",
            "lewisham_geojson": "z.geojson",
            "model_dir": "m",
            "output_dir": "o",
        },
        "features": {
            "continuous": [],
            "binary": [],
            "categorical": [],
            "ordinal": [],
            "engineered": [],
            "target": "arrears_flag",
            "excluded": [],
        },
    }


def test_default_config_loads():
    """The shipped default.yaml parses, validates, and produces a Config."""
    config = load_config()
    assert isinstance(config, Config)
    # Spot-check a few fields to confirm the YAML round-tripped.
    assert config.training.random_state == 42
    assert config.features.target == "arrears_flag"
    assert "monthly_rent" in config.features.continuous
    assert "tenure_type" in config.features.categorical
    assert config.equity.children == 0.05


def test_features_helper_property():
    """all_input_features concatenates the four typed feature lists + engineered."""
    config = load_config()
    expected = (
        config.features.continuous
        + config.features.binary
        + config.features.categorical
        + config.features.ordinal
        + config.features.engineered
    )
    assert config.features.all_input_features == expected
    assert config.features.target not in config.features.all_input_features


def test_custom_yaml_loads(tmp_path):
    """A custom YAML path is read and validated."""
    data = _minimal_valid_yaml()
    data["training"] = {"random_state": 7}
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(yaml.safe_dump(data))

    config = load_config(yaml_path)
    assert config.training.random_state == 7


def test_missing_yaml_raises(tmp_path):
    """A non-existent YAML path raises FileNotFoundError, not a quiet default."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_unknown_key_rejected(tmp_path):
    """extra='forbid' catches typos in YAML keys."""
    data = _minimal_valid_yaml()
    data["unkown_section"] = {"foo": "bar"}  # typo intentional
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(yaml_path)


def test_unknown_nested_key_rejected(tmp_path):
    """Typos inside nested sections (e.g. training) also fail validation."""
    data = _minimal_valid_yaml()
    data["training"] = {"randdom_state": 7}  # typo intentional
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(yaml_path)


def test_wrong_type_rejected(tmp_path):
    """A string where an int is expected raises ValidationError."""
    data = _minimal_valid_yaml()
    data["training"] = {"random_state": "forty-two"}
    yaml_path = tmp_path / "bad_type.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(yaml_path)


def test_out_of_range_value_rejected(tmp_path):
    """Field constraints (e.g. test_size in (0, 1)) are enforced."""
    data = _minimal_valid_yaml()
    data["training"] = {"test_size": 1.5}
    yaml_path = tmp_path / "bad_range.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError):
        load_config(yaml_path)


def test_env_var_overrides_yaml(monkeypatch):
    """ARM_-prefixed env vars override values from the YAML file."""
    monkeypatch.setenv("ARM_TRAINING__RANDOM_STATE", "7")
    config = load_config()
    assert config.training.random_state == 7


def test_env_var_nested_override(monkeypatch):
    """Nested env override using __ delimiter for two levels of nesting."""
    monkeypatch.setenv("ARM_MODELS__XGB__N_ESTIMATORS", "250")
    config = load_config()
    assert config.models.xgb.n_estimators == 250


def test_paths_resolved_against_repo_root(tmp_path):
    """Paths.resolved() turns relative paths into absolute paths."""
    config = load_config()
    resolved = config.paths.resolved()
    assert resolved.household_data.is_absolute()
    assert resolved.model_dir.is_absolute()
