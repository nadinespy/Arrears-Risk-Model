"""Configuration for the arrears risk model.

Loaded from `config/default.yaml` by default. Environment variables prefixed
with `ARM_` override YAML values, using `__` as the nested-field delimiter:

    ARM_TRAINING__RANDOM_STATE=7  -> overrides config.training.random_state

The split between YAML and pydantic exists so values stay easy to edit by
hand while pydantic gives typed validation, IDE autocomplete in callers,
and clear errors when the file is malformed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Anchor for resolving relative paths in YAML. config.py lives at
# src/arrears_risk_model/config.py, so parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class _StrictModel(BaseModel):
    """Base for nested config sections — rejects unknown keys to catch typos."""

    model_config = ConfigDict(extra="forbid")


class Paths(_StrictModel):
    household_data: Path
    imd_data: Path
    lewisham_geojson: Path
    model_dir: Path
    output_dir: Path

    def resolved(self, root: Path = REPO_ROOT) -> Paths:
        """Return a copy with relative paths resolved against `root`."""
        return Paths(
            **{
                name: (root / value if not value.is_absolute() else value)
                for name, value in self.model_dump().items()
            }
        )


class Features(_StrictModel):
    continuous: list[str]
    binary: list[str]
    categorical: list[str]
    ordinal: list[str]
    engineered: list[str]
    target: str
    excluded: list[str]

    @property
    def all_input_features(self) -> list[str]:
        """All features fed to the model (post-engineering, excluding target)."""
        return self.continuous + self.binary + self.categorical + self.ordinal + self.engineered


class Imputation(_StrictModel):
    ben_cap_amount_strategy: Literal["median", "mean", "constant", "zero"] = "median"
    ben_cap_amount_fill_value: float | None = None  # used only when strategy='constant'


class LRHyperparams(_StrictModel):
    C: float = Field(1.0, gt=0)
    penalty: Literal["l1", "l2", "elasticnet"] | None = "l2"
    solver: str = "lbfgs"
    max_iter: int = Field(1000, gt=0)
    class_weight: str | None = "balanced"


class XGBHyperparams(_StrictModel):
    n_estimators: int = Field(100, gt=0)
    max_depth: int = Field(6, gt=0)
    learning_rate: float = Field(0.1, gt=0)
    subsample: float = Field(1.0, gt=0, le=1.0)
    colsample_bytree: float = Field(1.0, gt=0, le=1.0)
    scale_pos_weight: float | None = None  # if None, train.py computes from data
    eval_metric: str = "logloss"


class Models(_StrictModel):
    lr: LRHyperparams = LRHyperparams()
    xgb: XGBHyperparams = XGBHyperparams()


class Training(_StrictModel):
    test_size: float = Field(0.2, gt=0, lt=1)
    cv_n_splits: int = Field(5, ge=2)
    cv_stratify: bool = True
    random_state: int = 42


class EquityWeights(_StrictModel):
    """Additive weights applied to predicted probability for prioritisation.

    Deliberately small and auditable per the original Section 6.4 framing —
    not a fairness model, just a transparent value-judgement uplift. The
    composite score = predicted_probability + sum of weights for criteria
    the household meets. See docs/model_card.md for discussion.
    """

    children: float = 0.05
    disability: float = 0.05


class Config(BaseSettings):
    """Top-level configuration. Use `load_config()` to construct."""

    model_config = SettingsConfigDict(
        yaml_file=str(DEFAULT_CONFIG_PATH),
        env_prefix="ARM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",
    )

    paths: Paths
    features: Features
    imputation: Imputation = Imputation()
    models: Models = Models()
    training: Training = Training()
    equity: EquityWeights = EquityWeights()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence (highest first): explicit init kwargs, env vars, YAML.
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


def load_config(yaml_path: Path | str | None = None) -> Config:
    """Load and validate configuration.

    With no argument, reads the shipped `config/default.yaml` and applies
    environment overrides (`ARM_*`).

    With `yaml_path`, loads that file directly and validates. Useful for
    tests and for swapping configurations between environments.
    """
    if yaml_path is None:
        return Config()

    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
