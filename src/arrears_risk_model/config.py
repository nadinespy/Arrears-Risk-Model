"""Configuration for the arrears risk model.

Loaded from ``config/default.yaml`` by default. Environment variables
prefixed with ``ARM_`` override YAML values.

**Env-var naming rule.** Take the dotted path of a field in the config
tree, uppercase it, and join nested levels with double underscores
(``__``). The result is the env var that overrides that field.

Examples::

    config.training.random_state           ARM_TRAINING__RANDOM_STATE=7
    config.training.test_size              ARM_TRAINING__TEST_SIZE=0.25
    config.models.xgb.n_estimators         ARM_MODELS__XGB__N_ESTIMATORS=250
    config.models.lr.C                     ARM_MODELS__LR__C=0.5
    config.equity.children                 ARM_EQUITY__CHILDREN=0.1
    config.paths.model_dir                 ARM_PATHS__MODEL_DIR=/tmp/models

The override precedence (highest to lowest) is: explicit kwargs to
``Config(...)``, then ``ARM_*`` environment variables, then YAML.

The split between YAML and pydantic exists so the file stays easy to
edit by hand, while pydantic adds typed validation, IDE autocomplete
in callers, and clear errors when YAML is malformed.
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
    """Filesystem locations. Override via ``ARM_PATHS__<FIELD>=...``.

    Relative paths in YAML are resolved against the repo root by
    :meth:`resolved` (called by callers that need absolute paths).
    """

    household_data: Path = Field(description="Raw household_data.xlsx (input)")
    imd_data: Path = Field(description="Raw IMD25 .xlsx (input)")
    lewisham_geojson: Path = Field(description="Lewisham LSOA→ward geojson (input)")
    model_dir: Path = Field(description="Where train.py writes serialised pipelines + metadata")
    output_dir: Path = Field(description="Where predict.py writes ranked priority lists")

    def resolved(self, root: Path = REPO_ROOT) -> Paths:
        """Return a copy with relative paths resolved against ``root``."""
        return Paths(
            **{
                name: (root / value if not value.is_absolute() else value)
                for name, value in self.model_dump().items()
            }
        )


class Features(_StrictModel):
    """Which columns the model consumes, grouped by how they're encoded.

    Override via ``ARM_FEATURES__<FIELD>=...``. List-valued fields take a
    JSON array as the env-var value, e.g.
    ``ARM_FEATURES__CONTINUOUS='["monthly_rent","income_after_costs"]'``.
    """

    continuous: list[str] = Field(description="Numeric features used as-is (scaled in LR pipeline)")
    binary: list[str] = Field(description="Recoded 0/1 features")
    categorical: list[str] = Field(description="Nominal features (one-hot for LR, ordinal for XGB)")
    ordinal: list[str] = Field(description="Features with meaningful ordering, treated as numeric")
    engineered: list[str] = Field(description="Features built in features.py from raw inputs")
    target: str = Field(description="Name of the binary target column")
    excluded: list[str] = Field(description="Columns dropped before modelling (IDs, leakage)")
    sensitive_features: list[str] = Field(
        description="Pre-pipeline columns sliced for fairness metrics in evaluation"
    )

    @property
    def all_input_features(self) -> list[str]:
        """All features fed to the model (post-engineering, excluding target)."""
        return self.continuous + self.binary + self.categorical + self.ordinal + self.engineered


class Imputation(_StrictModel):
    """Imputation strategy. Override via ``ARM_IMPUTATION__<FIELD>=...``."""

    ben_cap_amount_strategy: Literal["median", "mean", "constant", "zero"] = Field(
        "median",
        description="How to fill missing ben_cap_amount values",
    )
    ben_cap_amount_fill_value: float | None = Field(
        None,
        description="Constant used when strategy='constant'; ignored otherwise",
    )


class LRHyperparams(_StrictModel):
    """Logistic regression knobs. Override via ``ARM_MODELS__LR__<FIELD>=...``.

    Field names mirror sklearn's ``LogisticRegression`` constructor (1.8+ API).
    ``penalty`` was removed in sklearn 1.8; use ``l1_ratio`` instead:
    0.0 = L2, 1.0 = L1, 0 < l1_ratio < 1 = ElasticNet.
    """

    C: float = Field(1.0, gt=0, description="Inverse regularisation strength (smaller = more reg.)")
    l1_ratio: float = Field(
        0.0, ge=0, le=1,
        description="Mixing parameter: 0 = L2, 1 = L1, in-between = ElasticNet",
    )
    solver: str = Field("lbfgs", description="sklearn solver; lbfgs supports L2 only")
    max_iter: int = Field(1000, gt=0, description="Maximum solver iterations")
    class_weight: str | None = Field(
        "balanced",
        description="'balanced' adjusts for ~25%/75% target imbalance; None = no reweight",
    )


class XGBHyperparams(_StrictModel):
    """XGBoost knobs. Override via ``ARM_MODELS__XGB__<FIELD>=...``.

    Field names mirror xgboost's ``XGBClassifier`` constructor.
    """

    n_estimators: int = Field(100, gt=0, description="Number of boosting rounds")
    max_depth: int = Field(6, gt=0, description="Maximum tree depth")
    learning_rate: float = Field(0.1, gt=0, description="Shrinkage applied to each round")
    subsample: float = Field(1.0, gt=0, le=1.0, description="Row subsample ratio per tree")
    colsample_bytree: float = Field(
        1.0, gt=0, le=1.0, description="Column subsample ratio per tree"
    )
    scale_pos_weight: float | None = Field(
        None,
        description="Imbalance multiplier for positive class; None → derived from training data",
    )
    eval_metric: str = Field("logloss", description="Booster eval metric")


class Models(_StrictModel):
    """Model-by-model hyperparameters. Override via ``ARM_MODELS__<MODEL>__<FIELD>=...``."""

    lr: LRHyperparams = LRHyperparams()
    xgb: XGBHyperparams = XGBHyperparams()


class Training(_StrictModel):
    """Train/test split + cross-validation. Override via ``ARM_TRAINING__<FIELD>=...``."""

    test_size: float = Field(
        0.2, gt=0, lt=1, description="Held-out fraction for final evaluation"
    )
    cv_n_splits: int = Field(5, ge=2, description="K for stratified k-fold cross-validation")
    cv_stratify: bool = Field(True, description="Whether to stratify CV folds by the target")
    random_state: int = Field(42, description="Seed for split, CV shuffle, and model RNGs")


class EquityWeights(_StrictModel):
    """Additive prioritisation weights. Override via ``ARM_EQUITY__<FIELD>=...``.

    Composite score = predicted_probability + sum of weights for criteria
    the household meets. Deliberately small and auditable per the
    original Section 6.4 framing — not a fairness model, a transparent
    value-judgement uplift. See ``docs/model_card.md``.
    """

    children: float = Field(0.05, description="Uplift if household has children")
    disability: float = Field(0.05, description="Uplift if household has a disabled member")


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
