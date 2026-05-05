"""Preprocessing pipeline factories for the arrears risk model.

Two preprocessors are provided:

- :func:`make_lr_preprocessor` — for logistic regression: one-hot encodes
  nominal categoricals, ordinal-encodes + scales ordinal features,
  standard-scales continuous and engineered features, passes binary
  0/1 features through unchanged.
- :func:`make_xgb_preprocessor` — for XGBoost: ordinal-encodes all
  categorical and ordinal features (integers only, no scaling), imputes
  continuous features but leaves them unscaled (XGBoost is scale-invariant).

Both preprocessors include feature engineering (``total_shortfall``) as
a first ``FunctionTransformer`` step so the full preprocessing chain is
self-contained inside one sklearn ``Pipeline`` — it serialises cleanly
with ``joblib`` and applies identically to training and inference data.

**Column treatment summary**

  continuous  →  impute → scale (LR) | impute only (XGB)
  binary      →  passthrough (already 0/1 after recode_binaries)
  categorical →  OneHotEncoder handle_unknown=ignore (LR) | OrdinalEncoder (XGB)
  ordinal     →  OrdinalEncoder → scale (LR) | OrdinalEncoder only (XGB)
  engineered  →  impute → scale (LR) | impute only (XGB)
  excluded    →  dropped (remainder='drop')
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from arrears_risk_model.config import Config
from arrears_risk_model.schemas import AGE_BRACKETS

# Explicit category orderings for ordinal features. The order determines
# the integer codes assigned by OrdinalEncoder (0 = lowest).
# age_bracket: youngest-to-oldest, matching the sorting logic in the
# original notebook: int(bracket.split('-')[0].replace('+', '')).
ORDINAL_CATEGORIES: dict[str, list] = {
    "age_bracket": list(AGE_BRACKETS),  # already sorted youngest→oldest in schemas
    "imd_decile": list(range(1, 11)),   # 1 = most deprived; 10 = least deprived
}


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_shortfall`` (used inside the sklearn Pipeline as a FunctionTransformer).

    total_shortfall = lha_shortfall_weekly + ben_cap_amount
    Missing ben_cap_amount is treated as 0 (household has no benefit cap).
    The underlying ben_cap_amount column is kept and imputed separately by
    the continuous-feature imputer downstream — keep the YAML's
    ``imputation.ben_cap_amount_strategy`` aligned with this convention
    (default ``zero``) so the two paths agree on what "missing" means.
    """
    out = df.copy()
    out["total_shortfall"] = out["lha_shortfall_weekly"] + out["ben_cap_amount"].fillna(0.0)
    return out


def _build_ordinal_categories(ordinal_cols: list[str]) -> list[list]:
    """Return the category list for OrdinalEncoder, one entry per column.

    Raises ``KeyError`` if a column is listed as ordinal but has no entry
    in ``ORDINAL_CATEGORIES`` — this catches the case where someone adds a
    new ordinal feature in config without also defining its order here.
    """
    missing = [c for c in ordinal_cols if c not in ORDINAL_CATEGORIES]
    if missing:
        raise KeyError(
            f"No ordinal category order defined for: {missing}. "
            "Add an entry to features.ORDINAL_CATEGORIES."
        )
    return [ORDINAL_CATEGORIES[c] for c in ordinal_cols]


def _make_imputer(config: Config) -> SimpleImputer:
    """Construct a SimpleImputer from config.imputation settings.

    The config's 'zero' strategy maps to sklearn's 'constant' with
    fill_value=0 (sklearn doesn't have a 'zero' strategy natively).
    """
    strategy = config.imputation.ben_cap_amount_strategy
    if strategy == "zero":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    if strategy == "constant":
        return SimpleImputer(
            strategy="constant",
            fill_value=config.imputation.ben_cap_amount_fill_value,
        )
    return SimpleImputer(strategy=strategy)


def make_lr_preprocessor(config: Config) -> Pipeline:
    """Return the full preprocessing pipeline for logistic regression.

    Wraps feature engineering + a ``ColumnTransformer`` in a single
    sklearn ``Pipeline``. The ColumnTransformer step is accessible as
    ``pipeline.named_steps['column_transform']``.
    """
    continuous_cols = config.features.continuous + config.features.engineered
    categorical_cols = config.features.categorical
    ordinal_cols = config.features.ordinal
    binary_cols = config.features.binary

    continuous_pipe = Pipeline([
        ("imputer", _make_imputer(config)),
        ("scaler", StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        (
            "encoder",
            OrdinalEncoder(
                categories=_build_ordinal_categories(ordinal_cols),
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    column_transform = ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, continuous_cols),
            ("binary", "passthrough", binary_cols),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",  # unseen categories → all-zero row
                    sparse_output=False,
                    drop=None,
                ),
                categorical_cols,
            ),
            ("ordinal", ordinal_pipe, ordinal_cols),
        ],
        remainder="drop",  # excludes reference, lsoa21cd, arrears_amount, target, etc.
    )

    return Pipeline([
        ("engineer", FunctionTransformer(_add_engineered_features, validate=False)),
        ("column_transform", column_transform),
    ])


def make_xgb_preprocessor(config: Config) -> Pipeline:
    """Return the full preprocessing pipeline for XGBoost.

    XGBoost is scale-invariant, so no StandardScaler is applied.
    Categorical features are integer-encoded (OrdinalEncoder) rather than
    one-hot; XGBoost finds its own splits across the integer range.
    Unknown categories at inference time get -1 (XGBoost will treat them
    like any other unseen split value).
    """
    continuous_cols = config.features.continuous + config.features.engineered
    categorical_cols = config.features.categorical
    ordinal_cols = config.features.ordinal
    binary_cols = config.features.binary

    column_transform = ColumnTransformer(
        transformers=[
            ("continuous", _make_imputer(config), continuous_cols),
            ("binary", "passthrough", binary_cols),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_cols,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=_build_ordinal_categories(ordinal_cols),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("engineer", FunctionTransformer(_add_engineered_features, validate=False)),
        ("column_transform", column_transform),
    ])


__all__ = [
    "ORDINAL_CATEGORIES",
    "make_lr_preprocessor",
    "make_xgb_preprocessor",
]
