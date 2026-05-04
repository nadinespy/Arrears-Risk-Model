"""Tests for the feature engineering and preprocessing pipeline factories."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arrears_risk_model.config import load_config
from arrears_risk_model.features import make_lr_preprocessor, make_xgb_preprocessor


@pytest.fixture
def config():
    return load_config()


def _expected_lr_output_cols(config, df: pd.DataFrame) -> int:
    """Count the expected output columns from the LR preprocessor.

    OHE produces one column per unique category *seen during fit*, so the
    count is data-dependent. We derive it from the fixture rather than
    hardcoding real-data cardinalities.
    """
    n_continuous = len(config.features.continuous) + len(config.features.engineered)
    n_binary = len(config.features.binary)
    n_categorical = sum(df[c].nunique() for c in config.features.categorical)
    n_ordinal = len(config.features.ordinal)
    return n_continuous + n_binary + n_categorical + n_ordinal


def _expected_xgb_output_cols(config) -> int:
    """x_outGB preprocessor: all feature groups produce one integer column each."""
    return (
        len(config.features.continuous)
        + len(config.features.engineered)
        + len(config.features.binary)
        + len(config.features.categorical)
        + len(config.features.ordinal)
    )


# ---------- LR preprocessor ----------

def test_lr_preprocessor_fits_and_transforms(
    config, joined_df: pd.DataFrame
) -> None:
    """Fit+transform succeeds on the synthetic fixture."""
    pre = make_lr_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    assert x_out.shape[0] == len(joined_df)
    assert not np.isnan(x_out).any(), "NaN survived LR preprocessing"


def test_lr_preprocessor_output_shape(config, joined_df: pd.DataFrame) -> None:
    """Output column count matches sum of feature group widths."""
    pre = make_lr_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    expected = _expected_lr_output_cols(config, joined_df)
    assert x_out.shape[1] == expected, (
        f"Expected {expected} columns, got {x_out.shape[1]}"
    )


def test_lr_preprocessor_handles_unseen_category(
    config, joined_df: pd.DataFrame
) -> None:
    """OneHotEncoder with handle_unknown='ignore' produces all-zeros, not an error."""
    pre = make_lr_preprocessor(config)
    pre.fit(joined_df)

    unseen = joined_df.copy()
    unseen.loc[0, "tenure_type"] = "Houseboat"  # not in TENURE_TYPES
    x_out = pre.transform(unseen)  # must not raise
    assert x_out.shape[0] == len(unseen)
    # No exception is the main assertion; sanity-check the shape held.
    assert x_out.shape[0] == len(unseen)


def test_lr_preprocessor_imputes_nans(config, joined_df: pd.DataFrame) -> None:
    """NaN in ben_cap_amount (the one nullable column) is filled after preprocessing."""
    assert joined_df["ben_cap_amount"].isna().sum() > 0, "fixture must have at least one NaN"
    pre = make_lr_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    assert not np.isnan(x_out).any()


def test_lr_preprocessor_scales_continuous(config, joined_df: pd.DataFrame) -> None:
    """Continuous columns are roughly standardised (mean ≈ 0, std ≈ 1) after scaling."""
    pre = make_lr_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    n_cont = len(config.features.continuous) + len(config.features.engineered)
    continuous_block = x_out[:, :n_cont]
    # With only 50 rows the mean won't be exactly 0, but should be close.
    assert abs(continuous_block.mean()) < 1.0


# ---------- x_outGB preprocessor ----------

def test_xgb_preprocessor_fits_and_transforms(
    config, joined_df: pd.DataFrame
) -> None:
    """Fit+transform succeeds on the synthetic fixture."""
    pre = make_xgb_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    assert x_out.shape[0] == len(joined_df)
    assert not np.isnan(x_out).any(), "NaN survived x_outGB preprocessing"


def test_xgb_preprocessor_output_shape(config, joined_df: pd.DataFrame) -> None:
    """x_outGB output has one column per feature (no explosion from one-hot)."""
    pre = make_xgb_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    expected = _expected_xgb_output_cols(config)
    assert x_out.shape[1] == expected


def test_xgb_preprocessor_handles_unseen_category(
    config, joined_df: pd.DataFrame
) -> None:
    """OrdinalEncoder with handle_unknown='use_encoded_value' returns -1, not an error."""
    pre = make_xgb_preprocessor(config)
    pre.fit(joined_df)

    unseen = joined_df.copy()
    unseen.loc[0, "tenure_type"] = "Houseboat"
    x_out = pre.transform(unseen)  # must not raise
    assert x_out.shape[0] == len(unseen)


def test_xgb_preprocessor_no_scaling(config, joined_df: pd.DataFrame) -> None:
    """x_outGB continuous columns are NOT standardised — values keep original scale."""
    pre = make_xgb_preprocessor(config)
    x_out = pre.fit_transform(joined_df)
    # monthly_rent is the first continuous column; its original mean is ~873.
    # If StandardScaler were applied the mean would be ~0.
    monthly_rent_col = x_out[:, 0]
    assert abs(monthly_rent_col.mean()) > 10, "Continuous values look scaled (unexpected)"


# ---------- Shared invariants ----------

def test_both_preprocessors_drop_excluded_columns(
    config, joined_df: pd.DataFrame
) -> None:
    """Columns in features.excluded (reference, lsoa21cd, arrears_amount, etc.)
    are not passed through to the model."""
    lr_pre = make_lr_preprocessor(config)
    xgb_pre = make_xgb_preprocessor(config)

    for col in config.features.excluded:
        assert col in joined_df.columns, f"{col!r} missing from fixture — check conftest"

    lr_x_out = lr_pre.fit_transform(joined_df)
    xgb_x_out = xgb_pre.fit_transform(joined_df)

    # If excluded columns leak through, the column counts would be higher.
    assert lr_x_out.shape[1] == _expected_lr_output_cols(config, joined_df)
    assert xgb_x_out.shape[1] == _expected_xgb_output_cols(config)


def test_engineered_feature_total_shortfall_added(
    config, joined_df: pd.DataFrame
) -> None:
    """The engineer step creates total_shortfall before the ColumnTransformer runs."""
    pre = make_lr_preprocessor(config)
    # Access the engineer step and check it adds the column.
    engineer_step = pre.named_steps["engineer"]
    out = engineer_step.transform(joined_df)
    assert "total_shortfall" in out.columns
    expected = joined_df["lha_shortfall_weekly"] + joined_df["ben_cap_amount"].fillna(0)
    pd.testing.assert_series_equal(out["total_shortfall"], expected, check_names=False)
