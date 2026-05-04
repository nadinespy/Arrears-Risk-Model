"""Tests for evaluation functions.

Two categories:
- Integration tests: use a real fitted pipeline on the synthetic fixture
  to verify the functions run end-to-end and return valid ranges.
- Correctness tests: supply known y_true / y_proba via a mock pipeline
  to verify exact metric values.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from arrears_risk_model.config import load_config
from arrears_risk_model.evaluate import (
    CalibrationResults,
    CVResults,
    FairnessResults,
    HeldOutResults,
    compute_calibration,
    compute_fairness_metrics,
    cross_validate_model,
    evaluate_held_out,
)
from arrears_risk_model.models import make_lr_pipeline


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def fitted_lr(config, joined_df: pd.DataFrame):
    """LR pipeline fitted on the full synthetic fixture."""
    pipe = make_lr_pipeline(config)
    y = joined_df["arrears_flag"]
    x_df = joined_df.drop(columns=["arrears_flag"])
    pipe.fit(x_df, y)
    return pipe, x_df, y


def _mock_pipeline(y_true: np.ndarray, proba_positive: np.ndarray) -> MagicMock:
    """Return a mock that behaves like a fitted sklearn pipeline."""
    proba_2d = np.column_stack([1 - proba_positive, proba_positive])
    pipe = MagicMock()
    pipe.predict_proba.return_value = proba_2d
    pipe.predict.return_value = (proba_positive >= 0.5).astype(int)
    pipe.classes_ = [0, 1]
    return pipe


# ---------- cross_validate_model ----------

def test_cv_returns_cv_results(config, joined_df: pd.DataFrame) -> None:
    """cross_validate_model returns a CVResults with valid metric ranges."""
    pipe = make_lr_pipeline(config)
    y = joined_df["arrears_flag"]
    x_df = joined_df.drop(columns=["arrears_flag"])
    result = cross_validate_model(pipe, x_df, y, config, model_name="lr")
    assert isinstance(result, CVResults)
    assert result.model_name == "lr"
    assert result.n_splits == config.training.cv_n_splits
    assert 0.0 <= result.roc_auc_mean <= 1.0
    assert 0.0 <= result.pr_auc_mean <= 1.0
    assert 0.0 <= result.f1_mean <= 1.0
    assert result.roc_auc_std >= 0.0


# ---------- evaluate_held_out ----------

def test_held_out_valid_range(fitted_lr) -> None:
    """evaluate_held_out returns HeldOutResults with metrics in [0, 1]."""
    pipe, x_df, y = fitted_lr
    result = evaluate_held_out(pipe, x_df, y, model_name="lr")
    assert isinstance(result, HeldOutResults)
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= result.pr_auc <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.n_test == len(y)


def test_held_out_perfect_classifier() -> None:
    """A perfect classifier should yield ROC-AUC = 1 and F1 = 1."""
    n = 40
    y_true = np.array([1] * 10 + [0] * 30)
    proba = np.array([0.9] * 10 + [0.1] * 30)
    pipe = _mock_pipeline(y_true, proba)
    x_df = pd.DataFrame({"dummy": range(n)})
    y_ser = pd.Series(y_true)

    result = evaluate_held_out(pipe, x_df, y_ser, model_name="perfect")
    assert result.roc_auc == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_held_out_threshold_respected() -> None:
    """Predictions are thresholded at the supplied value, not always 0.5."""
    n = 20
    y_true = np.array([1] * 10 + [0] * 10)
    # All positive probabilities are 0.4 — below default 0.5 → all predicted 0.
    proba = np.full(n, 0.4)
    pipe = _mock_pipeline(y_true, proba)
    x_df = pd.DataFrame({"dummy": range(n)})
    y_ser = pd.Series(y_true)

    # At threshold=0.5 nothing is predicted positive → F1=0.
    result_default = evaluate_held_out(pipe, x_df, y_ser, model_name="t", threshold=0.5)
    assert result_default.f1 == pytest.approx(0.0)

    # At threshold=0.3 everything is predicted positive → recall=1.
    result_low = evaluate_held_out(pipe, x_df, y_ser, model_name="t", threshold=0.3)
    assert result_low.recall == pytest.approx(1.0)


# ---------- compute_calibration ----------

def test_calibration_brier_score_range(fitted_lr) -> None:
    """Brier score is in [0, 1]; reliability diagram data is non-empty."""
    pipe, x_df, y = fitted_lr
    result = compute_calibration(pipe, x_df, y, model_name="lr")
    assert isinstance(result, CalibrationResults)
    assert 0.0 <= result.brier_score <= 1.0
    assert len(result.fraction_of_positives) > 0
    assert len(result.fraction_of_positives) == len(result.mean_predicted_value)


def test_calibration_perfect_brier() -> None:
    """Perfect predictions give Brier score = 0."""
    n = 20
    y_true = np.array([1] * 10 + [0] * 10)
    proba = y_true.astype(float)  # probability exactly matches label
    pipe = _mock_pipeline(y_true, proba)
    x_df = pd.DataFrame({"dummy": range(n)})
    y_ser = pd.Series(y_true)

    result = compute_calibration(pipe, x_df, y_ser, model_name="perfect")
    assert result.brier_score == pytest.approx(0.0)


def test_calibration_worst_brier() -> None:
    """Worst-case predictions (confident and wrong) give Brier score = 1."""
    n = 20
    y_true = np.array([1] * 10 + [0] * 10)
    proba = 1 - y_true.astype(float)  # completely inverted
    pipe = _mock_pipeline(y_true, proba)
    x_df = pd.DataFrame({"dummy": range(n)})
    y_ser = pd.Series(y_true)

    result = compute_calibration(pipe, x_df, y_ser, model_name="worst")
    assert result.brier_score == pytest.approx(1.0)


# ---------- compute_fairness_metrics ----------

def test_fairness_returns_fairness_results(fitted_lr, joined_df: pd.DataFrame) -> None:
    """compute_fairness_metrics returns FairnessResults with non-empty slices."""
    pipe, x_df, y = fitted_lr
    result = compute_fairness_metrics(
        pipe, x_df, y,
        model_name="lr",
        sensitive_features=["disability", "household_type"],
    )
    assert isinstance(result, FairnessResults)
    assert len(result.slices) > 0


def test_fairness_slice_metrics_in_range(fitted_lr) -> None:
    """All per-slice metrics are in valid ranges."""
    pipe, x_df, y = fitted_lr
    result = compute_fairness_metrics(
        pipe, x_df, y,
        model_name="lr",
        sensitive_features=["disability"],
    )
    for sl in result.slices:
        assert 0.0 <= sl.base_rate <= 1.0
        assert 0.0 <= sl.selection_rate <= 1.0
        assert 0.0 <= sl.tpr <= 1.0
        assert sl.roc_auc is None or 0.0 <= sl.roc_auc <= 1.0


def test_fairness_slice_count_matches_unique_values(
    fitted_lr, joined_df: pd.DataFrame
) -> None:
    """Number of slices equals number of unique values for the feature."""
    pipe, x_df, y = fitted_lr
    feature = "household_type"
    n_unique = x_df[feature].nunique()
    result = compute_fairness_metrics(
        pipe, x_df, y, model_name="lr", sensitive_features=[feature]
    )
    assert len(result.slices) == n_unique


def test_fairness_unknown_feature_skipped(fitted_lr) -> None:
    """A sensitive feature not in X_test is skipped, not raised."""
    pipe, x_df, y = fitted_lr
    result = compute_fairness_metrics(
        pipe, x_df, y,
        model_name="lr",
        sensitive_features=["nonexistent_col"],
    )
    assert result.slices == []


def test_fairness_roc_auc_none_when_single_class() -> None:
    """ROC-AUC is None for a subgroup that contains only one class."""
    # Build a small X with a feature whose one group is all-positive.
    n = 20
    y_true = np.array([1] * 10 + [0] * 10)
    proba = np.full(n, 0.6)
    pipe = _mock_pipeline(y_true, proba)
    x_df = pd.DataFrame({
        "group": ["a"] * 10 + ["b"] * 10,
        "dummy": range(n),
    })
    y_ser = pd.Series(y_true)

    result = compute_fairness_metrics(
        pipe, x_df, y_ser,
        model_name="test",
        sensitive_features=["group"],
    )
    # Group "a" has only positive labels → AUC undefined.
    slice_a = next(s for s in result.slices if s.value == "a")
    assert slice_a.roc_auc is None
    # Group "b" has only negative labels → AUC also undefined.
    slice_b = next(s for s in result.slices if s.value == "b")
    assert slice_b.roc_auc is None
