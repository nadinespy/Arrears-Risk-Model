"""Tests for the full LR and x_outGBoost pipeline factories."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from arrears_risk_model.config import load_config
from arrears_risk_model.models import make_lr_pipeline, make_xgb_pipeline


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def x_out_y(joined_df: pd.DataFrame):
    """Split joined fixture into features frame and binary target."""
    y = joined_df["arrears_flag"]
    x_out = joined_df.drop(columns=["arrears_flag"])
    return x_out, y


# ---------- LR pipeline ----------

def test_lr_pipeline_fits(config, x_out_y) -> None:
    """LR pipeline fits on synthetic data without error."""
    x_train, y_train = x_out_y
    pipe = make_lr_pipeline(config)
    pipe.fit(x_train, y_train)


def test_lr_pipeline_predict_proba(config, x_out_y) -> None:
    """predict_proba returns an (n, 2) array with values in [0, 1]."""
    x_train, y_train = x_out_y
    pipe = make_lr_pipeline(config)
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_train)
    assert proba.shape == (len(x_train), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_lr_pipeline_positive_class_index(config, x_out_y) -> None:
    """The positive class (arrears = 1) is at index 1 of predict_proba output."""
    x_train, y_train = x_out_y
    pipe = make_lr_pipeline(config)
    pipe.fit(x_train, y_train)
    assert list(pipe.classes_) == [0, 1]


def test_lr_pipeline_serialises(config, x_out_y, tmp_path) -> None:
    """Fitted pipeline round-trips through joblib without error or score change."""
    x_train, y_train = x_out_y
    pipe = make_lr_pipeline(config)
    pipe.fit(x_train, y_train)
    proba_before = pipe.predict_proba(x_train)

    path = tmp_path / "lr_pipeline.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    proba_after = loaded.predict_proba(x_train)
    np.testing.assert_array_equal(proba_before, proba_after)


# ---------- x_outGB pipeline ----------

def test_xgb_pipeline_fits(config, x_out_y) -> None:
    """x_outGB pipeline fits on synthetic data without error."""
    x_train, y_train = x_out_y
    pipe = make_xgb_pipeline(config)
    pipe.fit(x_train, y_train)


def test_xgb_pipeline_predict_proba(config, x_out_y) -> None:
    """predict_proba returns an (n, 2) array with values in [0, 1]."""
    x_train, y_train = x_out_y
    pipe = make_xgb_pipeline(config)
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_train)
    assert proba.shape == (len(x_train), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_xgb_pipeline_serialises(config, x_out_y, tmp_path) -> None:
    """Fitted x_outGB pipeline round-trips through joblib."""
    x_train, y_train = x_out_y
    pipe = make_xgb_pipeline(config)
    pipe.fit(x_train, y_train)
    proba_before = pipe.predict_proba(x_train)

    path = tmp_path / "xgb_pipeline.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    proba_after = loaded.predict_proba(x_train)
    np.testing.assert_array_equal(proba_before, proba_after)


def test_xgb_scale_pos_weight_recomputed_per_fit(config, x_out_y) -> None:
    """scale_pos_weight is recomputed from y at every .fit() — ensures
    each CV fold gets its own value derived from its own training portion."""
    x_train, y_train = x_out_y
    pipe = make_xgb_pipeline(config)
    pipe.fit(x_train, y_train)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    expected_full = n_neg / n_pos if n_pos > 0 else 1.0
    assert pipe.named_steps["clf"].scale_pos_weight == pytest.approx(expected_full)

    # Refit on a balanced subset → scale_pos_weight must change accordingly.
    pos_idx = y_train.index[y_train == 1][:5]
    neg_idx = y_train.index[y_train == 0][:5]
    idx = pos_idx.append(neg_idx)
    pipe.fit(x_train.loc[idx], y_train.loc[idx])
    assert pipe.named_steps["clf"].scale_pos_weight == pytest.approx(1.0)
