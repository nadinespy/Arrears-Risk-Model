"""End-to-end tests for the prediction pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from arrears_risk_model.config import load_config
from arrears_risk_model.predict import (
    _CHILDREN_HOUSEHOLD_TYPES,
    apply_equity_overlay,
    load_latest_model,
    run_prediction,
)
from arrears_risk_model.predict import (
    main as predict_main,
)
from arrears_risk_model.train import run_training


@pytest.fixture
def predict_config(raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame, tmp_path: Path):
    """Config with tmp_path paths and pre-trained models already saved."""
    household_path = tmp_path / "household.xlsx"
    imd_path = tmp_path / "imd.xlsx"
    raw_household_df.to_excel(household_path, index=False)
    with pd.ExcelWriter(imd_path) as writer:
        raw_imd_df.to_excel(writer, sheet_name="IMD25", index=False)

    model_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    model_dir.mkdir()
    output_dir.mkdir()

    config = load_config()
    config = config.model_copy(update={
        "paths": config.paths.model_copy(update={
            "household_data": household_path,
            "imd_data": imd_path,
            "model_dir": model_dir,
            "output_dir": output_dir,
        }),
        # Same grid-shrinking as in test_train fixture — keeps the
        # train→predict round trip fast on the synthetic fixture.
        "hyperparameter_search": config.hyperparameter_search.model_copy(update={
            "lr_grid": {"C": [1.0]},
            "xgb_grid": {"max_depth": [3]},
            "n_jobs": 1,
        }),
    })
    run_training(config)
    return config


def test_run_prediction_returns_csv_path(predict_config) -> None:
    """run_prediction completes and returns a path to an existing CSV file."""
    output_path = run_prediction(predict_config)
    assert isinstance(output_path, Path)
    assert output_path.is_file()
    assert output_path.suffix == ".csv"


def test_output_has_expected_columns(predict_config) -> None:
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert list(df.columns) == ["reference", "predicted_proba", "composite_score", "rank"]


def test_output_covers_all_households(predict_config) -> None:
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert len(df) == 50


def test_rank_ordering_is_descending(predict_config) -> None:
    """composite_score is non-increasing; rank 1 is the top-scoring household."""
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert df["rank"].iloc[0] == 1
    assert df["composite_score"].is_monotonic_decreasing


def test_rank_is_consecutive(predict_config) -> None:
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert list(df["rank"]) == list(range(1, len(df) + 1))


def test_predicted_proba_in_unit_interval(predict_config) -> None:
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert (df["predicted_proba"] >= 0).all()
    assert (df["predicted_proba"] <= 1).all()


def test_composite_score_at_least_predicted_proba(predict_config) -> None:
    """Equity weights are non-negative so composite_score >= predicted_proba always."""
    output_path = run_prediction(predict_config)
    df = pd.read_csv(output_path)
    assert (df["composite_score"] >= df["predicted_proba"] - 1e-9).all()


def test_load_latest_model_raises_on_empty_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "no_runs"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No model runs"):
        load_latest_model(model_dir)


def test_load_latest_model_raises_on_wrong_model_name(predict_config) -> None:
    paths = predict_config.paths.resolved()
    with pytest.raises(FileNotFoundError):
        load_latest_model(paths.model_dir, model_name="nonexistent")


def test_load_latest_model_returns_pipeline(predict_config) -> None:
    paths = predict_config.paths.resolved()
    pipeline, metadata, run_dir = load_latest_model(paths.model_dir, "xgb")
    assert run_dir.is_dir()
    assert "trained_at" in metadata
    assert hasattr(pipeline, "predict_proba")


def test_input_flag_overrides_household_path(
    predict_config, raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame,
    tmp_path: Path, monkeypatch,
) -> None:
    """`arrears-predict --input <file>` scores the supplied file, not the configured one."""
    # Trained model + the original household fixture already exist via predict_config.
    # Build a *different* household file with a known marker reference and point --input at it.
    fresh = raw_household_df.copy()
    fresh["reference"] = [f"NEW{i:05d}" for i in range(len(fresh))]
    fresh_path = tmp_path / "refreshed.xlsx"
    fresh.to_excel(fresh_path, index=False)

    paths = predict_config.paths.resolved()
    monkeypatch.setattr(sys, "argv", [
        "arrears-predict",
        "--input", str(fresh_path),
        "--model-dir", str(paths.model_dir),
        "--output-dir", str(paths.output_dir),
    ])
    # Make the IMD path for this run resolvable: the default config points at a path
    # that doesn't exist in tmp; rewrite via env var.
    monkeypatch.setenv("ARM_PATHS__IMD_DATA", str(paths.imd_data))
    monkeypatch.setenv("ARM_PATHS__HOUSEHOLD_DATA", str(paths.household_data))
    monkeypatch.setenv("ARM_PATHS__LEWISHAM_GEOJSON", str(paths.lewisham_geojson))

    predict_main()

    # The output CSV must reference the NEW* rows from --input, not the old fixture.
    csvs = sorted(paths.output_dir.glob("*_predictions.csv"))
    assert csvs, "no predictions CSV produced"
    df_out = pd.read_csv(csvs[-1])
    assert df_out["reference"].str.startswith("NEW").all()


def test_equity_overlay_children_uplift(joined_df: pd.DataFrame) -> None:
    """Children-eligible households receive at least the configured children weight."""
    config = load_config()
    base = pd.Series(0.5, index=joined_df.index)
    composite = apply_equity_overlay(base, joined_df, config)

    children_mask = (
        joined_df["household_type"].isin(_CHILDREN_HOUSEHOLD_TYPES)
        | (joined_df["fsm_eligible"] == 1)
    )
    if children_mask.any():
        assert (
            composite[children_mask] >= base[children_mask] + config.equity.children - 1e-9
        ).all()


def test_equity_overlay_disability_uplift(joined_df: pd.DataFrame) -> None:
    """Disability-eligible households receive at least the configured disability weight."""
    config = load_config()
    base = pd.Series(0.5, index=joined_df.index)
    composite = apply_equity_overlay(base, joined_df, config)

    disability_mask = joined_df["disability"] == 1
    if disability_mask.any():
        assert (
            composite[disability_mask] >= base[disability_mask] + config.equity.disability - 1e-9
        ).all()


def test_equity_overlay_no_uplift_for_ineligible(joined_df: pd.DataFrame) -> None:
    """Households with no children or disability flags receive no uplift."""
    config = load_config()
    base = pd.Series(0.5, index=joined_df.index)
    composite = apply_equity_overlay(base, joined_df, config)

    children_mask = (
        joined_df["household_type"].isin(_CHILDREN_HOUSEHOLD_TYPES)
        | (joined_df["fsm_eligible"] == 1)
    )
    disability_mask = joined_df["disability"] == 1
    no_overlay = ~children_mask & ~disability_mask
    if no_overlay.any():
        assert (composite[no_overlay] == base[no_overlay]).all()
