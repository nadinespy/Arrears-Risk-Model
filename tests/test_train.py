"""End-to-end tests for the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from arrears_risk_model.config import load_config
from arrears_risk_model.train import run_training


@pytest.fixture
def training_config(raw_household_df: pd.DataFrame, raw_imd_df: pd.DataFrame, tmp_path: Path):
    """Config with paths pointing to synthetic fixture files in tmp_path."""
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
    return config.model_copy(update={
        "paths": config.paths.model_copy(update={
            "household_data": household_path,
            "imd_data": imd_path,
            "model_dir": model_dir,
            "output_dir": output_dir,
        }),
        # Shrink the tuning grids so end-to-end tests stay fast on the
        # 50-row fixture. Tuning is still exercised — just over a 1-cell grid.
        "hyperparameter_search": config.hyperparameter_search.model_copy(update={
            "lr_grid": {"C": [1.0]},
            "xgb_grid": {"max_depth": [3]},
            "n_jobs": 1,  # avoid nested parallelism with XGBoost on tiny fixtures
        }),
    })


def test_run_training_returns_run_dir(training_config) -> None:
    """run_training completes and returns a path to the created run directory."""
    run_dir = run_training(training_config)
    assert isinstance(run_dir, Path)
    assert run_dir.is_dir()


def test_run_training_produces_pipeline_files(training_config) -> None:
    """Both joblib pipeline files are present in the run directory."""
    run_dir = run_training(training_config)
    assert (run_dir / "lr_pipeline.joblib").is_file()
    assert (run_dir / "xgb_pipeline.joblib").is_file()


def test_run_training_produces_metadata(training_config) -> None:
    """metadata.json is present and contains the expected top-level keys."""
    run_dir = run_training(training_config)
    metadata_path = run_dir / "metadata.json"
    assert metadata_path.is_file()

    with open(metadata_path) as f:
        meta = json.load(f)

    for key in ("trained_at", "n_train", "n_test", "features", "target",
                "config", "library_versions", "results"):
        assert key in meta, f"Expected key {key!r} missing from metadata.json"


def test_metadata_results_contain_both_models(training_config) -> None:
    """results block has entries for both lr and xgb."""
    run_dir = run_training(training_config)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    assert "lr" in meta["results"]
    assert "xgb" in meta["results"]
    for model_name in ("lr", "xgb"):
        for section in ("cv", "held_out", "calibration", "fairness"):
            assert section in meta["results"][model_name], (
                f"Missing {section!r} under results.{model_name}"
            )


def test_metadata_metric_ranges(training_config) -> None:
    """ROC-AUC values in metadata are in [0, 1]."""
    run_dir = run_training(training_config)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    for model_name in ("lr", "xgb"):
        cv = meta["results"][model_name]["cv"]
        assert 0.0 <= cv["roc_auc_mean"] <= 1.0
        held = meta["results"][model_name]["held_out"]
        assert 0.0 <= held["roc_auc"] <= 1.0


def test_saved_pipelines_are_loadable_and_predict(training_config) -> None:
    """Loaded pipelines produce valid predict_proba output."""
    run_dir = run_training(training_config)
    config = training_config

    # Re-load the household fixture to score against.
    household_path = config.paths.household_data
    imd_path = config.paths.imd_data

    from arrears_risk_model.data import prepare_dataset
    dataset = prepare_dataset(household_path, imd_path)
    x_df = dataset.drop(columns=[config.features.target])

    for fname in ("lr_pipeline.joblib", "xgb_pipeline.joblib"):
        pipe = joblib.load(run_dir / fname)
        proba = pipe.predict_proba(x_df)
        assert proba.shape == (len(x_df), 2)
        assert (proba >= 0).all() and (proba <= 1).all()


def test_metadata_records_tuning(training_config) -> None:
    """When tuning is enabled, cv.tuned is True and best_params is recorded."""
    run_dir = run_training(training_config)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    for model_name in ("lr", "xgb"):
        cv = meta["results"][model_name]["cv"]
        assert cv["tuned"] is True
        assert cv["best_params"] is not None


def test_metadata_no_tuning_when_disabled(training_config) -> None:
    """Disabling search keeps cv.tuned False and best_params None."""
    cfg = training_config.model_copy(update={
        "hyperparameter_search": training_config.hyperparameter_search.model_copy(
            update={"enabled": False}
        )
    })
    run_dir = run_training(cfg)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)

    for model_name in ("lr", "xgb"):
        cv = meta["results"][model_name]["cv"]
        assert cv["tuned"] is False
        assert cv["best_params"] is None


def test_metadata_uses_config_threshold(training_config) -> None:
    """held_out / fairness threshold matches config.evaluation.threshold."""
    cfg = training_config.model_copy(update={
        "evaluation": training_config.evaluation.model_copy(update={"threshold": 0.3})
    })
    run_dir = run_training(cfg)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    for model_name in ("lr", "xgb"):
        assert meta["results"][model_name]["held_out"]["threshold"] == 0.3
        assert meta["results"][model_name]["fairness"]["threshold"] == 0.3


def test_metadata_n_train_n_test_sum(training_config) -> None:
    """n_train + n_test should equal total dataset rows (50 in fixture)."""
    run_dir = run_training(training_config)
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["n_train"] + meta["n_test"] == 50
