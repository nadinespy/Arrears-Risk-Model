"""Training CLI for the arrears risk model.

Orchestration::

    load config
    → load & validate data (prepare_dataset)
    → stratified train/test split
    → for each model (LR, XGBoost):
          cross-validate on training set
          fit on full training set
          evaluate on held-out test set
          compute calibration + fairness metrics
    → save artefacts to models/{timestamp}/

Artefacts written per run::

    models/{timestamp}/
    ├── lr_pipeline.joblib     fitted sklearn Pipeline (preprocessor + LR)
    ├── xgb_pipeline.joblib    fitted sklearn Pipeline (preprocessor + XGB)
    └── metadata.json          training context + all evaluation results

**Usage**::

    # default config + default data paths
    python -m arrears_risk_model.train

    # custom config file
    python -m arrears_risk_model.train --config path/to/custom.yaml

    # override model output directory
    python -m arrears_risk_model.train --model-dir /tmp/models

    # or via environment variable
    ARM_PATHS__MODEL_DIR=/tmp/models python -m arrears_risk_model.train

The installed script entry point is ``arrears-train``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

from arrears_risk_model.config import Config, load_config
from arrears_risk_model.data import prepare_dataset
from arrears_risk_model.evaluate import (
    compute_calibration,
    compute_fairness_metrics,
    cross_validate_model,
    evaluate_held_out,
)
from arrears_risk_model.logging_config import configure_logging, get_logger
from arrears_risk_model.models import make_lr_pipeline, make_xgb_pipeline

logger = get_logger(__name__)

# Features used for fairness slicing. These are the columns in the
# pre-preprocessed X DataFrame that correspond to protected/sensitive
# characteristics. Kept here rather than in config because they are a
# modelling/audit concern, not a tuning knob.
SENSITIVE_FEATURES = ["disability", "household_type", "age_bracket"]


def run_training(config: Config) -> Path:
    """Execute the full training run and return the path to the run directory.

    This function is the testable core; ``main()`` is the CLI wrapper.
    """
    paths = config.paths.resolved()

    # --- Load data ----------------------------------------------------------
    logger.info("Loading data")
    dataset = prepare_dataset(paths.household_data, paths.imd_data)
    logger.info("Dataset ready: %d rows, %d columns", len(dataset), len(dataset.columns))

    # --- Split --------------------------------------------------------------
    target = config.features.target
    x_df = dataset.drop(columns=[target])
    y = dataset[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=y if config.training.cv_stratify else None,
    )
    logger.info(
        "Split: %d train / %d test  (positive rate: train=%.1f%% test=%.1f%%)",
        len(x_train), len(x_test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    # --- Compute class ratio for XGBoost ------------------------------------
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info("XGB scale_pos_weight derived from training set: %.2f", spw)

    # --- Train and evaluate -------------------------------------------------
    model_specs = [
        ("lr",  make_lr_pipeline(config)),
        ("xgb", make_xgb_pipeline(config, scale_pos_weight=spw)),
    ]

    all_results: dict[str, dict] = {}
    fitted_pipelines: dict[str, object] = {}

    for model_name, pipeline in model_specs:
        logger.info("=== %s ===", model_name.upper())

        cv_res = cross_validate_model(pipeline, x_train, y_train, config, model_name)

        logger.info("Fitting %s on full training set", model_name)
        pipeline.fit(x_train, y_train)

        held_out_res = evaluate_held_out(pipeline, x_test, y_test, model_name)
        calibration_res = compute_calibration(pipeline, x_test, y_test, model_name)
        fairness_res = compute_fairness_metrics(
            pipeline, x_test, y_test, model_name, SENSITIVE_FEATURES
        )

        all_results[model_name] = {
            "cv": cv_res,
            "held_out": held_out_res,
            "calibration": calibration_res,
            "fairness": fairness_res,
        }
        fitted_pipelines[model_name] = pipeline

    # --- Save artefacts -----------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = paths.model_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for model_name, pipeline in fitted_pipelines.items():
        pipeline_path = run_dir / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)
        logger.info("Saved %s to %s", model_name, pipeline_path)

    metadata = {
        "trained_at": timestamp,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "features": config.features.all_input_features,
        "target": target,
        "config": config.model_dump(mode="json"),
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "results": {
            name: {
                "cv": r["cv"].model_dump(),
                "held_out": r["held_out"].model_dump(),
                "calibration": r["calibration"].model_dump(),
                "fairness": r["fairness"].model_dump(),
            }
            for name, r in all_results.items()
        },
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Training complete. Artefacts in %s", run_dir)
    return run_dir


def main() -> None:
    """CLI entry point. Parses args, loads config, runs training."""
    parser = argparse.ArgumentParser(
        description="Train the arrears risk model and save artefacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a custom YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override the model output directory (also: ARM_PATHS__MODEL_DIR)",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)

    if args.model_dir:
        config = config.model_copy(update={
            "paths": config.paths.model_copy(update={"model_dir": args.model_dir})
        })

    run_training(config)


if __name__ == "__main__":
    main()
