"""Prediction CLI for the arrears risk model.

Loads the most recently trained pipeline from ``models/``, scores all
households in the input data, applies the config-driven equity overlay,
and writes a ranked priority list to ``outputs/``.

**Usage**::

    # default model (xgb), default config and paths
    python -m arrears_risk_model.predict

    # use the LR pipeline instead
    python -m arrears_risk_model.predict --model lr

    # custom config file
    python -m arrears_risk_model.predict --config path/to/custom.yaml

The installed script entry point is ``arrears-predict``.

**Output columns**

``reference``
    Household identifier, unchanged from the source data.
``predicted_proba``
    Raw positive-class probability from the pipeline.
``composite_score``
    ``predicted_proba`` + applicable equity weights (see below).
``rank``
    1 = highest priority. Sorted descending by ``composite_score``.

**Equity overlay**

A small additive weight is applied before ranking:

- ``+config.equity.children`` if ``household_type`` ∈ {Lone parent,
  Couple with children} or ``fsm_eligible = 1``.
- ``+config.equity.disability`` if ``disability = 1``.

Weights are configured in ``config/default.yaml`` and represent a
transparent policy choice, not a statistical adjustment. See
``docs/model_card.md`` for the framing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from arrears_risk_model.config import Config, load_config
from arrears_risk_model.data import prepare_dataset
from arrears_risk_model.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

_CHILDREN_HOUSEHOLD_TYPES = {"Lone parent", "Couple with children"}


def load_latest_model(model_dir: Path | str, model_name: str = "xgb") -> tuple[Any, dict, Path]:
    """Load the most recently trained pipeline from ``model_dir``.

    Returns ``(pipeline, metadata_dict, run_dir)``. Run directories are
    named by ISO timestamp so lexicographic sort gives chronological order.

    Raises ``FileNotFoundError`` if no runs exist or the requested model
    file is absent from the latest run.
    """
    model_dir = Path(model_dir)
    run_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No model runs found in {model_dir}. Run arrears-train first.")

    latest = run_dirs[-1]
    pipeline_path = latest / f"{model_name}_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"No {model_name!r} pipeline in {latest}. "
            f"Available files: {[p.name for p in latest.iterdir()]}"
        )

    pipeline = joblib.load(pipeline_path)
    with open(latest / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(
        "Loaded %s pipeline from %s (trained %s)", model_name, latest, metadata["trained_at"]
    )
    return pipeline, metadata, latest


def apply_equity_overlay(
    predicted_proba: pd.Series,
    x_df: pd.DataFrame,
    config: Config,
) -> pd.Series:
    """Add config-driven equity weights to ``predicted_proba``.

    Returns a new Series; the input is not mutated.
    """
    composite = predicted_proba.copy()

    children_mask = (
        x_df["household_type"].isin(_CHILDREN_HOUSEHOLD_TYPES)
        | (x_df["fsm_eligible"] == 1)
    )
    composite[children_mask] += config.equity.children

    disability_mask = x_df["disability"] == 1
    composite[disability_mask] += config.equity.disability

    logger.info(
        "Equity overlay applied: %d children-eligible, %d disability-eligible households",
        int(children_mask.sum()),
        int(disability_mask.sum()),
    )
    return composite


def run_prediction(config: Config, model_name: str = "xgb") -> Path:
    """Score all households and write a ranked priority list to ``outputs/``.

    Returns the path to the saved CSV file.
    """
    paths = config.paths.resolved()

    pipeline, _metadata, run_dir = load_latest_model(paths.model_dir, model_name)

    logger.info("Loading data for scoring")
    dataset = prepare_dataset(paths.household_data, paths.imd_data)

    target = config.features.target
    references = dataset["reference"].reset_index(drop=True)
    x_df = dataset.drop(columns=[target]).reset_index(drop=True)

    predicted_proba = pd.Series(
        pipeline.predict_proba(x_df)[:, 1],
        index=x_df.index,
        name="predicted_proba",
    )
    composite_score = apply_equity_overlay(predicted_proba, x_df, config)
    composite_score.name = "composite_score"

    output = pd.DataFrame({
        "reference": references,
        "predicted_proba": predicted_proba,
        "composite_score": composite_score,
    })
    output = (
        output
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    output["rank"] = output.index + 1

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = paths.output_dir / f"{timestamp}_predictions.csv"
    output.to_csv(output_path, index=False)

    logger.info(
        "Priority list saved to %s (%d households, model run: %s)",
        output_path, len(output), run_dir.name,
    )
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Score households and produce a ranked arrears-risk priority list."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a custom YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--model",
        choices=["lr", "xgb"],
        default="xgb",
        help="Which trained pipeline to use (default: xgb)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override the directory to search for trained pipelines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the directory where the priority list is written",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)

    if args.model_dir:
        config = config.model_copy(update={
            "paths": config.paths.model_copy(update={"model_dir": args.model_dir})
        })
    if args.output_dir:
        config = config.model_copy(update={
            "paths": config.paths.model_copy(update={"output_dir": args.output_dir})
        })

    run_prediction(config, model_name=args.model)


if __name__ == "__main__":
    main()
