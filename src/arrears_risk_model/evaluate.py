"""Evaluation functions for the arrears risk model.

All public functions return typed pydantic result objects so callers
(e.g. ``train.py``) can serialise them to JSON without extra glue.

Five functions are provided:

- :func:`cross_validate_model` — stratified K-fold CV without tuning;
  returns mean/std for ROC-AUC, PR-AUC, and F1 across folds.
- :func:`tune_and_cross_validate` — grid search via stratified K-fold CV
  (i.e. tuning *and* CV in one call); returns the refit best estimator
  plus the same metric statistics for the winning configuration.
- :func:`evaluate_held_out` — metrics on the held-out test set for a
  pipeline that is already fitted.
- :func:`compute_calibration` — reliability diagram data (fraction of
  positives vs mean predicted probability per bin) plus Brier score.
- :func:`compute_fairness_metrics` — descriptive per-subgroup metrics
  (selection rate, TPR, FPR, precision, ROC-AUC) sliced by nominated
  sensitive features. No pass/fail gate — these are for documentation
  and monitoring, not automated deployment decisions.

All metrics operate on binary 0/1 targets and on the positive-class
probability from ``predict_proba(...)[: , 1]``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from arrears_risk_model.logging_config import get_logger

logger = get_logger(__name__)


class _ResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CVResults(_ResultBase):
    """Cross-validation summary for one model.

    When ``tuned`` is True, the metric statistics describe the best
    hyperparameter configuration found by grid search, and
    ``best_params`` records that configuration (without the ``clf__``
    prefix used internally).
    """

    model_name: str
    n_splits: int
    roc_auc_mean: float
    roc_auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    f1_mean: float
    f1_std: float
    tuned: bool = False
    best_params: dict | None = None


class HeldOutResults(_ResultBase):
    """Metrics on the held-out test set for one fitted model."""

    model_name: str
    n_test: int
    threshold: float
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float


class CalibrationResults(_ResultBase):
    """Reliability diagram data and Brier score for one fitted model."""

    model_name: str
    n_bins: int
    brier_score: float
    # Reliability diagram: fraction_of_positives[i] is the observed
    # positive rate in the i-th probability bin; mean_predicted_value[i]
    # is the mean predicted probability for that bin.
    fraction_of_positives: list[float]
    mean_predicted_value: list[float]


class FairnessSlice(_ResultBase):
    """Descriptive metrics for one subgroup of one sensitive feature."""

    feature: str
    value: str
    n: int
    base_rate: float        # observed positive rate in this slice
    selection_rate: float   # predicted positive rate at threshold
    tpr: float              # true-positive rate (recall) in this slice
    fpr: float              # false-positive rate in this slice
    precision: float        # precision in this slice
    # None when the slice contains only one class (AUC undefined).
    roc_auc: float | None


class FairnessResults(_ResultBase):
    """All fairness slices for one fitted model."""

    model_name: str
    threshold: float
    slices: list[FairnessSlice]


def cross_validate_model(
    pipeline: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: Any,
    model_name: str,
) -> CVResults:
    """Stratified K-fold CV; returns mean ± std for ROC-AUC, PR-AUC, F1.

    The pipeline is cloned once per fold by sklearn — it should not be
    pre-fitted. XGBoost's ``scale_pos_weight`` is recomputed per-fit by
    :class:`~arrears_risk_model.models.XGBClassifierAutoSPW`, so each fold
    derives its own value from its own training portion (no leakage).
    """
    cv = StratifiedKFold(
        n_splits=config.training.cv_n_splits,
        shuffle=True,
        random_state=config.training.random_state,
    )
    scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        return_train_score=False,
    )
    result = CVResults(
        model_name=model_name,
        n_splits=config.training.cv_n_splits,
        roc_auc_mean=float(scores["test_roc_auc"].mean()),
        roc_auc_std=float(scores["test_roc_auc"].std()),
        pr_auc_mean=float(scores["test_pr_auc"].mean()),
        pr_auc_std=float(scores["test_pr_auc"].std()),
        f1_mean=float(scores["test_f1"].mean()),
        f1_std=float(scores["test_f1"].std()),
    )
    logger.info(
        "%s CV: ROC-AUC %.3f ± %.3f | PR-AUC %.3f ± %.3f | F1 %.3f ± %.3f",
        model_name,
        result.roc_auc_mean, result.roc_auc_std,
        result.pr_auc_mean, result.pr_auc_std,
        result.f1_mean, result.f1_std,
    )
    return result


def tune_and_cross_validate(
    pipeline: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list],
    config: Any,
    model_name: str,
) -> tuple[Any, CVResults]:
    """Grid-search ``pipeline``'s ``clf`` step over ``param_grid``.

    The grid is supplied without the ``clf__`` prefix (e.g. ``{"C": [...]}``);
    the prefix is added here so callers don't need to know how the
    pipeline is wired internally. Multi-metric scoring (ROC-AUC, PR-AUC,
    F1) lets ``CVResults`` carry the same shape as without tuning, with
    the refit picking the best configuration on the configured primary
    scorer.

    Returns ``(best_estimator, cv_results)`` where ``best_estimator`` has
    already been refit on the full training set by GridSearchCV.
    """
    if not param_grid:
        raise ValueError(f"Empty param_grid for {model_name!r} — tuning has nothing to search")

    prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    cv = StratifiedKFold(
        n_splits=config.training.cv_n_splits,
        shuffle=True,
        random_state=config.training.random_state,
    )
    search = GridSearchCV(
        pipeline,
        param_grid=prefixed_grid,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"},
        refit=config.hyperparameter_search.scoring,
        n_jobs=config.hyperparameter_search.n_jobs,
        return_train_score=False,
    )
    search.fit(x_train, y_train)

    best = search.best_index_
    cv_res = CVResults(
        model_name=model_name,
        n_splits=config.training.cv_n_splits,
        roc_auc_mean=float(search.cv_results_["mean_test_roc_auc"][best]),
        roc_auc_std=float(search.cv_results_["std_test_roc_auc"][best]),
        pr_auc_mean=float(search.cv_results_["mean_test_pr_auc"][best]),
        pr_auc_std=float(search.cv_results_["std_test_pr_auc"][best]),
        f1_mean=float(search.cv_results_["mean_test_f1"][best]),
        f1_std=float(search.cv_results_["std_test_f1"][best]),
        tuned=True,
        best_params={k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
    )
    logger.info(
        "%s tuned: best_params=%s | ROC-AUC %.3f ± %.3f | PR-AUC %.3f ± %.3f | F1 %.3f ± %.3f",
        model_name,
        cv_res.best_params,
        cv_res.roc_auc_mean, cv_res.roc_auc_std,
        cv_res.pr_auc_mean, cv_res.pr_auc_std,
        cv_res.f1_mean, cv_res.f1_std,
    )
    return search.best_estimator_, cv_res


def evaluate_held_out(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> HeldOutResults:
    """Compute metrics on a held-out test set for an already-fitted pipeline."""
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    result = HeldOutResults(
        model_name=model_name,
        n_test=len(y_test),
        threshold=threshold,
        roc_auc=float(roc_auc_score(y_test, y_proba)),
        pr_auc=float(average_precision_score(y_test, y_proba)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
    )
    logger.info(
        "%s held-out: ROC-AUC %.3f | PR-AUC %.3f | F1 %.3f | "
        "precision %.3f | recall %.3f",
        model_name,
        result.roc_auc, result.pr_auc, result.f1,
        result.precision, result.recall,
    )
    return result


def compute_calibration(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_bins: int = 10,
) -> CalibrationResults:
    """Reliability diagram data + Brier score for an already-fitted pipeline."""
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
    brier = float(brier_score_loss(y_test, y_proba))

    result = CalibrationResults(
        model_name=model_name,
        n_bins=n_bins,
        brier_score=brier,
        fraction_of_positives=frac_pos.tolist(),
        mean_predicted_value=mean_pred.tolist(),
    )
    logger.info("%s Brier score: %.4f", model_name, brier)
    return result


def compute_fairness_metrics(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    sensitive_features: list[str],
    threshold: float = 0.5,
) -> FairnessResults:
    """Descriptive subgroup metrics for nominated sensitive features.

    Slices by each unique value of each feature in ``sensitive_features``.
    All metrics are descriptive — there is no automated pass/fail gate.
    See ``docs/model_card.md`` for discussion of formal fairness criteria
    and the impossibility result that prevents satisfying all simultaneously.
    """
    y_proba = pd.Series(
        pipeline.predict_proba(x_test)[:, 1], index=x_test.index
    )
    y_pred = (y_proba >= threshold).astype(int)

    slices: list[FairnessSlice] = []
    for feature in sensitive_features:
        if feature not in x_test.columns:
            logger.warning("Sensitive feature %r not in X_test — skipping", feature)
            continue
        for value in sorted(x_test[feature].unique(), key=str):
            mask = x_test[feature] == value
            y_t = y_test[mask]
            y_p = y_pred[mask]
            y_pr = y_proba[mask]
            n = int(mask.sum())

            if n == 0:
                continue

            base_rate = float(y_t.mean())
            selection_rate = float(y_p.mean())

            n_pos = int(y_t.sum())
            n_neg = n - n_pos

            tpr = float(recall_score(y_t, y_p, zero_division=0))
            fpr = (
                float(((y_p == 1) & (y_t == 0)).sum() / n_neg)
                if n_neg > 0 else float("nan")
            )
            prec = float(precision_score(y_t, y_p, zero_division=0))

            # ROC-AUC is undefined when the slice contains only one class.
            auc = float(roc_auc_score(y_t, y_pr)) if n_pos > 0 and n_neg > 0 else None

            slices.append(FairnessSlice(
                feature=feature,
                value=str(value),
                n=n,
                base_rate=base_rate,
                selection_rate=selection_rate,
                tpr=tpr,
                fpr=fpr,
                precision=prec,
                roc_auc=auc,
            ))

    logger.info(
        "%s fairness: computed %d slices across %d feature(s)",
        model_name, len(slices), len(sensitive_features),
    )
    return FairnessResults(
        model_name=model_name,
        threshold=threshold,
        slices=slices,
    )


__all__ = [
    "CVResults",
    "CalibrationResults",
    "FairnessResults",
    "FairnessSlice",
    "HeldOutResults",
    "compute_calibration",
    "compute_fairness_metrics",
    "cross_validate_model",
    "evaluate_held_out",
    "tune_and_cross_validate",
]
