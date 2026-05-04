"""Full sklearn Pipeline factories for the arrears risk model.

Each factory combines a preprocessing pipeline (from :mod:`features`)
with an estimator configured from :class:`Config`. The resulting
``Pipeline`` objects are self-contained: fit on training data, they
can be serialised with ``joblib.dump`` and reloaded for inference
without any external state.

Usage::

    config = load_config()
    lr_pipe  = make_lr_pipeline(config)
    xgb_pipe = make_xgb_pipeline(config, scale_pos_weight=spw)

    lr_pipe.fit(X_train, y_train)
    proba = lr_pipe.predict_proba(X_new)[:, 1]

``scale_pos_weight`` for XGBoost is derived from the training-set class
ratio in ``train.py``; the factory accepts it as an optional override so
the parameter stays close to where it is computed rather than being
buried in config.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from arrears_risk_model.config import Config
from arrears_risk_model.features import make_lr_preprocessor, make_xgb_preprocessor


def make_lr_pipeline(config: Config) -> Pipeline:
    """Return a fitted-ready LR pipeline: preprocessing → LogisticRegression.

    Hyperparameters come from ``config.models.lr``; the random state is
    shared with the rest of the pipeline via ``config.training.random_state``.
    """
    clf = LogisticRegression(
        C=config.models.lr.C,
        l1_ratio=config.models.lr.l1_ratio,
        solver=config.models.lr.solver,
        max_iter=config.models.lr.max_iter,
        class_weight=config.models.lr.class_weight,
        random_state=config.training.random_state,
    )
    return Pipeline([
        ("preprocessor", make_lr_preprocessor(config)),
        ("clf", clf),
    ])


def make_xgb_pipeline(
    config: Config,
    scale_pos_weight: float | None = None,
) -> Pipeline:
    """Return a fitted-ready XGBoost pipeline: preprocessing → XGBClassifier.

    ``scale_pos_weight`` adjusts for class imbalance. When ``None`` here
    *and* ``None`` in config, XGBoost defaults to 1.0 (no reweighting).
    ``train.py`` computes ``neg / pos`` from the training split and passes
    it as the override when config leaves it unset.
    """
    spw = scale_pos_weight if scale_pos_weight is not None else config.models.xgb.scale_pos_weight
    clf = XGBClassifier(
        n_estimators=config.models.xgb.n_estimators,
        max_depth=config.models.xgb.max_depth,
        learning_rate=config.models.xgb.learning_rate,
        subsample=config.models.xgb.subsample,
        colsample_bytree=config.models.xgb.colsample_bytree,
        scale_pos_weight=spw,
        eval_metric=config.models.xgb.eval_metric,
        random_state=config.training.random_state,
    )
    return Pipeline([
        ("preprocessor", make_xgb_preprocessor(config)),
        ("clf", clf),
    ])


__all__ = ["make_lr_pipeline", "make_xgb_pipeline"]
