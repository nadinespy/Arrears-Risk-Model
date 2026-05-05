"""Full sklearn Pipeline factories for the arrears risk model.

Each factory combines a preprocessing pipeline (from :mod:`features`)
with an estimator configured from :class:`Config`. The resulting
``Pipeline`` objects are self-contained: fit on training data, they
can be serialised with ``joblib.dump`` and reloaded for inference
without any external state.

Usage::

    config = load_config()
    lr_pipe  = make_lr_pipeline(config)
    xgb_pipe = make_xgb_pipeline(config)

    lr_pipe.fit(X_train, y_train)
    proba = lr_pipe.predict_proba(X_new)[:, 1]

Class imbalance for XGBoost is handled by :class:`XGBClassifierAutoSPW`,
which recomputes ``scale_pos_weight`` from ``y`` at every ``.fit()`` call.
This guarantees that during cross-validation each fold derives its own
value from its own training portion — eliminating the small leakage that
would result from a value computed once on the full training set.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from arrears_risk_model.config import Config
from arrears_risk_model.features import make_lr_preprocessor, make_xgb_preprocessor


class XGBClassifierAutoSPW(XGBClassifier):
    """XGBClassifier that recomputes ``scale_pos_weight`` from y at fit time.

    Standard usage of ``scale_pos_weight = n_neg / n_pos`` requires the
    ratio to be computed before fitting. If the value is computed once
    on the full training set and then passed into a CV loop, each fold
    sees a value derived in part from its own validation portion — a
    small but real leak. Computing it inside ``fit`` makes the value a
    function of the data the estimator is currently being trained on,
    which is exactly the contract sklearn's CV expects.
    """

    def fit(self, x, y, **kwargs):
        # sklearn's convention is uppercase X; repo uses lowercase for consistency.
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        return super().fit(x, y, **kwargs)


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


def make_xgb_pipeline(config: Config) -> Pipeline:
    """Return a fitted-ready XGBoost pipeline: preprocessing → XGBClassifier.

    Class imbalance is handled inside the estimator: see
    :class:`XGBClassifierAutoSPW`.
    """
    clf = XGBClassifierAutoSPW(
        n_estimators=config.models.xgb.n_estimators,
        max_depth=config.models.xgb.max_depth,
        learning_rate=config.models.xgb.learning_rate,
        subsample=config.models.xgb.subsample,
        colsample_bytree=config.models.xgb.colsample_bytree,
        eval_metric=config.models.xgb.eval_metric,
        random_state=config.training.random_state,
    )
    return Pipeline([
        ("preprocessor", make_xgb_preprocessor(config)),
        ("clf", clf),
    ])


__all__ = ["XGBClassifierAutoSPW", "make_lr_pipeline", "make_xgb_pipeline"]
