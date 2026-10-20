from __future__ import annotations
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from .models import CORE, NEWS, CALENDAR, EXTERNAL, training_positions

FEATURES = CORE + NEWS + CALENDAR + EXTERNAL


OBJECTIVES = {"l1": "regression_l1", "huber": "huber"}


def model_name(loss, strategy):
    return f"fi_{loss}_{strategy}"


def fit_estimator(x, vintages, origin, horizon, loss, features, seed):
    rows = training_positions(origin, horizon)
    y = vintages.iloc[origin, rows + horizon].to_numpy(float)
    rows, y = rows[np.isfinite(y)], y[np.isfinite(y)]
    estimator = LGBMRegressor(
        objective=OBJECTIVES[loss], n_estimators=100, learning_rate=.035,
        max_depth=3, num_leaves=7, min_child_samples=15, reg_lambda=10.,
        reg_alpha=.1, random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True, importance_type="gain")
    estimator.fit(x.iloc[rows][features], y)
    return estimator


def gain_ranking(models, features=FEATURES):
    """Normalize within each horizon before averaging, avoiding gain scale bias."""
    table = pd.DataFrame({"feature": features, "original_position": range(len(features))})
    normalized = []
    for h, estimator in sorted(models.items()):
        if list(estimator.feature_name_) != list(features):
            raise ValueError("Feature names and importance positions differ")
        gain = estimator.booster_.feature_importance(importance_type="gain").astype(float)
        if not np.isfinite(gain).all() or (gain < 0).any():
            raise ValueError("Invalid gain importance")
        table[f"gain_h{h}"] = gain
        fraction = gain / gain.sum() if gain.sum() else np.zeros(len(gain))
        table[f"gain_fraction_h{h}"] = fraction
        normalized.append(fraction)
    table["importance"] = np.mean(normalized, axis=0)
    return table.sort_values(["importance", "original_position"], ascending=[False, True],
                             kind="stable").reset_index(drop=True)


def select_features(ranking, strategy, features=FEATURES):
    if strategy == "full":
        return list(features)
    positive = ranking[ranking.importance.gt(0)]
    if positive.empty:
        raise ValueError("Full model has no positive feature importance")
    if strategy == "top05":
        count = min(5, len(positive))
    elif strategy == "nonzero":
        count = len(positive)
    else:
        raise ValueError(f"Unknown feature selection strategy: {strategy}")
    chosen = set(positive.feature.iloc[:count])
    return [f for f in features if f in chosen]


def fit_origin(matrices, vintages, origin, seed, policies):
    """Feature selection and refitting see the same historical train labels only."""
    fitted, definitions = {}, {}
    for loss in OBJECTIVES:
        full = {h: fit_estimator(x, vintages, origin, h, loss, FEATURES, seed)
                for h, x in matrices.items()}
        ranking = gain_ranking(full)
        strategy = policies[loss]
        features = select_features(ranking, strategy)
        name = model_name(loss, strategy)
        fitted[name] = full if features == FEATURES else {
            h: fit_estimator(x, vintages, origin, h, loss, features, seed) for h, x in matrices.items()}
        definitions[name] = {'features': features}
    return fitted, definitions


def predict(matrices, origin, fitted, definitions):
    values = {}
    for name, models in fitted.items():
        features = definitions[name]["features"]
        values[name] = np.array([max(0., float(estimator.predict(matrices[h].iloc[[origin]][features])[0]))
                                 for h, estimator in sorted(models.items())])
        if not np.isfinite(values[name]).all():
            raise ValueError(f"Nonfinite forecast: {name}")
    return values
