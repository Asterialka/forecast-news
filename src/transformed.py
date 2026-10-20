from __future__ import annotations
import numpy as np

from lightgbm import LGBMRegressor
from .models import training_positions

def transform_target(y, anchor, kind):
    if kind == "delta":
        return y - anchor
    if kind == "log_delta":
        return np.log1p(y) - np.log1p(anchor)
    return y


def restore_target(predicted, anchor, kind):
    if kind == "delta":
        predicted = anchor + predicted
    elif kind == "log_delta":
        predicted = np.expm1(np.log1p(anchor) + predicted)
    return np.maximum(0., predicted)


def fit_model(spec, x, vintages, origin, h, seed):
    rows = training_positions(origin, h)
    y = vintages.iloc[origin, rows+h].to_numpy(float)
    anchors = x.count_latest.iloc[rows].to_numpy(float)
    valid = np.isfinite(y)
    if spec["target"] != "level":
        valid &= np.isfinite(anchors)
    rows, y, anchors = rows[valid], y[valid], anchors[valid]
    if len(rows) < 25:
        raise ValueError("Insufficient observed training labels")
    estimator = LGBMRegressor(
        objective=spec["objective"], n_estimators=100, learning_rate=.035,
        max_depth=3, num_leaves=7, min_child_samples=15, reg_lambda=10.,
        reg_alpha=.1, random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True, importance_type="gain")
    estimator.fit(x.iloc[rows][spec["features"]], transform_target(y, anchors, spec["target"]))
    return estimator
