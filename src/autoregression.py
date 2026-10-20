from __future__ import annotations
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from .features import autoregressive_features
from .models import NEWS, EXTERNAL, training_positions

VARIANTS = {"ar_only": (False, False), "ar_news": (True, False),
            "ar_fx": (False, True), "ar_news_fx": (True, True)}


PARAMETERS = {"n_estimators": 160, "learning_rate": .04, "max_depth": 3,
              "num_leaves": 5, "min_child_samples": 5, "reg_lambda": 1.,
              "reg_alpha": .1, "max_bin": 31, "colsample_bytree": 1., "extra_trees": False}


def build_bundle(vintages, news, external):
    for label, frame in [("news", news), ("external", external)]:
        if not frame.index.equals(vintages.index) or not frame.index.is_unique:
            raise ValueError(f"{label} issue calendar does not match target vintages")
    cutoff = vintages.index + pd.Timedelta(days=7)
    issued = cutoff + pd.Timedelta(hours=12)
    if not np.array_equal(pd.to_datetime(news.asof_export_before).to_numpy(), cutoff.to_numpy()):
        raise ValueError("Unexpected GDELT availability cutoff")
    if not np.array_equal(pd.to_datetime(news.forecast_issued_at_utc).to_numpy(), issued.to_numpy()):
        raise ValueError("Unexpected GDELT issue timestamp")
    observation = pd.to_datetime(external.fx_observation_date)
    if observation.isna().any() or not (observation.to_numpy() <= vintages.index.to_numpy()).all():
        raise ValueError("FX observations must precede issue by at least seven days")
    bundle = autoregressive_features(vintages)
    ar_columns = bundle["families"]["raw"]
    columns = {name: list(ar_columns) + (NEWS if with_news else []) + (EXTERNAL if with_fx else [])
               for name, (with_news, with_fx) in VARIANTS.items()}
    added = news[NEWS].join(external[EXTERNAL]).replace([np.inf, -np.inf], np.nan)
    bundle["matrices"] = {h: matrix.join(added) for h, matrix in bundle["matrices"].items()}
    bundle["variant_columns"] = columns
    return bundle


def train_rows(bundle, origin, horizon, labels=None):
    positions = training_positions(origin, horizon, warmup=8)
    targets = bundle["vintages"] if labels is None else labels
    actual = targets.iloc[origin, positions+horizon].to_numpy(float)
    valid = np.isfinite(actual)
    positions, actual = positions[valid], actual[valid]
    if len(positions) < 25:
        raise ValueError("Insufficient mature labels")
    # The anchor is from each example's own issue, not a later revised vintage.
    response = actual-bundle["anchor"][positions]
    return positions, actual, response


def fit_origin(bundle, origin, variant, labels=None):
    columns = bundle["variant_columns"][variant]
    fitted = {}
    for h in range(1, 5):
        positions, actual, response = train_rows(bundle, origin, h, labels)
        model = LGBMRegressor(objective="regression_l1", n_jobs=1, verbosity=-1,
            random_state=42, deterministic=True, force_col_wise=True,
            importance_type="gain", min_data_in_bin=1, **PARAMETERS)
        model.fit(bundle["matrices"][h].iloc[positions][columns].to_numpy(), response, feature_name=columns)
        fitted[h] = model
    return fitted


def forecast(bundle, origin, fitted, variant):
    columns = bundle["variant_columns"][variant]
    response = []
    for h, model in sorted(fitted.items()):
        response.append(model.booster_.predict(bundle["matrices"][h].iloc[[origin]][columns].to_numpy(), num_threads=1)[0])
    result = np.maximum(0., bundle["anchor"][origin] + np.asarray(response))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite forecast")
    return result
