from __future__ import annotations
import numpy as np

from lightgbm import LGBMRegressor
from .features import autoregressive_features, history_at
from .arima import fit_arima, forecast_arima
from .models import training_positions

MIN_AR_ORIGIN = 26


def historical_arima(vintages):
    predicted = np.full((len(vintages), 4), np.nan)
    parameters = {}
    for origin in range(MIN_AR_ORIGIN, len(vintages)):
        history, _ = history_at(vintages, origin)
        fitted, _ = fit_arima(history, (1, 0, 0))
        predicted[origin] = forecast_arima(history, fitted)
        parameters[origin] = fitted
        if (origin-MIN_AR_ORIGIN+1) % 40 == 0:
            print(f"Historical ARIMA forecasts: {origin-MIN_AR_ORIGIN+1}/{len(vintages)-MIN_AR_ORIGIN}", flush=True)
    return predicted, parameters


def correction_bundle(vintages, predicted):
    bundle = autoregressive_features(vintages)
    if predicted.shape != (len(vintages), 4):
        raise ValueError("ARIMA forecasts do not match the issue calendar")
    bundle["base"] = predicted
    return bundle


def train_rows(bundle, origin, horizon):
    positions = training_positions(origin, horizon, warmup=MIN_AR_ORIGIN)
    actual = bundle["vintages"].iloc[origin, positions+horizon].to_numpy(float)
    base = bundle["base"][positions, horizon-1]
    valid = np.isfinite(actual) & np.isfinite(base)
    positions, actual, base = positions[valid], actual[valid], base[valid]
    if len(positions) < 25:
        raise ValueError("Insufficient matured out-of-sample ARIMA errors")
    return positions, actual-base


def fit_origin(bundle, origin, spec):
    columns = list(bundle["families"]["raw"])
    models = {}
    for h in range(1, 5):
        positions, errors = train_rows(bundle, origin, h)
        model = LGBMRegressor(objective=spec["objective"], n_jobs=1, verbosity=-1,
            random_state=42, deterministic=True, force_col_wise=True,
            min_data_in_bin=1, importance_type="gain", **spec["parameters"])
        model.fit(bundle["matrices"][h].iloc[positions][columns].to_numpy(), errors, feature_name=columns)
        models[h] = model
    return models, columns


def predicted_errors(bundle, origin, models, columns):
    values = []
    for h, model in sorted(models.items()):
        values.append(float(model.booster_.predict(bundle["matrices"][h].iloc[[origin]][columns].to_numpy(), num_threads=1)[0]))
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predicted ARIMA errors")
    return values


def combine(base, errors, strength):
    return np.maximum(0., np.asarray(base) + strength*np.asarray(errors))


def test_bundle(vintages, development_base, frozen):
    base = np.full((len(vintages), 4), np.nan)
    base[:len(development_base)] = development_base
    for origin in range(195, len(vintages)):
        history, _ = history_at(vintages, origin)
        base[origin] = forecast_arima(history, frozen)
    return correction_bundle(vintages, base)
