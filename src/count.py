"""2021–2024: rolling validation and a frozen final 12-week holdout."""
import numpy as np
import pandas as pd
from . import models, autoregression, feature_selection, correction, transformed
from .arima import fit_arima, forecast_arima
from .common import ROOT
from .data import core_features
from .features import asof_features, history_at

TRANSFORMS = {
    "delta_l2": {"target": "delta", "objective": "regression", "features": models.CORE},
    "log_delta_l1": {"target": "log_delta", "objective": "regression_l1", "features": models.CORE},
}


def prepare(panel, news, ecb):
    panel = panel.iloc[:208, :208].copy()
    news = news.iloc[:208].copy()
    fx = asof_features(ecb, panel.index)
    base = core_features(panel, news)
    return panel, base, models.feature_matrices(base, fx), autoregression.build_bundle(panel, news, fx)


def fit_and_predict(origin, panel, base, matrices, ar_bundle, correction_bundle, ar_history, cfg, frozen=None):
    rows, standard = models.predict_origin(matrices, panel, origin, cfg["seed"],
                                          None if frozen is None else frozen["standard"])
    history, _ = history_at(panel, origin)
    values = {}
    if frozen is None:
        frozen = {"standard": standard, "arima": {
            "arima_1_0_0": ar_history[origin],
            "arima_1_1_1": fit_arima(history, (1, 1, 1))[0]},
            "ar": {name: autoregression.fit_origin(ar_bundle, origin, name)
                   for name in autoregression.VARIANTS},
            "transforms": {(name, h): transformed.fit_model(spec, base, panel, origin, h, cfg["seed"])
                           for name, spec in TRANSFORMS.items() for h in range(1, 5)}}
        fitted, definitions = feature_selection.fit_origin(matrices, panel, origin, cfg["seed"], cfg["feature_selection"])
        frozen["fi"] = fitted, definitions
        fitted, columns = correction.fit_origin(correction_bundle, origin, cfg["correction"])
        frozen["correction"] = fitted, columns
    for name, params in frozen["arima"].items():
        values[name] = forecast_arima(history, params)
    for name, fitted in frozen["ar"].items():
        values[name] = autoregression.forecast(ar_bundle, origin, fitted, name)
    for name, spec in TRANSFORMS.items():
        raw = [frozen["transforms"][name, h].predict(base.iloc[[origin]][spec["features"]])[0] for h in range(1, 5)]
        values[name] = transformed.restore_target(np.asarray(raw), history[-1], spec["target"])
    fi = feature_selection.predict(matrices, origin, *frozen["fi"])
    values.update(fi)
    errors = correction.predicted_errors(correction_bundle, origin, *frozen["correction"])
    values["ar_correction_06_lambda050"] = correction.combine(values["arima_1_0_0"], errors, .5)
    core = np.array([r["predicted"] for r in rows if r["model"] == "lgbm_poisson_core"])
    values["blend_global"] = .75*history[-1]+.25*core
    template = {r["horizon"]: r for r in rows if r["model"] == "last_value"}
    rows.extend({**template[h], "model": name, "predicted": float(p[h-1])}
                for name, p in values.items() for h in range(1, 5))
    return rows, frozen


def run(panel, news, ecb, cfg):
    panel, base, matrices, ar_bundle = prepare(panel, news, ecb)
    # Historical residual targets require ARIMA forecasts made at their own origins.
    development = panel.iloc[:196, :196]
    ar_pred, ar_params = correction.historical_arima(development)
    correction_dev = correction.correction_bundle(development, ar_pred)
    rows = []
    origins = list(range(103, 162))+list(range(167, 191))
    for count, origin in enumerate(origins, 1):
        predicted, _ = fit_and_predict(origin, panel, base, matrices, ar_bundle, correction_dev, ar_params, cfg)
        block, snapshot = ("selection", 166) if origin <= 161 else ("calibration", 195)
        for row in predicted:
            row.update(block=block, actual=float(panel.iloc[snapshot, origin+row["horizon"]]))
        rows.extend(predicted)
        if count % 10 == 0 or count == len(origins):
            print(f"Count CV: {count}/{len(origins)}", flush=True)
    cv = pd.DataFrame(rows)
    selection = models.metrics(cv[cv.block.eq("selection") & cv.model.isin(models.MODELS)])
    selected = cfg["count"]["forecast_model"]
    if selected not in cv.model.unique():
        raise ValueError(f"Unknown count forecast model: {selected}")
    interval_models = list(dict.fromkeys([*models.MODELS, selected]))
    radii = models.calibrate(cv[cv.block.eq("calibration") & cv.model.isin(interval_models)])
    _, frozen = fit_and_predict(195, panel, base, matrices, ar_bundle, correction_dev, ar_params, cfg)
    correction_all = correction.test_bundle(panel, ar_pred, frozen["arima"]["arima_1_0_0"])
    truth = pd.read_csv(ROOT/"data/preprocessed/truth_2024.csv", index_col=0, parse_dates=True).iloc[:, 0]
    holdout = []
    for origin in range(195, 207):
        predicted, _ = fit_and_predict(origin, panel, base, matrices, ar_bundle, correction_all, ar_params, cfg, frozen)
        for row in predicted:
            target = origin+row["horizon"]
            if target < 208:
                row.update(block="holdout", actual=float(truth.iloc[target]))
                holdout.append(row)
    holdout = pd.DataFrame(holdout)
    with_intervals = holdout.model.isin(interval_models)
    calibrated = models.add_intervals(holdout[with_intervals], radii)
    holdout = pd.concat([calibrated, holdout[~with_intervals]], ignore_index=True)
    future, _ = fit_and_predict(207, panel, base, matrices, ar_bundle, correction_all, ar_params, cfg, frozen)
    future = pd.DataFrame(future)
    forecast = models.add_intervals(future[future.model.eq(selected)], radii)
    forecast["country"], forecast["target_type"] = "Israel", "count"
    forecast["predicted_value"], forecast["predicted_prob"] = forecast.predicted, np.nan
    forecast["forecast_week"] = forecast.forecast_week.dt.strftime("%G-W%V")
    forecast = forecast[["country", "forecast_week", "horizon", "target_type", "predicted_value",
                         "predicted_prob", "ci_lower", "ci_upper", "model", "forecast_issued_at_utc"]]
    predictions = pd.concat([cv, holdout], ignore_index=True)
    result = {"selected_model": selected, "selection_scope": [selected],
              "selection_method": "Fixed explicitly in config; retrospective user choice",
              "original_cv_selected_model": str(selection.iloc[0].model),
              "original_cv_selection_scope": models.MODELS,
              "calibration_radii": radii, "interval_level": .9,
              "interval_metrics": models.interval_metrics(calibrated).to_dict("records"),
              "holdout_cases_per_model": {str(h): int(n) for h, n in holdout[holdout.model.eq(selected)].groupby("horizon").size().items()},
              "research_variants_are_retrospective": True}
    return summarize(predictions), forecast, result, predictions


def summarize(predictions):
    scores = []
    for block, group in predictions.groupby('block'):
        scores.extend([models.metrics(group).assign(block=block, horizon=0),
                       models.metrics(group, True).assign(block=block)])
    return pd.concat(scores, ignore_index=True).assign(benchmark='count_2024')
