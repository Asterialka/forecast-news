from __future__ import annotations
import numpy as np
import pandas as pd

import warnings
import holidays
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .features import causal_fill

BASELINE = "rolling_mean_4"


NAIVE_MODELS = [BASELINE, "last_value", "seasonal_naive_52"]


ETS_MODELS = ["ets_level", "ets_damped"]


ML_MODELS = ["lgbm_poisson_core", "lgbm_poisson_no_external", "lgbm_poisson_full", "lgbm_l2_full"]


MODELS = NAIVE_MODELS + ETS_MODELS + ML_MODELS


CORE = ["count_latest", "count_lag1", "count_lag2", "count_lag4", "count_lag12", "count_lag52",
        "count_mean_4w", "count_std_12w", "count_max_8w", "count_sum_12w",
        "count_mean_26w", "weeks_since_major_spike", "spike_ever_observed", "missing_history_12w"]


NEWS = ["tone_latest", "goldstein_latest", "goldstein_std_latest", "total_events_latest",
        "domains_latest", "conflict_share_latest", "news_volume_change_wow",
        "root_03_latest", "root_04_latest", "root_09_latest", "root_13_latest",
        "root_14_latest", "root_17_latest", "root_18_latest", "root_19_latest"]


CALENDAR = ["forecast_week_sin", "forecast_week_cos", "forecast_month", "is_holiday", "holiday_days"]


EXTERNAL = ["fx_ils_per_usd_lagged", "fx_return_5d_lagged", "fx_return_20d_lagged", "fx_volatility_20d_lagged"]


def feature_matrices(base: pd.DataFrame, external: pd.DataFrame, horizons=(1, 2, 3, 4)) -> dict[int, pd.DataFrame]:
    joined = base.join(external)
    calendar = holidays.country_holidays("IL", years=range(base.index[0].year, base.index[-1].year+2))
    matrices = {}
    for h in horizons:
        x = joined.copy()
        target_week = x.index + pd.Timedelta(weeks=h)
        iso = target_week.isocalendar().week.to_numpy(float)
        x["forecast_week_sin"] = np.sin(2*np.pi*iso/52.1775)
        x["forecast_week_cos"] = np.cos(2*np.pi*iso/52.1775)
        x["forecast_month"] = target_week.month
        x["holiday_days"] = [sum((day + pd.Timedelta(days=j)).date() in calendar for j in range(7)) for day in target_week]
        x["is_holiday"] = x.holiday_days.gt(0).astype(int)
        matrices[h] = x[CORE + NEWS + CALENDAR + EXTERNAL].replace([np.inf, -np.inf], np.nan)
    return matrices


def columns(model: str) -> list[str]:
    if model.endswith("_core"):
        return CORE
    if model.endswith("_no_external"):
        return CORE + NEWS + CALENDAR
    return CORE + NEWS + CALENDAR + EXTERNAL


def training_positions(origin: int, horizon: int, warmup: int = 52) -> np.ndarray:
    result = np.arange(warmup, origin-horizon)
    if len(result) < 30:
        raise ValueError("Insufficient training history after feature warmup and label embargo")
    assert (result + horizon <= origin-1).all()
    return result


def fit_ml(model: str, x: pd.DataFrame, vintages: pd.DataFrame, origin: int, horizon: int, seed: int):
    rows = training_positions(origin, horizon)
    y = vintages.iloc[origin, rows+horizon].to_numpy(float)
    observed = np.isfinite(y)
    rows, y = rows[observed], y[observed]
    estimator = LGBMRegressor(
        objective="regression" if "_l2_" in model else "poisson", n_estimators=100,
        learning_rate=.035, max_depth=3, num_leaves=7, min_child_samples=15,
        reg_lambda=10., reg_alpha=.1, random_state=seed, n_jobs=2, verbosity=-1,
        deterministic=True, force_col_wise=True, importance_type="gain")
    estimator.fit(x.iloc[rows][columns(model)], y)
    return estimator


def fit_ets(history: np.ndarray, model: str) -> dict:
    scale = max(float(np.median(history)), 1.)
    trend = "add" if model == "ets_damped" else None
    with warnings.catch_warnings(record=True) as notices:
        fitted = ExponentialSmoothing(history/scale, trend=trend, damped_trend=bool(trend),
                                      initialization_method="estimated").fit(optimized=True)
    p = fitted.params
    return {"model": model, "scale": scale, "smoothing_level": float(p["smoothing_level"]),
            "smoothing_trend": float(p["smoothing_trend"]) if trend else 0.,
            "damping_trend": float(p["damping_trend"]) if trend else 1.,
            "initial_level": float(p["initial_level"]),
            "initial_trend": float(p["initial_trend"]) if trend else 0.,
            "optimizer_success": bool(fitted.mle_retvals.get("success", True)),
            "warnings": [str(w.message) for w in notices]}


def ets_predict(history: np.ndarray, parameters: dict, max_horizon=4) -> np.ndarray:
    """Replay observed counts with frozen smoothing parameters; no test fitting."""
    alpha, beta, phi = (parameters[k] for k in ["smoothing_level", "smoothing_trend", "damping_trend"])
    level, trend = parameters["initial_level"], parameters["initial_trend"]
    for value in history/parameters["scale"]:
        previous = level
        level = alpha*value + (1-alpha)*(level+phi*trend)
        trend = beta*(level-previous) + (1-beta)*phi*trend
    return np.maximum(0., [parameters["scale"]*(level + sum(phi**j for j in range(1,h+1))*trend)
                           for h in range(1, max_horizon+1)])


def predict_origin(matrices: dict, vintages: pd.DataFrame, origin: int, seed: int, frozen: dict | None = None):
    observed_history = vintages.iloc[origin, :origin+1]
    history = causal_fill(observed_history).to_numpy(float)
    fitted = frozen if frozen is not None else {"ml": {}, "ets": {}}
    if frozen is None:
        for model in ETS_MODELS:
            fitted["ets"][model] = fit_ets(history, model)
        for model in ML_MODELS:
            fitted["ml"][model] = {h: fit_ml(model, x, vintages, origin, h, seed) for h,x in matrices.items()}
    ets_forecasts = {m: ets_predict(history, p) for m,p in fitted["ets"].items()}
    scale = observed_history.diff().abs().mean()
    rows = []
    for h, x in matrices.items():
        predictions = {BASELINE: history[-4:].mean(), "last_value": history[-1],
                       "seasonal_naive_52": history[origin+h-52]}
        predictions.update({m: forecasts[h-1] for m,forecasts in ets_forecasts.items()})
        predictions.update({m: max(0., float(fitted["ml"][m][h].predict(x.iloc[[origin]][columns(m)])[0])) for m in ML_MODELS})
        for model, predicted in predictions.items():
            if not np.isfinite(predicted):
                raise ValueError(f"Non-finite forecast: {model}, origin={origin}, h={h}")
            rows.append({"origin_index": origin, "origin_week": vintages.index[origin],
                         "forecast_issued_at_utc": vintages.index[origin] + pd.Timedelta(days=7, hours=12),
                         "forecast_week": vintages.index[origin] + pd.Timedelta(weeks=h),
                         "horizon": h, "model": model, "predicted": predicted, "mase_scale": scale})
    return rows, fitted


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    p = predictions.copy()
    error = p.actual-p.predicted
    p["abs_error"], p["sq_error"] = abs(error), error**2
    denominator = abs(p.actual)+abs(p.predicted)
    p["sMAPE"] = np.divide(200*abs(error), denominator, out=np.zeros(len(p)), where=denominator.to_numpy()!=0)
    p["MASE"] = p.abs_error/p.mase_scale.replace(0, np.nan)
    return p


def metrics(predictions: pd.DataFrame, by_horizon: bool = False) -> pd.DataFrame:
    p = score(predictions.dropna(subset=["actual", "predicted"]))
    group = ["model", "horizon"] if by_horizon else ["model"]
    result = p.groupby(group).agg(n=("actual", "size"), MASE=("MASE", "mean"),
                                sMAPE=("sMAPE", "mean"), RMSE=("sq_error", lambda x: np.sqrt(x.mean())),
                                MAE=("abs_error", "mean")).reset_index()
    if by_horizon:
        baseline = result[result.model.eq(BASELINE)].set_index("horizon").RMSE
        result["RMSE_vs_baseline_pct"] = 100*(1-result.RMSE/result.horizon.map(baseline))
    elif result.model.eq(BASELINE).any():
        baseline = result.loc[result.model.eq(BASELINE), "RMSE"].iloc[0]
        result["RMSE_vs_baseline_pct"] = 100*(1-result.RMSE/baseline)
    return result.sort_values(["horizon", "RMSE"] if by_horizon else ["RMSE"])


def calibrate(predictions: pd.DataFrame, alpha=.1) -> dict:
    radii = {}
    for (model,h), group in predictions.groupby(["model", "horizon"]):
        errors = np.sort(abs(group.actual-group.predicted).to_numpy())
        rank = min(len(errors), int(np.ceil((len(errors)+1)*(1-alpha))))
        radii.setdefault(model, {})[int(h)] = float(errors[rank-1])
    return radii


def add_intervals(predictions: pd.DataFrame, radii: dict) -> pd.DataFrame:
    p = predictions.copy()
    radius = np.array([radii[m][h] for m,h in zip(p.model,p.horizon)])
    p["ci_lower"] = np.maximum(0., p.predicted-radius)
    p["ci_upper"] = p.predicted+radius
    return p


def interval_metrics(predictions: pd.DataFrame, alpha=.1) -> pd.DataFrame:
    p = predictions.copy()
    p["covered"] = (p.actual >= p.ci_lower) & (p.actual <= p.ci_upper)
    p["width"] = p.ci_upper-p.ci_lower
    p["interval_score"] = p.width + 2/alpha*((p.ci_lower-p.actual).clip(lower=0)+(p.actual-p.ci_upper).clip(lower=0))
    return p.groupby(["model","horizon"]).agg(n=("actual","size"), coverage=("covered","mean"),
        mean_width=("width","mean"), interval_score=("interval_score","mean")).reset_index()
