from __future__ import annotations
import numpy as np
import pandas as pd

from .features import autoregressive_features, history_at
from .models import NEWS, EXTERNAL

def local_threshold(history):
    y = np.asarray(history, float)
    return float(max(1.5 * y[-4:].mean(), 1.5 * y[-1])) if len(y) >= 4 and np.isfinite(y[-4:]).all() else np.nan


def make_dataset(panel, news, truth, external=None, threshold_fn=local_threshold, canonical=False):
    ar = autoregressive_features(panel)
    matrices, rows = [], []
    for h in range(1, 5):
        x = ar["matrices"][h].copy()
        x["horizon"] = h
        x["seasonal_naive_52"] = x.seasonal_target_52
        x["seasonal_level_adjusted"] = np.nan
        x["seasonal_growth"] = np.nan
        x["seasonal_mean_3"] = np.nan
        x["ewm_4"] = np.nan
        x["ewm_12"] = np.nan
        x["trend_forecast"] = np.maximum(0, x.lag_1 + h * x.slope_4)
        x["rolling_mean_4"] = x.mean_4
        x["last_value"] = x.lag_1
        for i in range(len(panel)):
            history, scale = history_at(panel, i)
            threshold = threshold_fn(panel.iloc[i, :i+1])
            seasonal = i + h - 52
            if seasonal >= 0:
                prev_level = np.mean(history[max(0, i-55):i-51]) if i >= 55 else np.nan
                x.loc[panel.index[i], "seasonal_level_adjusted"] = history[seasonal] * ar["scale"][i] / max(prev_level, 1.) if np.isfinite(prev_level) else np.nan
                x.loc[panel.index[i], "seasonal_growth"] = history[seasonal] / max(history[i-52], 1.) if i >= 52 else np.nan
                x.loc[panel.index[i], "seasonal_mean_3"] = np.mean(history[max(0, seasonal-1):seasonal+2])
            for span in [4, 12]:
                x.loc[panel.index[i], f"ewm_{span}"] = pd.Series(history).ewm(span=span, adjust=False).mean().iloc[-1]
            j = i+h
            actual = float(truth.iloc[j]) if j < len(truth) else np.nan
            target = panel.index[i] + pd.Timedelta(weeks=h)
            rows.append({"origin_index": i, "horizon": h, "origin_week": panel.index[i],
                         "forecast_week": target, "issued_at_utc": panel.index[i]+pd.Timedelta(days=7, hours=12),
                         "actual": actual, "threshold": threshold, "anchor": history[-1],
                         "mase_scale": scale, "eligible": bool(np.isfinite(threshold) and np.isfinite(actual)),
                         "positive": bool(np.isfinite(threshold) and np.isfinite(actual) and actual > threshold),
                         "onset": False})
        for c in ["seasonal_naive_52", "seasonal_level_adjusted", "seasonal_mean_3", "trend_forecast", "ewm_4", "ewm_12"]:
            x[f"relative_{c}"] = x[c] / ar["scale"]
        additions = news[NEWS].copy()
        for c in NEWS:
            additions[c+"_change1"] = news[c].diff()
            additions[c+"_mean4"] = news[c].rolling(4).mean()
            if c.startswith("root_"):
                additions[c+"_share"] = news[c] / news.total_events_latest.clip(lower=1)
        x = x.join(additions)
        if external is not None:
            x = x.join(external[EXTERNAL])
        matrices.append(x.reset_index(drop=True))
    meta = pd.DataFrame(rows)
    events = catalogue(panel, truth, threshold_fn).reindex(pd.DatetimeIndex(meta.forecast_week))
    meta["onset"] = events.onset.fillna(False).to_numpy(bool)
    if canonical:
        for column in ["event_threshold", "positive", "onset", "label_valid", "episode_id"]:
            meta[column] = events[column].to_numpy()
        for column in ["positive", "onset", "label_valid"]:
            meta[column] = meta[column].fillna(False).astype(bool)
        meta["eligible"] &= meta.label_valid
    features = pd.concat(matrices, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return features, meta


def feature_columns(x, family):
    news = [c for c in x if any(c == n or c.startswith(n+"_") for n in NEWS)]
    fx = [c for c in EXTERNAL if c in x]
    seasonal = [c for c in x if "seasonal" in c]
    if family == "lags":
        return [c for c in x if c.startswith("lag_") or c in ["horizon", "forecast_week_sin", "forecast_week_cos"]]
    if family == "core":
        return [c for c in x if c not in news+fx+seasonal]
    if family == "seasonal":
        return [c for c in x if c not in news+fx]
    if family == "relative":
        return [c for c in x if c.startswith(("relative_", "log_return_", "ratio_")) or c in ["horizon", "rising_weeks", "falling_weeks", "forecast_week_sin", "forecast_week_cos"]]
    if family == "full":
        return list(x)
    raise ValueError(family)


def catalogue(panel, truth, threshold_fn):
    rows = []
    active_id = None
    previous_flag, previous_valid = False, False
    for j, week in enumerate(panel.index):
        history = panel.iloc[j-1, :j] if j else pd.Series(dtype=float)
        threshold = threshold_fn(history)
        valid = bool(np.isfinite(threshold) and np.isfinite(truth.iloc[j]))
        flag = bool(valid and truth.iloc[j] > threshold)
        onset = bool(flag and previous_valid and not previous_flag)
        if onset:
            active_id = week.strftime("%Y-%m-%d")
        elif not flag or not previous_valid:
            active_id = None
        rows.append({"week_start": week, "actual": float(truth.iloc[j]),
                     "event_threshold": threshold, "year_observations": int(history.iloc[-52:].notna().sum()),
                     "label_valid": valid, "positive": flag, "onset": onset,
                     "episode_id": active_id if flag else None,
                     "censored_start": bool(flag and not previous_valid)})
        previous_flag, previous_valid = flag, valid
    return pd.DataFrame(rows).set_index("week_start")

