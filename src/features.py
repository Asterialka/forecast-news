from __future__ import annotations
import numpy as np
import pandas as pd


def causal_fill(history: pd.Series) -> pd.Series:
    filled = history.copy()
    for i in np.flatnonzero(filled.isna().to_numpy()):
        past = filled.iloc[max(0,i-4):i].dropna()
        if past.empty:
            raise ValueError("Cannot fill initial unavailable history without past observations")
        filled.iloc[i] = past.mean()
    return filled

LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 26, 52]


WINDOWS = [3, 4, 8, 12, 26]


def history_at(vintages, origin):
    observed = vintages.iloc[origin, :origin + 1]
    return causal_fill(observed).to_numpy(float), float(observed.diff().abs().mean())


def autoregressive_features(vintages):
    records = []
    magnitude = []
    for origin in range(len(vintages)):
        observed = vintages.iloc[origin, :origin + 1]
        y = causal_fill(observed).to_numpy(float)
        r = {f"lag_{lag}": y[-lag] if len(y) >= lag else np.nan for lag in LAGS}
        for width in WINDOWS:
            tail = y[-width:]
            complete = len(tail) == width
            r[f"mean_{width}"] = float(tail.mean()) if complete else np.nan
            if width != 3:
                r[f"std_{width}"] = float(tail.std(ddof=1)) if complete else np.nan
                r[f"max_{width}"] = float(tail.max()) if complete else np.nan
                if width in [4, 12]:
                    r[f"min_{width}"] = float(tail.min()) if complete else np.nan
        for lag in [1, 2, 4, 12]:
            r[f"change_{lag}"] = y[-1] - y[-1-lag] if len(y) > lag else np.nan
            r[f"log_return_{lag}"] = float(np.log1p(y[-1]) - np.log1p(y[-1-lag])) if len(y) > lag else np.nan
        for width in [3, 4, 8]:
            x = np.arange(width, dtype=float) - (width - 1) / 2
            r[f"slope_{width}"] = float(x @ y[-width:] / (x @ x)) if len(y) >= width else np.nan
        for width in [12, 26]:
            r[f"ratio_mean4_mean{width}"] = (r["mean_4"] + 1) / (r[f"mean_{width}"] + 1)
        for name, direction in [("rising_weeks", 1), ("falling_weeks", -1)]:
            r[name] = 0
            for change in np.diff(observed.to_numpy(float))[::-1]:
                if not np.isfinite(change) or direction * change <= 0:
                    break
                r[name] += 1
        r["missing_8"] = int(observed.iloc[-8:].isna().sum())
        r["missing_26"] = int(observed.iloc[-26:].isna().sum())
        scale = max(float(y[-4:].mean()), 1.)
        magnitude.append((float(y[-1]), scale))
        records.append(r)
    raw = pd.DataFrame(records, index=vintages.index)
    numeric_level = [c for c in raw if c.startswith(("lag_", "mean_", "std_", "max_", "min_", "change_", "slope_"))]
    scales = np.array([s for _, s in magnitude])
    relative = raw[numeric_level].div(scales, axis=0).add_prefix("relative_")
    context = [c for c in raw if c not in numeric_level]
    families = {
        "lags": [f"lag_{lag}" for lag in LAGS] + ["seasonal_target_52"],
        "raw": list(raw.columns) + ["seasonal_target_52"],
        "relative": list(relative.columns) + context + ["relative_seasonal_target_52"],
        "mixed": list(raw.columns) + list(relative.columns) + ["seasonal_target_52", "relative_seasonal_target_52"],
    }
    calendar = ["forecast_week_sin", "forecast_week_cos", "forecast_month"]
    for columns in families.values():
        columns.extend(calendar)
    matrices = {}
    for h in range(1, 5):
        matrix = raw.join(relative)
        past = np.array([vintages.iloc[t, t+h-52] if t+h >= 52 else np.nan for t in range(len(vintages))], float)
        matrix["seasonal_target_52"] = past
        matrix["relative_seasonal_target_52"] = past / scales
        target_week = vintages.index + pd.Timedelta(weeks=h)
        iso = target_week.isocalendar().week.to_numpy(float)
        matrix["forecast_week_sin"] = np.sin(2*np.pi*iso/52.1775)
        matrix["forecast_week_cos"] = np.cos(2*np.pi*iso/52.1775)
        matrix["forecast_month"] = target_week.month
        matrices[h] = matrix.replace([np.inf, -np.inf], np.nan)
    return {"matrices": matrices, "families": families,
            "anchor": np.array([a for a, _ in magnitude]), "scale": scales, "vintages": vintages}

def asof_features(ecb: pd.DataFrame, origin_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    for week in origin_weeks:
        cutoff = week
        known = ecb.loc[ecb.index <= cutoff, "ILS_per_USD"]
        if len(known) < 60:
            raise ValueError(f"Insufficient historical ECB rates for {week}")
        returns = np.log(known).diff()
        records.append({"week_start": week, "fx_observation_date": known.index[-1],
                        "fx_ils_per_usd_lagged": known.iloc[-1],
                        "fx_return_5d_lagged": known.iloc[-1] / known.iloc[-6] - 1,
                        "fx_return_20d_lagged": known.iloc[-1] / known.iloc[-21] - 1,
                        "fx_volatility_20d_lagged": returns.iloc[-20:].std(),
                        "fx_staleness_days": (week + pd.Timedelta(days=7) - known.index[-1]).days})
    return pd.DataFrame(records).set_index("week_start")
