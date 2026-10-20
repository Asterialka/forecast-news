"""Fixed input snapshots; no fitted models or previous experiment outputs."""
import json
import numpy as np
import pandas as pd
from .common import ROOT, sha256
from .models import CORE


def load():
    directory = ROOT / "data" / "preprocessed"
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Input checksum mismatch: {name}")
    panel = pd.read_parquet(directory / "count_vintages.parquet")
    news = pd.read_csv(directory / "news_asof.csv", index_col=0, parse_dates=True)
    for column in ["forecast_issued_at_utc", "asof_export_before"]:
        news[column] = pd.to_datetime(news[column], format="ISO8601")
    truth = pd.read_csv(directory / "truth_7d.csv", index_col=0, parse_dates=True).iloc[:, 0]
    ecb = pd.read_csv(directory / "ecb_ils.csv", index_col=0, parse_dates=True)
    assert panel.index.equals(news.index) and panel.index.equals(truth.index)
    return panel, news, truth, ecb


def core_features(panel, news):
    records = []
    for i in range(len(panel)):
        history = panel.iloc[i, :i+1]
        row = {"count_latest": history.iloc[-1]}
        for lag in [1, 2, 4, 12, 52]:
            row[f"count_lag{lag}"] = history.iloc[-1-lag] if len(history) > lag else np.nan
        for stat, width in [("mean", 4), ("std", 12), ("max", 8), ("sum", 12), ("mean", 26)]:
            values = history.iloc[-width:]
            row[f"count_{stat}_{width}w"] = float(getattr(values, stat)()) if len(values) == width else np.nan
        past = history.shift(1).rolling(52, min_periods=52)
        peaks = np.flatnonzero((history > past.mean()+2*past.std()).to_numpy())
        row["weeks_since_major_spike"] = len(history)-1-peaks[-1] if len(peaks) else np.nan
        row["spike_ever_observed"] = int(bool(len(peaks)))
        if pd.isna(history.iloc[-1]):
            row = dict.fromkeys(row, np.nan)
        row["missing_history_12w"] = int(history.iloc[-12:].isna().sum())
        records.append(row)
    return pd.DataFrame(records, index=panel.index)[CORE].join(news)


def config():
    return json.loads((ROOT / "config.json").read_text())
