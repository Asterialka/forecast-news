from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"forecast-task-matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from .arima import fit_arima, forecast_arima
from .common import ROOT, sha256
from .data import load
from .features import history_at

TEAL, BLUE, GRAY, INK = "#087f8c", "#6082b6", "#89939f", "#263442"
ARIMA = "arima_1_0_0"
BASELINE = "rolling_mean_4"


def holdout_predictions(panel, truth, metrics, details):
    """Refit only the fixed ARIMA before holdout, then update its filtering state."""
    parameters, _ = fit_arima(history_at(panel, 195)[0], (1, 0, 0))
    radii = details["count"]["calibration_radii"][ARIMA]
    rows = []
    for origin in range(195, 207):
        history, _ = history_at(panel, origin)
        predicted = forecast_arima(history, parameters)
        for horizon in range(1, 5):
            target = origin+horizon
            if target >= 208:
                continue
            radius = radii[str(horizon)]
            rows.append({
                "origin_week": panel.index[origin], "forecast_week": panel.index[target],
                "horizon": horizon, "actual": truth.iloc[target],
                ARIMA: predicted[horizon-1], BASELINE: history[-4:].mean(),
                "ci_lower": max(0., predicted[horizon-1]-radius),
                "ci_upper": predicted[horizon-1]+radius,
            })
    predictions = pd.DataFrame(rows)
    for model in (ARIMA, BASELINE):
        for horizon in range(5):
            frame = predictions if horizon == 0 else predictions[predictions.horizon.eq(horizon)]
            saved = metrics[
                metrics.benchmark.eq("count_2024") & metrics.block.eq("holdout")
                & metrics.model.eq(model) & metrics.horizon.eq(horizon)
            ]
            if len(saved) != 1 or len(frame) != saved.iloc[0]["n"]:
                raise ValueError(f"Wrong metric coverage for {model}, h{horizon}")
            error = frame[model]-frame.actual
            np.testing.assert_allclose(
                [np.sqrt(np.mean(error**2)), np.mean(abs(error))],
                saved.iloc[0][["RMSE", "MAE"]].to_numpy(float), rtol=1e-9, atol=1e-7,
            )
    return predictions, parameters


def decorate(ax, *, dates=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e8ed", linewidth=.7)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if dates:
        span = ax.get_xlim()[1]-ax.get_xlim()[0]
        locator = (mdates.YearLocator() if span > 730 else mdates.MonthLocator() if span > 120
                   else mdates.WeekdayLocator(byweekday=mdates.MO, interval=3))
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y" if span > 730 else "%d.%m"))


def save(fig, output, name, note):
    fig.text(.02, .015, note, fontsize=9, color="#586674", va="bottom")
    fig.tight_layout(rect=(0, .065, 1, .96))
    for suffix in ("png", "svg"):
        fig.savefig(output/f"{name}.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


def history_plot(truth, output):
    fig, ax = plt.subplots(figsize=(12, 4.9))
    ax.axvspan(truth.index[0], pd.Timestamp("2022-12-26"), color="#f0f2f5", label="Начальная история")
    ax.axvspan(pd.Timestamp("2022-12-26"), pd.Timestamp("2024-02-12"), color="#e5effa", label="Ранняя CV: срезы истории")
    ax.axvspan(pd.Timestamp("2024-03-18"), pd.Timestamp("2024-09-02"), color="#e0f2ec", label="Поздняя CV: срезы истории")
    ax.axvspan(pd.Timestamp("2024-10-07"), pd.Timestamp("2024-12-30"), color="#fae8d9", label="Holdout: целевые недели")
    ax.plot(truth.index, truth, color=INK, linewidth=1.25)
    ax.axvline(pd.Timestamp("2024-10-07"), color="#c37e43", linestyle="--", linewidth=1)
    ax.set(title="Израиль · недельная конфликтная активность, 2021–2024",
           ylabel="Уникальные записи GDELT: корни 18 и 19", xlim=(pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-30")))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    decorate(ax, dates=True)
    save(fig, output, "01_history",
         "Факт: архивы по 05.01.2025. Пропуски оставлены разрывами. Holdout: 07.10–29.12.2024.\n"
         "CV отмечена по последней завершённой неделе на выпуске; holdout — по целевой неделе.")


def metrics_plot(metrics, output):
    table = metrics[
        metrics.benchmark.eq("count_2024") & metrics.block.eq("holdout") & metrics.horizon.eq(0)
    ].sort_values("RMSE", ascending=True)
    baseline = float(table.loc[table.model.eq(BASELINE), "RMSE"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [TEAL if model == ARIMA else GRAY if model == BASELINE else BLUE for model in table.model]
    bars = ax.barh(table.model, table.RMSE, color=colors, height=.72)
    ax.invert_yaxis()
    ax.axvline(baseline, color="#596574", linestyle="--", linewidth=1)
    for bar, value in zip(bars, table.RMSE):
        change = 100*(1-value/baseline)
        ax.text(value+65, bar.get_y()+bar.get_height()/2,
                f"{value:,.0f}  ({change:+.1f}%)", va="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
    ax.set(xlim=(0, table.RMSE.max()*1.24), xlabel="RMSE ↓", title="Все модели количества · holdout 2024")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", color="#e3e8ed", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    save(fig, output, "02_model_comparison",
         "42 прогноза на модель, объединены h1–h4. Проценты: снижение RMSE относительно rolling_mean_4.\n"
         "Бирюзовый — ARIMA для forecast.csv. Ретроспективное сравнение; модели отсортированы по RMSE.")


def holdout_plot(predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for horizon, ax in enumerate(axes.flat, 1):
        frame = predictions[predictions.horizon.eq(horizon)].sort_values("forecast_week")
        ax.fill_between(frame.forecast_week, frame.ci_lower, frame.ci_upper,
                        color=TEAL, alpha=.12, label="90%-й интервал ARIMA")
        ax.plot(frame.forecast_week, frame.actual, color=INK, linewidth=1.8, marker="o", markersize=3, label="Факт")
        ax.plot(frame.forecast_week, frame[ARIMA], color=TEAL, linewidth=1.8, marker="o", markersize=3, label="ARIMA(1,0,0)")
        ax.plot(frame.forecast_week, frame[BASELINE], color=GRAY, linewidth=1.6, linestyle="--", label="Rolling mean 4")
        ax.set(title=f"h{horizon} · {len(frame)} прогнозов", ylim=(0, None), ylabel="Записи GDELT")
        decorate(ax, dates=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .95))
    fig.suptitle("ARIMA против baseline · последние 12 недель 2024", y=.995)
    save(fig, output, "03_holdout",
         "Коэффициенты ARIMA заморожены до holdout; состояние обновляется доступной историей. Интервалы — по calibration.\n"
         "Каждая точка — отдельный исторический выпуск. h1 выпускается через 12 часов после начала целевой недели.")


def forecast_plot(truth, forecast, output):
    frame = forecast.sort_values("horizon").copy()
    frame["week"] = pd.to_datetime([
        date.fromisocalendar(int(value[:4]), int(value[-2:]), 1) for value in frame.forecast_week
    ])
    fig, ax = plt.subplots(figsize=(12, 4.9))
    tail = truth.iloc[-12:]
    ax.plot(tail.index, tail, color=INK, linewidth=1.8, marker="o", markersize=4, label="Исторический факт")
    ax.fill_between(frame.week, frame.ci_lower, frame.ci_upper, color=TEAL, alpha=.14, label="90%-й интервал")
    ax.plot(frame.week, frame.predicted_value, color=TEAL, linewidth=2, marker="o", label="ARIMA(1,0,0)")
    for row in frame.itertuples():
        ax.annotate(f"{row.predicted_value:,.0f}", (row.week, row.predicted_value),
                    xytext=(0, 10), textcoords="offset points", ha="center", color=TEAL, fontsize=10)
    ax.axvline(frame.week.min(), color=GRAY, linewidth=1, linestyle="--")
    ax.set(title="Исторический прогноз · выпуск 30.12.2024", ylabel="Записи GDELT", ylim=(0, None))
    ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=3)
    decorate(ax, dates=True)
    save(fig, output, "04_forecast",
         "Прогноз на 2025-W01…W04 из forecast.csv; не актуальный прогноз на сегодня.\n"
         "ARIMA выбрана ретроспективно. Слева — факт из оценочного снимка; модель использовала доступные на выпуске vintages.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "bold", "axes.labelcolor": INK,
                         "text.color": INK, "svg.fonttype": "none"})
    panel, _, _, _ = load()
    panel = panel.iloc[:208, :208]
    truth = pd.read_csv(ROOT/"data/preprocessed/truth_2024.csv", index_col=0, parse_dates=True).iloc[:, 0]
    metrics = pd.read_csv(ROOT/"outputs/metrics.csv")
    details = json.loads((ROOT/"outputs/metrics.json").read_text())
    forecast = pd.read_csv(ROOT/"outputs/forecast.csv")
    predictions, parameters = holdout_predictions(panel, truth, metrics, details)
    if forecast.model.tolist() != [ARIMA]*4 or forecast.horizon.tolist() != [1, 2, 3, 4]:
        raise ValueError("Expected the saved four-horizon ARIMA forecast")
    np.testing.assert_allclose(forecast.predicted_value, forecast_arima(history_at(panel, 207)[0], parameters), rtol=1e-9)
    predictions.to_csv(args.output/"holdout_predictions.csv", index=False)
    history_plot(truth, args.output)
    metrics_plot(metrics, args.output)
    holdout_plot(predictions, args.output)
    forecast_plot(truth, forecast, args.output)
    inputs = ["data/preprocessed/manifest.json", "outputs/metrics.csv",
              "outputs/metrics.json", "outputs/forecast.csv", "src/plot_results.py",
              "src/arima.py", "src/features.py"]
    manifest = {"inputs_sha256": {name: sha256(ROOT/name) for name in inputs},
                "matplotlib": matplotlib.__version__,
                "verification": "ARIMA and baseline holdout RMSE/MAE matched for pooled and h1-h4; final ARIMA forecast matched.",
                "files_sha256": {p.name: sha256(p) for p in sorted(args.output.iterdir())
                                  if p.suffix in {".png", ".svg", ".csv"}}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(f"Verified and saved four figures (PNG/SVG): {args.output}")


if __name__ == "__main__":
    main()
