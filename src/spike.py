import pandas as pd
from . import event_models, autoregression, models
from .arima import fit_arima, forecast_arima
from .features import asof_features, history_at
from .event_features import make_dataset
from .event_metrics import metric_row

FREEZE = 207
FOLDS = [(78, 103), (104, 129), (130, 155), (156, 181), (182, 204)]

def validation_mask(meta, start, end):
    return meta.eligible & meta.origin_index.between(start, end) & (meta.origin_index+meta.horizon <= 206)


def test_mask(meta):
    return meta.eligible & meta.origin_index.ge(FREEZE) & meta.forecast_week.between("2025-01-06", "2025-12-29")


def validation(meta):
    mask = pd.Series(False, index=meta.index)
    for start, end in FOLDS:
        mask |= validation_mask(meta, start, end)
    return mask


def baselines(panel, x, meta):
    pieces = []
    for name in models.NAIVE_MODELS:
        for block, mask in [("validation", validation(meta)), ("test", test_mask(meta))]:
            p = meta.loc[mask].copy()
            p["predicted"] = x.loc[mask, name]
            if name == "seasonal_naive_52":
                p["predicted"] = [history_at(panel, i)[0][i+h-52] for i, h in zip(p.origin_index, p.horizon)]
            p["score"] = p.predicted/p.threshold
            pieces.append(p.assign(model=name, block=block, cutoff=1., probability=False))
    return pd.concat(pieces, ignore_index=True)


def original_models(panel, news, ecb, meta):
    bundle = autoregression.build_bundle(panel, news, asof_features(ecb, panel.index))
    history, _ = history_at(panel, FREEZE)
    ar = {name: fit_arima(history, order)[0] for name, order in
          [("arima_1_0_0", (1, 0, 0)), ("arima_1_1_1", (1, 1, 1))]}
    ets = {name: models.fit_ets(history, name) for name in models.ETS_MODELS}
    fitted = {name: autoregression.fit_origin(bundle, FREEZE, name) for name in autoregression.VARIANTS}
    names = {"ar_only": "lgbm_ar", "ar_news": "lgbm_news", "ar_fx": "lgbm_fx", "ar_news_fx": "lgbm_news_fx"}
    test = meta.loc[test_mask(meta)]
    rows = []
    for origin, group in test.groupby("origin_index"):
        history, _ = history_at(panel, origin)
        predictions = {name: forecast_arima(history, params) for name, params in ar.items()}
        predictions.update({name: models.ets_predict(history, params) for name, params in ets.items()})
        predictions.update({names[name]: autoregression.forecast(bundle, origin, estimator, name)
                            for name, estimator in fitted.items()})
        for name, values in predictions.items():
            p = group.copy()
            p["predicted"] = values[p.horizon.to_numpy()-1]
            p["score"] = p.predicted/p.threshold
            rows.append(p.assign(model=name, block="test", cutoff=1., probability=False))
    return pd.concat(rows, ignore_index=True)


def run(panel, news, truth, ecb, cfg):
    x, meta = make_dataset(panel, news, truth, asof_features(ecb, panel.index))
    pieces = [baselines(panel, x, meta), original_models(panel, news, ecb, meta)]
    for spec in cfg['event_models']:
        for start, end in FOLDS+[(FREEZE, None)]:
            fitted = event_models.fit(spec, x, meta, start)
            mask = test_mask(meta) if end is None else validation_mask(meta, start, end)
            p = event_models.predict(spec, fitted, x, meta, mask)
            pieces.append(p.assign(model=spec['id'], cutoff=spec['spike_cutoff'],
                block='test' if end is None else 'validation', probability=event_models.is_probability(spec)))
        print(f"Spike: {spec['id']}", flush=True)
    predictions = pd.concat(pieces, ignore_index=True)
    return summarize(predictions), predictions


def summarize(predictions):
    rows = []
    for (model, block), group in predictions.groupby(["model", "block"]):
        for h in [0, 1, 2, 3, 4]:
            p = group[group.horizon.eq(h)] if h else group[group.horizon.ge(2)]
            rows.append({'benchmark': 'spike_2025', 'model': model, 'block': block,
                         'horizon': h, **metric_row(p, empty_recall=0.)})
    return pd.DataFrame(rows)
