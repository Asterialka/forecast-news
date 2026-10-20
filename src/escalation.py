"""Annual mean + 2 sample std: expanding fits and advance episode detection."""
from functools import partial
import numpy as np
import pandas as pd
from . import models, autoregression, event_models
from .arima import fit_arima, forecast_arima
from .data import core_features
from .event_features import make_dataset, catalogue
from .event_metrics import metric_row
from .features import asof_features, history_at

AR_NAMES = {'ar_only': 'lgbm_ar', 'ar_news': 'lgbm_news', 'ar_fx': 'lgbm_fx', 'ar_news_fx': 'lgbm_news_fx'}


def annual_threshold(history, minimum=48):
    if len(history) < 52:
        return np.nan
    values = np.asarray(history, float)[-52:]
    values = values[np.isfinite(values)]
    return float(values.mean()+2*values.std(ddof=1)) if len(values) >= minimum else np.nan


def dataset(panel, news, truth, cfg, fx=None):
    threshold = partial(annual_threshold, minimum=cfg['min_year_observations'])
    x, meta = make_dataset(panel, news, truth, fx, threshold_fn=threshold, canonical=True)
    return x, meta, catalogue(panel, truth, threshold)


def evaluation_mask(meta, cfg):
    return meta.eligible & meta.origin_index.ge(cfg['first_origin']) & meta.forecast_week.le(pd.Timestamp(cfg['last_target']))


def run(panel, news, truth, ecb, cfg):
    settings = cfg['escalation']
    fx = asof_features(ecb, panel.index)
    x, meta, events = dataset(panel, news, truth, settings, fx)
    standard = models.feature_matrices(core_features(panel, news), fx)
    ar = autoregression.build_bundle(panel, news, fx)
    labels = pd.DataFrame(np.tile(truth.to_numpy(), (len(panel), 1)), index=panel.index, columns=panel.columns)
    for i in range(len(labels)):
        labels.iloc[i, i:] = np.nan
    eligible = evaluation_mask(meta, settings)
    last = int(meta.loc[eligible, 'origin_index'].max())
    pieces = []
    for start in range(settings['first_origin'], last+1, settings['refit_every']):
        mask = eligible & meta.origin_index.between(start, start+settings['refit_every']-1)
        history, _ = history_at(panel, start)
        fitted = {'ml': {name: {h: models.fit_ml(name, matrix, labels, start, h, cfg['seed'])
                               for h, matrix in standard.items()} for name in models.ML_MODELS},
                  'ets': {name: models.fit_ets(history, name) for name in models.ETS_MODELS}}
        arima = {name: fit_arima(history, order)[0] for name, order in
                 [('arima_1_0_0', (1,0,0)), ('arima_1_1_1', (1,1,1))]}
        boosted = {name: autoregression.fit_origin(ar, start, name, labels) for name in AR_NAMES}
        for origin, group in meta.loc[mask].groupby('origin_index'):
            rows, _ = models.predict_origin(standard, panel, origin, cfg['seed'], fitted)
            values = {name: np.array([r['predicted'] for r in rows if r['model']==name]) for name in models.MODELS}
            history, _ = history_at(panel, origin)
            values.update({name: forecast_arima(history, params) for name, params in arima.items()})
            values.update({AR_NAMES[name]: autoregression.forecast(ar, origin, fit, name) for name, fit in boosted.items()})
            for name, forecast in values.items():
                p = group.copy()
                p['predicted'] = forecast[p.horizon.to_numpy()-1]
                p['score'] = p.predicted/p.threshold
                pieces.append(p.assign(model=name, cutoff=1., probability=False, fit_origin=start))
        for spec in cfg['event_models']:
            fitted = event_models.fit(spec, x, meta, start, warmup=52)
            p = event_models.predict(spec, fitted, x, meta, mask)
            probability = event_models.is_probability(spec)
            pieces.append(p.assign(model=spec['id'], cutoff=settings['probability_cutoff'] if probability else 1.,
                                   probability=probability, fit_origin=start))
        print(f'Escalation: {panel.index[start].date()}', flush=True)
    for name, score in [('no_alert', 0.), ('always_alert', 1.)]:
        pieces.append(meta.loc[eligible].assign(model=name, predicted=np.nan, score=score, probability=True, cutoff=.5, fit_origin=-1))
    predictions = pd.concat(pieces, ignore_index=True)
    return summarize(predictions), predictions, episode_scores(predictions, events)


def summarize(predictions):
    rows = []
    for (name, h), p in predictions.groupby(['model', 'horizon']):
        rows.append({'model': name, 'horizon': h, 'block': 'all', **metric_row(p)})
        for year, sub in p.groupby(p.forecast_week.dt.year):
            rows.append({'model': name, 'horizon': h, 'block': str(year), **metric_row(sub)})
    return pd.DataFrame(rows).assign(benchmark='escalation_52w')


def episode_scores(predictions, events):
    rows = []
    for name, group in predictions.groupby('model'):
        advance = group[group.horizon.ge(2)].copy()
        advance['alarm'] = advance.score.gt(advance.cutoff)
        leads = []
        onsets = events.index[events.onset & events.index.isin(advance.forecast_week)]
        for week in onsets:
            hits = advance[advance.alarm & advance.forecast_week.eq(week)]
            if len(hits):
                leads.append(float(((week-hits.issued_at_utc)/pd.Timedelta(days=1)).max()))
        advance['confirmed_alarm'] = advance.alarm & advance.positive
        issued = advance.groupby('origin_index').agg(alarm=('alarm', 'any'), confirmed=('confirmed_alarm', 'any'))
        rows.append({'model': name, 'episodes': len(onsets), 'detected_episodes': len(leads),
            'episode_recall_any_h2_h4': len(leads)/len(onsets) if len(onsets) else np.nan,
            'issues_any_h2_h4': len(issued), 'alarm_issues_any_h2_h4': int(issued.alarm.sum()),
            'false_alarm_issues_any_h2_h4': int((issued.alarm & ~issued.confirmed).sum()),
            'median_lead_days_detected': float(np.median(leads)) if leads else np.nan})
    return pd.DataFrame(rows)
