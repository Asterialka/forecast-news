from __future__ import annotations
import numpy as np
import pandas as pd

from .features import causal_fill
from .common import ROOT, sha256

QUANTILES = [.5, .8, .9]


def context_at(panel, news, origin, length, multivariate):
    start = max(0, origin+1-length)
    target = causal_fill(panel.iloc[origin, :origin+1]).to_numpy(float)[start:]
    result = [target]
    if multivariate:
        for name in ["total_events_latest", "conflict_share_latest", "tone_latest", "root_13_latest"]:
            result.append(causal_fill(news[name].iloc[:origin+1]).to_numpy(float)[start:])
    return np.asarray(result, dtype=np.float32)


def load_engine(engine, path):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if engine == "chronos":
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(str(path), device_map="cpu", local_files_only=True)
        model.model.eval()
    else:
        from timesfm3 import ModelConfig, TimesFM3Forecaster
        model = TimesFM3Forecaster(ModelConfig(checkpoint_path=str(path), device="cpu",
                                              per_core_batch_size=4, local_files_only=True))
    return model


def forecast(model, engine, contexts, context_length):
    if engine == "chronos":
        outputs, _ = model.predict_quantiles(contexts, prediction_length=4,
            quantile_levels=QUANTILES, batch_size=8, context_length=context_length, cross_learning=False)
        result = np.array([o.detach().cpu().numpy()[0] for o in outputs])
    else:
        outputs = model.predict_batch(contexts, horizon=4, return_quantiles=True,
            use_symmetric_averaging=False, make_positive=True, sort_quantiles=True)
        result = np.array([np.asarray(o.quantiles)[0][:, [4, 7, 8]] for o in outputs])
    if result.shape != (len(contexts), 4, 3) or not np.isfinite(result).all():
        raise ValueError(f"Unexpected {engine} result: {result.shape}")
    return np.maximum(0., result)


def resolve_weights(engine, spec, directory=None):
    from pathlib import Path
    if directory is None:
        from huggingface_hub import snapshot_download
        directory = snapshot_download(spec["repo"], revision=spec["revision"],
                                      allow_patterns=["config.json", "model.safetensors"])
    directory = Path(directory)
    for name, key in [("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")]:
        if sha256(directory/name) != spec[key]:
            raise ValueError(f"Unexpected {engine} weights: {directory/name}")
    return directory


def infer(model, engine, panel, news, origins, length, multi, block):
    contexts = [context_at(panel, news, i, length, multi) for i in origins]
    values = []
    for offset in range(0, len(origins), 16):
        values.extend(forecast(model, engine, contexts[offset:offset+16], length))
        print(f'{engine} c{length} {block}: {min(offset+16,len(origins))}/{len(origins)}', flush=True)
    return np.asarray(values, dtype=float), {origin: i for i, origin in enumerate(origins)}


def event_predictions(spec, meta, mask, values, mapping, length, multi):
    pieces = []
    for setting in spec['models']:
        if (setting['context'], setting['multivariate']) != (length, multi):
            continue
        p = meta.loc[mask].copy()
        qi = QUANTILES.index(setting['quantile'])
        p['predicted'] = [values[mapping[i], h-1, qi] for i, h in zip(p.origin_index, p.horizon)]
        p['score'] = p.predicted/p.threshold
        pieces.append(p.assign(model=setting['id'], cutoff=1., probability=False, fit_origin=-1))
    return pieces


def run(engine, cfg, directory, panel, news, truth, benchmarks):
    from . import count, models, spike, escalation
    from .event_features import make_dataset
    spec = cfg['foundation'][engine]
    model = load_engine(engine, resolve_weights(engine, spec, directory))
    metrics, predictions, details = [], [], {}
    settings = {(s['context'], s['multivariate']) for s in spec['models']}
    if 'spike' in benchmarks:
        _, meta = make_dataset(panel, news, truth)
        pieces = []
        for length, multi in sorted(settings):
            for block, mask in [('validation', spike.validation(meta)), ('test', spike.test_mask(meta))]:
                origins = sorted(meta.loc[mask, 'origin_index'].unique())
                values, mapping = infer(model, engine, panel, news, origins, length, multi, block)
                pieces.extend(p.assign(block=block) for p in event_predictions(spec, meta, mask, values, mapping, length, multi))
        p = pd.concat(pieces, ignore_index=True)
        metrics.append(spike.summarize(p))
        predictions.append(p.assign(benchmark='spike_2025'))
    if {'count', 'escalation'} & set(benchmarks):
        _, meta, events = escalation.dataset(panel, news, truth, cfg['escalation'])
        mask = escalation.evaluation_mask(meta, cfg['escalation'])
        count_origins = list(range(103,162))+list(range(167,191))+list(range(195,207))
        origins = sorted(set(meta.loc[mask, 'origin_index']) | set(count_origins))
        settings |= {tuple(s) for s in spec['forecast_contexts']}
        old_truth = pd.read_csv(ROOT/'data/preprocessed/truth_2024.csv', index_col=0, parse_dates=True).iloc[:, 0]
        count_rows, annual_pieces = [], []
        for length, multi in sorted(settings):
            values, mapping = infer(model, engine, panel, news, origins, length, multi, 'forecast/annual')
            if 'escalation' in benchmarks:
                annual_pieces.extend(event_predictions(spec, meta, mask, values, mapping, length, multi))
            if 'count' in benchmarks:
                name = f"{engine}_c{length}_{'news' if multi else 'uni'}_q50"
                for origin in count_origins:
                    block, snapshot = ('selection',166) if origin <= 161 else ('calibration',195) if origin < 195 else ('holdout',None)
                    for h in range(1,5):
                        if origin+h >= 208:
                            continue
                        count_rows.append({'model':name,'block':block,'origin_index':origin,'horizon':h,
                            'forecast_week':panel.index[origin+h], 'predicted':values[mapping[origin],h-1,0],
                            'actual':float(panel.iloc[snapshot,origin+h]) if snapshot is not None else float(old_truth.iloc[origin+h]),
                            'mase_scale':float(panel.iloc[origin,:origin+1].diff().abs().mean())})
        if annual_pieces:
            p = pd.concat(annual_pieces, ignore_index=True)
            metrics.append(escalation.summarize(p))
            details['episodes'] = escalation.episode_scores(p, events)
            predictions.append(p.assign(benchmark='escalation_52w'))
        if count_rows:
            p = pd.DataFrame(count_rows)
            table = count.summarize(p)
            selected = table[table.block.eq('selection') & table.horizon.eq(0)].sort_values('RMSE').iloc[0].model
            radii = models.calibrate(p[p.block.eq('calibration') & p.model.eq(selected)])
            holdout = models.add_intervals(p[p.block.eq('holdout') & p.model.eq(selected)], radii)
            details['foundation_count'] = {'engine':engine,'selected_model':selected,'criterion':'Early CV pooled RMSE',
                'interval_level':.9,'calibration_radii':radii,'interval_metrics':models.interval_metrics(holdout)}
            metrics.append(table)
            predictions.append(p.assign(benchmark='count_2024'))
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True), details

