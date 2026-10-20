import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

from src.common import ROOT, sha256, write_json


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def worker(args, benchmarks):
    from src.data import load, config
    panel, news, truth, ecb = load()
    cfg, details = config(), {}
    if args.worker == 'count':
        from src.count import run
        scores, forecast, details['count'], predictions = run(panel, news, ecb, cfg)
        forecast.to_csv(args.output/'forecast.csv', index=False)
        predictions['benchmark'] = 'count_2024'
    elif args.worker == 'spike':
        from src.spike import run
        scores, predictions = run(panel, news, truth, ecb, cfg)
        predictions['benchmark'] = 'spike_2025'
    elif args.worker == 'escalation':
        from src.escalation import run
        scores, predictions, episodes = run(panel, news, truth, ecb, cfg)
        predictions['benchmark'] = 'escalation_52w'
        details['episodes'] = records(episodes)
    else:
        from src.foundation import run
        scores, predictions, details = run(args.worker, cfg, getattr(args, f'{args.worker}_dir'), panel, news, truth, benchmarks)
        if 'episodes' in details:
            details['episodes'] = records(details['episodes'])
        if 'foundation_count' in details:
            details['foundation_count']['interval_metrics'] = records(details['foundation_count']['interval_metrics'])
    scores.to_csv(args.output/'metrics.csv', index=False)
    write_json(args.output/'details.json', details)
    if args.save_predictions:
        for column in predictions.select_dtypes(include='floating'):
            predictions[column] = predictions[column].astype(float)
        predictions.to_csv(args.output/'predictions.csv.gz', index=False, compression={'method':'gzip', 'mtime':0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'count', 'spike', 'escalation'], default='all')
    parser.add_argument('--without-foundation', action='store_true')
    parser.add_argument('--chronos-dir', type=Path, help='Local weights; otherwise download pinned revision')
    parser.add_argument('--timesfm-dir', type=Path, help='Local weights; otherwise download pinned revision')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--worker', choices=['count','spike','escalation','chronos','timesfm'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    benchmarks = ['count','spike','escalation'] if args.stage == 'all' else [args.stage]
    if args.worker:
        worker(args, benchmarks)
        return
    stages = benchmarks + ([] if args.without_foundation else ['chronos','timesfm'])
    import pandas as pd
    parts, details = [], []
    with tempfile.TemporaryDirectory(prefix='forecast-run-') as temporary:
        for stage in stages:
            print(f'Running {stage}', flush=True)
            output = Path(temporary)/stage
            command = [sys.executable, '-B', str(ROOT/'run.py'), '--stage', args.stage,
                       '--worker', stage, '--output', str(output)]
            for engine in ['chronos','timesfm']:
                directory = getattr(args, f'{engine}_dir')
                if directory:
                    command += [f'--{engine}-dir', str(directory.resolve())]
            if args.save_predictions:
                command += ['--save-predictions']
            subprocess.run(command, check=True, cwd=ROOT)
            parts.append(pd.read_csv(output/'metrics.csv', dtype={'block':str}))
            details.append(json.loads((output/'details.json').read_text()))
        scores = pd.concat(parts, ignore_index=True)
        keys = ['benchmark','block','horizon']
        ref = scores[scores.model.eq('rolling_mean_4')][keys+['RMSE']].rename(columns={'RMSE':'baseline_RMSE'})
        scores = scores.drop(columns='RMSE_vs_baseline_pct', errors='ignore').merge(ref, on=keys, validate='many_to_one')
        scores['RMSE_vs_baseline_pct'] = 100*(1-scores.RMSE/scores.baseline_RMSE)
        scores = scores.drop(columns='baseline_RMSE').sort_values(keys+['model']).reset_index(drop=True)
        dependencies = ['numpy','pandas','scipy','statsmodels','lightgbm','holidays','scikit-learn','pyarrow']
        if not args.without_foundation:
            dependencies += ['torch','chronos-forecasting','timesfm','transformers','huggingface-hub','safetensors']
        result = {'completed_stages':stages,'python':sys.version.split()[0],
            'versions':{name:importlib.metadata.version(name) for name in dependencies},
            'inputs_sha256':sha256(ROOT/'data/preprocessed/manifest.json'),'config_sha256':sha256(ROOT/'config.json'),
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'run.py',*sorted((ROOT/'src').glob('*.py'))]},
            'protocol':{'count':'Original 232 early-CV / 96 calibration / 42 holdout cases; h0 pools h1..4',
                'spike':'2025 local jump above max(1.5*mean4,1.5*last); h0 pools h2..4; labels depend on issue',
                'escalation':'Annual mean + 2 sample std, target excluded, minimum 48/52 weeks; fixed 7-day truth; canonical labels',
                'escalation_training':'44 expanding refits, every 4 weeks; 37 model variants + 2 controls with foundation enabled',
                'escalation_signal':'Count forecast > issue-known annual threshold; classifier probability > 0.5',
                'advance':'h2/h3/h4: 6.5/13.5/20.5 days; h1 is already-started week',
                'embargo':'Training origin + horizon + 1 <= fit origin',
                'limitations':'Retrospective configurations; foundation weights postdate historical test, pretraining overlap unknown'},
            'count':next((d['count'] for d in details if 'count' in d),None),
            'foundation_count':[d['foundation_count'] for d in details if 'foundation_count' in d],
            'escalation_episodes':[row for d in details for row in d.get('episodes',[])],
            'metrics':records(scores)}
        # Publish only after all requested stages succeed.
        for stage in stages:
            for name in ['forecast.csv','predictions.csv.gz']:
                source = Path(temporary)/stage/name
                if source.exists():
                    target = f'{stage}_{name}' if name.startswith('predictions') else name
                    shutil.copyfile(source, args.output/target)
        write_json(args.output/'metrics.json', result)
        scores.to_csv(args.output/'metrics.csv', index=False)
    print(f'Done: {len(scores)} metric rows -> {args.output}', flush=True)


if __name__ == '__main__':
    main()
