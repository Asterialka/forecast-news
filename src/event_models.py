import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .event_features import feature_columns


def is_probability(spec):
    return spec['target'] == 'classifier' if spec['kind'] == 'lightgbm' else spec['kind'] not in ['ridge', 'tree_q90']


def fit(spec, x, meta, origin, warmup=12):
    mask = meta.eligible & meta.origin_index.ge(warmup) & (meta.origin_index+meta.horizon+1 <= origin)
    columns = feature_columns(x, spec['features'])
    probability = is_probability(spec)
    target = meta.loc[mask, 'positive'].astype(int) if probability else (
        np.log1p(meta.loc[mask, 'actual'])-np.log1p(meta.loc[mask, 'anchor']))
    if probability and target.nunique() < 2:
        return float(target.iloc[0]), columns
    kind = spec['kind']
    if kind == 'lightgbm':
        regularized = spec['complexity'] == 'regularized'
        params = dict(n_estimators=140, learning_rate=.035, num_leaves=5 if regularized else 9,
            max_depth=3, min_child_samples=15 if regularized else 5, reg_lambda=20. if regularized else 3.,
            reg_alpha=.1, max_bin=31, verbosity=-1, n_jobs=1, random_state=42, deterministic=True, force_col_wise=True)
        model = LGBMClassifier(**params) if probability else LGBMRegressor(
            objective='quantile', alpha=int(spec['target'][1:])/100, **params)
        model.fit(x.loc[mask, columns].to_numpy(), target.to_numpy(), feature_name=columns)
    else:
        strength = spec['strength']
        if kind == 'logistic':
            estimator = LogisticRegression(C=1/strength, class_weight='balanced', max_iter=1000, random_state=42)
        elif kind in ['extra_trees', 'random_forest', 'tree_q90']:
            cls = {'extra_trees': ExtraTreesClassifier, 'random_forest': RandomForestClassifier,
                   'tree_q90': ExtraTreesRegressor}[kind]
            params = {'class_weight': 'balanced'} if probability else {}
            estimator = cls(n_estimators=180, min_samples_leaf=strength+2, max_features=.8,
                            n_jobs=1, random_state=42, **params)
        elif kind == 'knn':
            estimator = KNeighborsClassifier(n_neighbors=5, weights='distance')
        elif kind == 'ridge':
            estimator = Ridge(alpha=100*strength)
        else:
            raise ValueError(f'Unknown event model: {kind}')
        model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), estimator)
        model.fit(x.loc[mask, columns].to_numpy(), target.to_numpy())
    return model, columns


def predict(spec, fitted, x, meta, mask):
    model, columns = fitted
    p = meta.loc[mask].copy()
    values = x.loc[mask, columns].to_numpy()
    if is_probability(spec):
        if isinstance(model, float):
            p['score'] = model
        elif spec['kind'] == 'lightgbm':
            p['score'] = model.booster_.predict(values, num_threads=1)
        else:
            p['score'] = model.predict_proba(values)[:, 1]
        p['predicted'] = np.nan
    else:
        if spec['kind'] == 'lightgbm':
            response = model.booster_.predict(values, num_threads=1)
        elif spec['kind'] == 'tree_q90':
            transformed = model[:-1].transform(values)
            response = np.quantile([tree.predict(transformed) for tree in model[-1].estimators_], .9, axis=0)
        else:
            response = model.predict(values)
        p['predicted'] = np.maximum(0, np.expm1(np.log1p(p.anchor)+response))
        p['score'] = p.predicted/p.threshold
    return p
