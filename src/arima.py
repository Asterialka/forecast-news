from __future__ import annotations
import numpy as np

import warnings
from statsmodels.tsa.arima.model import ARIMA

def arima_model(history, order, scale):
    return ARIMA(np.asarray(history) / scale, order=tuple(order),
                 trend="c" if order[1] == 0 else "n",
                 enforce_stationarity=True, enforce_invertibility=True)


def fit_arima(history, order):
    scale = max(float(np.median(history)), 1.)
    notices, attempts = [], []
    fitted = None
    for optimizer, maxiter in [("lbfgs", 300), ("powell", 1000)]:
        start = fitted.params if fitted is not None and np.isfinite(fitted.params).all() else None
        with warnings.catch_warnings(record=True) as caught:
            fitted = arima_model(history, order, scale).fit(
                start_params=start, cov_type="none",
                method_kwargs={"method": optimizer, "maxiter": maxiter})
        notices.extend(str(w.message) for w in caught)
        converged = bool(fitted.mle_retvals.get("converged", False))
        attempts.append({"optimizer": optimizer, "converged": converged})
        if converged and np.isfinite(fitted.params).all() and np.isfinite(fitted.llf):
            break
    else:
        raise RuntimeError(f"ARIMA{order} did not converge on {len(history)} weeks")
    parameters = {"order": list(order), "trend": "c" if order[1] == 0 else "n",
                  "scale": scale, "parameter_names": fitted.param_names,
                  "parameters": fitted.params.tolist()}
    diagnostics = {"attempts": attempts, "warnings": list(dict.fromkeys(notices)),
                   "log_likelihood_scaled": float(fitted.llf)}
    return parameters, diagnostics


def forecast_arima(history, parameters, steps=4):
    # Refilter the currently available vintage, including historical revisions,
    # with fixed coefficients and the fixed training scale. No holdout fitting.
    state = arima_model(history, parameters["order"], parameters["scale"]).filter(
        parameters["parameters"], cov_type="none")
    predicted = np.asarray(state.forecast(steps)) * parameters["scale"]
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite ARIMA forecast")
    return np.maximum(0., predicted)
