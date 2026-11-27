"""Exact Gaussian Vasicek transitions and maximum-likelihood calibration.

Works with short rates or rate spreads. All internal rates, b and sigma are
decimals: 0.15 means 15% and sigma=0.002 means 0.2 percentage points per
square-root year. Time is in calendar years (elapsed days / 365).

This is historical P-measure estimation, NOT risk-neutral bond pricing.
Gaussian likelihoods and nominal profile intervals assume a time-homogeneous
Vasicek process; jumps and policy regime changes can invalidate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2


def prepare_transitions(
    frame: pd.DataFrame,
    rate_column: str,
    *,
    date_column: str = "date",
    values_in_percent: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build consecutive (previous rate, next rate, elapsed years) arrays.

    With values_in_percent=True, ruonia_pct or spread_pp is divided by 100.
    Intervals between dates are retained, without filling weekends. The
    first observation is an initial state, not a standalone transition.
    """
    missing = {date_column, rate_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    sample = frame[[date_column, rate_column]].copy()
    sample[date_column] = pd.to_datetime(sample[date_column], errors="raise")
    if sample[date_column].isna().any():
        raise ValueError("Missing observation dates")
    if sample[date_column].duplicated().any():
        raise ValueError("Duplicate observation dates")

    sample = sample.sort_values(date_column, kind="stable").reset_index(drop=True)
    if len(sample) < 3:
        raise ValueError("At least three dated observations are required")

    rates = pd.to_numeric(sample[rate_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(rates).all():
        raise ValueError("Rates must be finite and non-missing")
    if values_in_percent:
        rates = rates / 100.0

    elapsed_days = (
        sample[date_column].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        / 86400.0
    )
    dt = elapsed_days / 365.0
    if not np.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError("Non-positive or non-finite calendar intervals")

    return rates[:-1], rates[1:], dt


def _checked_transitions(
    r_prev: np.ndarray, r_next: np.ndarray, dt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0 = np.asarray(r_prev, dtype=float)
    r1 = np.asarray(r_next, dtype=float)
    gaps = np.asarray(dt, dtype=float)
    if (
        r0.ndim != 1
        or r1.ndim != 1
        or gaps.ndim != 1
        or len(r0) < 2
        or r0.shape != r1.shape
        or r0.shape != gaps.shape
    ):
        raise ValueError("r_prev, r_next and dt must be 1-D arrays of equal length >= 2")
    if not (np.isfinite(r0).all() and np.isfinite(r1).all() and np.isfinite(gaps).all()):
        raise ValueError("Rates and dt must contain only finite values")
    if not (gaps > 0).all():
        raise ValueError("Every dt must be positive")
    return r0, r1, gaps


def conditional_moments(
    a: float, b: float, sigma: float, r_prev: np.ndarray | float, dt: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditional mean and variance for one or more Vasicek steps.

    r_next | r_prev is normal with mean b + (r_prev-b)*exp(-a*dt) and
    variance sigma**2 * (1-exp(-2*a*dt))/(2*a).

    expm1 avoids cancellation for short observation intervals.
    """
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(sigma)):
        raise ValueError("Parameters must be finite")
    if a <= 0 or sigma <= 0:
        raise ValueError("Vasicek requires a > 0 and sigma > 0")

    rates, gaps = np.broadcast_arrays(
        np.asarray(r_prev, dtype=float), np.asarray(dt, dtype=float)
    )
    if not (np.isfinite(rates).all() and np.isfinite(gaps).all()) or not (gaps > 0).all():
        raise ValueError("Rates must be finite and intervals positive")

    phi = np.exp(-a * gaps)
    mu = b + (rates - b) * phi
    variance = sigma**2 * (-np.expm1(-2.0 * a * gaps)) / (2.0 * a)
    return mu, variance


def vasicek_nll(
    params: tuple[float, float, float] | np.ndarray,
    r_prev: np.ndarray,
    r_next: np.ndarray,
    dt: np.ndarray,
) -> float:
    """Negative exact Gaussian conditional log likelihood."""
    r0, r1, gaps = _checked_transitions(r_prev, r_next, dt)
    if len(params) != 3:
        raise ValueError("params must be (a, b, sigma)")
    a, b, sigma = (float(x) for x in params)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(sigma)):
        return float("inf")
    if a <= 0 or sigma <= 0:
        return float("inf")
    mu, variance = conditional_moments(a, b, sigma, r0, gaps)
    return float(0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + (r1 - mu) ** 2 / variance
    ))


def profile_params(
    a: float, r_prev: np.ndarray, r_next: np.ndarray, dt: np.ndarray
) -> tuple[float, float, float]:
    """Analytic MLE of (b, sigma) for fixed a; return (b, sigma, NLL)."""
    r0, r1, gaps = _checked_transitions(r_prev, r_next, dt)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be finite and strictly positive")

    phi = np.exp(-a * gaps)
    x = -np.expm1(-a * gaps)
    q = -np.expm1(-2.0 * a * gaps) / (2.0 * a)
    y = r1 - phi * r0

    b = float(np.sum(x * y / q) / np.sum(x**2 / q))
    residuals = y - b * x
    sigma2 = float(np.mean(residuals**2 / q))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("Residual diffusion variance must be positive")

    variance = sigma2 * q
    nll = float(0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + residuals**2 / variance
    ))
    return b, float(np.sqrt(sigma2)), nll


def calibrate_profile(
    r_prev: np.ndarray,
    r_next: np.ndarray,
    dt: np.ndarray,
    *,
    a_bounds: tuple[float, float] = (0.001, 20.0),
    confidence: float = 0.95,
) -> dict[str, float]:
    """Profile MLE over a with a nominal likelihood-ratio confidence interval.

    Returns a, b, sigma, nll, a_lower, a_upper, as in notebook 02. NaN
    endpoints mean the chosen a_bounds did not contain the interval endpoint.
    Confidence limits assume the Gaussian model is correctly specified.
    """
    r0, r1, gaps = _checked_transitions(r_prev, r_next, dt)
    low, high = (float(a_bounds[0]), float(a_bounds[1]))
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("a_bounds must satisfy 0 < lower < upper")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    def objective(a: float) -> float:
        return profile_params(a, r0, r1, gaps)[2]

    optimum = minimize_scalar(
        objective, bounds=(low, high), method="bounded", options={"xatol": 1e-8}
    )
    if not optimum.success:
        raise RuntimeError(f"Vasicek profile optimisation failed: {optimum.message}")

    a = float(optimum.x)
    b, sigma, min_nll = profile_params(a, r0, r1, gaps)
    threshold = min_nll + float(chi2.ppf(confidence, 1)) / 2.0

    def lower_endpoint() -> float:
        value = objective(low) - threshold
        if np.isclose(value, 0.0, atol=1e-10, rtol=0.0):
            return low
        if value < 0:
            return float("nan")
        return float(brentq(lambda x: objective(x) - threshold, low, a))

    def upper_endpoint() -> float:
        value = objective(high) - threshold
        if np.isclose(value, 0.0, atol=1e-10, rtol=0.0):
            return high
        if value < 0:
            return float("nan")
        return float(brentq(lambda x: objective(x) - threshold, a, high))

    return {
        "a": a,
        "b": b,
        "sigma": sigma,
        "nll": min_nll,
        "a_lower": lower_endpoint(),
        "a_upper": upper_endpoint(),
    }


def standardized_residuals(
    a: float,
    b: float,
    sigma: float,
    r_prev: np.ndarray,
    r_next: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """Standardized one-step innovations, in observation order."""
    r0, r1, gaps = _checked_transitions(r_prev, r_next, dt)
    mu, variance = conditional_moments(a, b, sigma, r0, gaps)
    return (r1 - mu) / np.sqrt(variance)
