"""Regression tests for exact Vasicek likelihood and its reusable API."""

import numpy as np
import pandas as pd
import pytest

from russian_rates.models.vasicek import (
    calibrate_profile,
    conditional_moments,
    prepare_transitions,
    profile_params,
    standardized_residuals,
    vasicek_nll,
)


def _transitions():
    r0 = np.array([0.07, 0.072, 0.071, 0.076, 0.074, 0.080, 0.077])
    r1 = np.array([0.072, 0.071, 0.076, 0.074, 0.080, 0.077, 0.079])
    dt = np.array([1, 3, 1, 1, 2, 3, 1], dtype=float) / 365
    return r0, r1, dt


def _simulated(seed=41, n=900):
    rng = np.random.default_rng(seed)
    dt = np.resize(np.array([7, 30, 14, 3], dtype=float) / 365, n)
    a, b, sigma = 1.4, 0.085, 0.027
    values = np.empty(n + 1)
    values[0] = 0.07
    for i, step in enumerate(dt):
        mu, var = conditional_moments(a, b, sigma, values[i], step)
        values[i + 1] = mu + np.sqrt(var) * rng.standard_normal()
    return values[:-1], values[1:], dt


def test_prepare_transitions_keeps_actual_intervals_and_converts_percent():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-01", "2026-01-02"]),
        "spread_pp": [-0.20, -0.10, -0.15],
    })
    r0, r1, dt = prepare_transitions(frame, "spread_pp")
    np.testing.assert_allclose(r0, [-0.001, -0.0015])
    np.testing.assert_allclose(r1, [-0.0015, -0.002])
    np.testing.assert_allclose(dt, [1 / 365, 3 / 365])
    # Does not alter input or fill in weekends.
    assert frame.iloc[0]["date"] == pd.Timestamp("2026-01-05")
    np.testing.assert_allclose(
        prepare_transitions(frame, "spread_pp", values_in_percent=False)[0],
        [-0.1, -0.15],
    )


@pytest.mark.parametrize(
    "data",
    [
        {"date": ["2026-01-01", "2026-01-01", "2026-01-02"], "spread_pp": [0, 1, 2]},
        {"date": ["2026-01-01", None, "2026-01-02"], "spread_pp": [0, 1, 2]},
        {"date": ["2026-01-01", "2026-01-02", "2026-01-03"], "spread_pp": [0, np.nan, 2]},
    ],
)
def test_prepare_transitions_rejects_missing_or_duplicate_data(data):
    with pytest.raises(ValueError):
        prepare_transitions(pd.DataFrame(data), "spread_pp")


def test_prepare_transitions_requires_required_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        prepare_transitions(pd.DataFrame({"date": ["2026-01-01"]}), "spread_pp")


def test_exact_conditional_moments_and_short_intervals():
    mu, var = conditional_moments(
        0.8, 0.10, 0.03, np.array([0.08, 0.09]), np.array([1, 3]) / 365
    )
    dt = np.array([1, 3]) / 365
    np.testing.assert_allclose(mu, 0.10 + (np.array([0.08, 0.09]) - 0.10) * np.exp(-0.8 * dt))
    np.testing.assert_allclose(var, 0.03**2 * (1 - np.exp(-1.6 * dt)) / 1.6)
    assert (var > 0).all()


def test_analytic_profile_nll_matches_full_nll():
    r0, r1, dt = _transitions()
    for a in [0.05, 0.8, 2.7]:
        b, sigma, profiled_nll = profile_params(a, r0, r1, dt)
        assert sigma > 0
        assert np.isfinite(b)
        assert vasicek_nll((a, b, sigma), r0, r1, dt) == pytest.approx(profiled_nll, abs=1e-10)


def test_nll_rejects_invalid_model_parameters():
    r0, r1, dt = _transitions()
    assert np.isinf(vasicek_nll((0, 0.1, 0.03), r0, r1, dt))
    assert np.isinf(vasicek_nll((0.5, 0.1, 0), r0, r1, dt))
    assert vasicek_nll((0.5, 0.1, 1e-6), r0, r1, dt) > vasicek_nll(
        (0.5, 0.1, 0.03), r0, r1, dt
    )


def test_calibration_fits_simulated_ou_and_standardized_innovations():
    r0, r1, dt = _simulated()
    fit = calibrate_profile(r0, r1, dt)
    assert 0.2 < fit["a"] < 3.5
    assert fit["b"] == pytest.approx(0.085, abs=0.025)
    assert fit["sigma"] == pytest.approx(0.027, rel=0.15)
    assert fit["nll"] <= profile_params(0.4, r0, r1, dt)[2] + 1e-6
    assert np.isnan(fit["a_lower"]) or fit["a_lower"] <= fit["a"]
    assert np.isnan(fit["a_upper"]) or fit["a_upper"] >= fit["a"]
    z = standardized_residuals(fit["a"], fit["b"], fit["sigma"], r0, r1, dt)
    assert len(z) == len(r1)
    assert abs(float(np.mean(z))) < 0.15
    assert float(np.std(z)) == pytest.approx(1.0, abs=0.02)


def test_invalid_shapes_and_bounds_rejected():
    r0, r1, dt = _transitions()
    with pytest.raises(ValueError, match="equal length"):
        profile_params(0.5, r0[:-1], r1, dt)
    with pytest.raises(ValueError, match="positive"):
        profile_params(0.5, r0, r1, -dt)
    with pytest.raises(ValueError, match="a_bounds"):
        calibrate_profile(r0, r1, dt, a_bounds=(-1, 2))
    with pytest.raises(ValueError, match="confidence"):
        calibrate_profile(r0, r1, dt, confidence=1.0)
