"""Synthetic daily FX return generation for GIFT Risk.

Generates realistic daily log-return series for cross-border currency pairs
relevant to a GIFT IFSC IBU treasury desk. Distributions use a skewed
Student-t base (fat tails + mild skew) rather than a clean Gaussian, because
real FX returns exhibit excess kurtosis and asymmetry — especially in
EM-linked pairs like USD/INR.

All data here is SYNTHETIC. Parameters are chosen to be realistic in
magnitude (annualised vols in line with historically observed FX vol
regimes) but are not calibrated to any real market feed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Currency pair parameterisation (annualised vol -> daily, 252 trading days).
# USD/INR: managed-float, moderate vol, persistent depreciation drift, fat
#   tails from episodic RBI-intervention gaps.
# SGD/INR: cross of a low-vol managed SGD against INR — slightly lower vol.
# AED/INR: AED is USD-pegged, so AED/INR inherits USD/INR dynamics almost
#   one-for-one with marginally different noise.
# ---------------------------------------------------------------------------
PAIR_PARAMS = {
    "USD/INR": {
        "spot": 88.20,          # synthetic spot level
        "ann_vol": 0.048,       # ~4.8% annualised — typical managed-float regime
        "ann_drift": 0.025,     # mild INR depreciation drift
        "skew": -2.5,           # left-skew in INR terms (depreciation gaps)
        "df": 4.5,              # Student-t degrees of freedom -> fat tails
    },
    "SGD/INR": {
        "spot": 65.10,
        "ann_vol": 0.055,
        "ann_drift": 0.018,
        "skew": -1.8,
        "df": 5.0,
    },
    "AED/INR": {
        "spot": 24.02,
        "ann_vol": 0.049,       # tracks USD/INR via the AED-USD peg
        "ann_drift": 0.024,
        "skew": -2.4,
        "df": 4.5,
    },
}

TRADING_DAYS = 252


def generate_returns(pair: str, n_days: int = 1500, seed: int | None = 42) -> np.ndarray:
    """Generate synthetic daily log returns for one currency pair.

    Uses a skew-t distribution (scipy skewnorm applied to a t base via
    Azzalini-style construction) to get mild skew and fat tails together.
    Returns are in FX-rate terms: negative = INR appreciation vs. foreign leg,
    positive = INR depreciation (loss for a short-foreign-currency book is
    scenario-dependent; we treat losses on the left tail downstream).
    """
    if pair not in PAIR_PARAMS:
        raise KeyError(f"Unknown pair {pair!r}; expected one of {list(PAIR_PARAMS)}")
    p = PAIR_PARAMS[pair]
    rng = np.random.default_rng(seed)

    daily_vol = p["ann_vol"] / np.sqrt(TRADING_DAYS)
    daily_drift = p["ann_drift"] / TRADING_DAYS

    # Fat-tailed core: standardised Student-t
    t_draws = rng.standard_t(p["df"], size=n_days)
    t_draws /= np.sqrt(p["df"] / (p["df"] - 2))  # unit variance

    # Mild skew: Azzalini transform using correlated normal draws
    delta = p["skew"] / np.sqrt(1 + p["skew"] ** 2)
    z = rng.standard_normal(n_days)
    skewed = delta * np.abs(z) + np.sqrt(1 - delta**2) * t_draws
    # re-centre / re-scale to unit variance, zero mean
    skewed = (skewed - skewed.mean()) / skewed.std()

    returns = daily_drift + daily_vol * skewed
    return returns


def price_path(pair: str, returns: np.ndarray) -> np.ndarray:
    """Convert log returns to a synthetic FX rate path from the pair's spot."""
    spot = PAIR_PARAMS[pair]["spot"]
    return spot * np.exp(np.cumsum(returns))


def summary_stats(returns: np.ndarray) -> dict:
    return {
        "mean_daily": float(np.mean(returns)),
        "vol_daily": float(np.std(returns)),
        "ann_vol": float(np.std(returns) * np.sqrt(TRADING_DAYS)),
        "skewness": float(stats.skew(returns)),
        "excess_kurtosis": float(stats.kurtosis(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }
