"""10 macro stress scenarios for GIFT Risk.

Each scenario transforms a position's baseline daily-return distribution via:
  vol_multiplier   — scales dispersion
  mean_shift       — shifts the mean daily return (negative = INR depreciation
                     pressure / mark-to-market loss for the IBU book)
  skew_adjustment  — additional left-skew applied on top of baseline
  tail_weight      — probability mass injected into an extreme left tail
  distribution_type— normal | skewed | fat_tail | bimodal

`apply_scenario` produces the stressed return sample used by BOTH the
classical Monte Carlo engine and the quantum IQAE pipeline, so the two
methods always estimate the same underlying stressed distribution.
"""

from __future__ import annotations

import numpy as np

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "name": "Baseline: Normal Market",
        "description": "Standard trading conditions — no macro stress.",
        "narrative": (
            "Standard GIFT IFSC trading conditions. No macro stress. "
            "VaR reflects baseline FX volatility."
        ),
        "vol_multiplier": 1.0,
        "mean_shift": 0.0,
        "skew_adjustment": 0.0,
        "tail_weight": 0.0,
        "distribution_type": "normal",
    },
    "fed_rate_hike": {
        "name": "Fed Rate Hike",
        "description": "Unexpected Fed tightening — dollar rallies, INR depreciates.",
        "narrative": (
            "Fed tightens unexpectedly. Dollar rallies, INR depreciates. "
            "Capital outflows from GIFT City IBUs toward USD assets accelerate."
        ),
        "vol_multiplier": 1.6,
        "mean_shift": -0.008,
        "skew_adjustment": -1.0,
        "tail_weight": 0.0,
        "distribution_type": "skewed",
    },
    "oil_price_shock": {
        "name": "Oil Price Shock",
        "description": "Brent +30% — import bill widens deficit, INR under pressure.",
        "narrative": (
            "Brent crude spikes 30%. India, importing 85% of oil needs, faces "
            "higher import costs. Current account deficit widens, INR depreciates. "
            "GIFT City trade finance desks see elevated settlement risk."
        ),
        "vol_multiplier": 2.0,
        "mean_shift": -0.012,
        "skew_adjustment": -0.5,
        "tail_weight": 0.02,
        "distribution_type": "fat_tail",
    },
    "gold_flight_to_safety": {
        "name": "Gold Flight-to-Safety",
        "description": "Global risk-off — flows into gold and USD, EM outflows.",
        "narrative": (
            "Global risk-off episode. Capital flows into gold and USD. "
            "EM positions including GIFT City IBU books see outflows. "
            "INR weakens moderately."
        ),
        "vol_multiplier": 1.4,
        "mean_shift": -0.004,
        "skew_adjustment": -0.3,
        "tail_weight": 0.015,
        "distribution_type": "fat_tail",
    },
    "us_jobs_miss": {
        "name": "US Jobs Report Miss",
        "description": "NFP shock — recession fears, sharp EM sell-off, rapid onset.",
        "narrative": (
            "Non-Farm Payrolls miss consensus badly. Recession fears spike. "
            "EM currencies sell off sharply. Credit spreads widen at GIFT City. "
            "Rapid onset — catches unhedged IBU treasury desks."
        ),
        "vol_multiplier": 1.8,
        "mean_shift": -0.010,
        "skew_adjustment": -0.8,
        "tail_weight": 0.04,  # 3-5% extreme tail events — shock onset
        "distribution_type": "fat_tail",
    },
    "gift_liquidity_crunch": {
        "name": "GIFT City Liquidity Crunch",
        "description": "IFSC-specific settlement bottleneck — spreads blow out.",
        "narrative": (
            "GIFT IFSC-specific stress: simultaneous IBU liquidity demands create "
            "settlement bottlenecks. Bid-ask spreads blow out, positions are harder "
            "to unwind. A scenario unique to GIFT City's concentrated institutional "
            "structure — not in generic risk tools."
        ),
        "vol_multiplier": 2.5,
        "mean_shift": -0.018,
        "skew_adjustment": -0.5,
        "tail_weight": 0.03,
        "distribution_type": "bimodal",
    },
    "china_slowdown": {
        "name": "China Slowdown",
        "description": "Chinese GDP miss — Asia EM contagion via trade channels.",
        "narrative": (
            "Chinese GDP miss triggers Asia EM sell-off. INR, SGD, AED all under "
            "pressure via trade and capital channel contagion. GIFT City "
            "cross-border flows from Asian counterparties slow sharply."
        ),
        "vol_multiplier": 1.7,
        "mean_shift": -0.009,
        "skew_adjustment": -1.2,
        "tail_weight": 0.0,
        "distribution_type": "skewed",
    },
    "india_downgrade": {
        "name": "India Sovereign Downgrade",
        "description": "Rating downgrade — capital flight, funding costs spike.",
        "narrative": (
            "Rating agency downgrades India sovereign debt. Capital flight "
            "accelerates, INR depreciates sharply, GIFT City IBU funding costs "
            "spike. Structural impact — VaR underestimates multi-day risk in "
            "this regime."
        ),
        "vol_multiplier": 2.2,
        "mean_shift": -0.015,
        "skew_adjustment": -0.7,
        "tail_weight": 0.03,
        "distribution_type": "fat_tail",
    },
    "equity_crash": {
        "name": "Global Equity Crash (SPY -20%)",
        "description": "Correlated risk-asset sell-off — diversification fails.",
        "narrative": (
            "S&P 500 drops 20% rapidly. Risk assets sell off simultaneously — "
            "EM FX, credit, equities reprice lower together. Correlations spike "
            "to 1. GIFT City positions face simultaneous mark-to-market losses "
            "across books. Diversification fails."
        ),
        "vol_multiplier": 2.3,
        "mean_shift": -0.016,
        "skew_adjustment": -1.0,
        "tail_weight": 0.05,  # high kurtosis, extreme left tail
        "distribution_type": "fat_tail",
    },
    "crypto_contagion": {
        "name": "Crypto Contagion (BTC -50%)",
        "description": "Stablecoin redemptions tighten USD liquidity — EM pressure.",
        "narrative": (
            "Bitcoin drops 50% in 48 hours. Stablecoin redemptions create USD "
            "demand, tightening dollar liquidity. EM currencies including INR face "
            "pressure from USD repatriation flows. A modern transmission mechanism "
            "not in pre-2020 risk models — increasingly relevant for GIFT City "
            "digital asset exposure."
        ),
        "vol_multiplier": 1.5,
        "mean_shift": -0.007,
        "skew_adjustment": -0.9,
        "tail_weight": 0.0,
        "distribution_type": "skewed",
    },
}


def apply_scenario(base_returns: np.ndarray, scenario_key: str, seed: int = 0) -> np.ndarray:
    """Transform baseline daily returns into a scenario-stressed sample.

    The transform is applied deterministically per (scenario, seed) so the
    classical and quantum engines see an identical stressed distribution.
    Losses live in the LEFT tail (negative returns = P&L loss on the book).
    """
    sc = SCENARIOS[scenario_key]
    rng = np.random.default_rng(seed)
    r = base_returns.copy()

    # 1) scale dispersion around the mean, then shift the mean
    mu = r.mean()
    r = mu + (r - mu) * sc["vol_multiplier"]
    # mean_shift is expressed as a DAILY shock to the return distribution,
    # scaled down to keep 1-day VaR magnitudes plausible (shock unfolds
    # over ~a week, so ~1/5 hits the 1-day horizon)
    r = r + sc["mean_shift"] / 5.0

    # 2) skew adjustment: exponential-tilt the left side
    if sc["skew_adjustment"] != 0.0:
        sigma = r.std()
        left = r < r.mean()
        r[left] = r[left] + sc["skew_adjustment"] * 0.08 * sigma * np.abs(
            (r[left] - r.mean()) / sigma
        )

    # 3) tail_weight: replace a fraction of points with extreme left-tail draws
    if sc["tail_weight"] > 0.0:
        n_tail = int(len(r) * sc["tail_weight"])
        idx = rng.choice(len(r), size=n_tail, replace=False)
        sigma = r.std()
        r[idx] = r.mean() - rng.uniform(3.0, 6.0, size=n_tail) * sigma

    # 4) bimodal: split mass into a "settles" mode and a "gaps lower" mode
    if sc["distribution_type"] == "bimodal":
        n_gap = int(len(r) * 0.25)
        idx = rng.choice(len(r), size=n_gap, replace=False)
        sigma = r.std()
        r[idx] = r[idx] - 2.5 * sigma  # second mode: unwind-at-a-loss regime

    return r
