"""FCNR(B) funding book module — RBI special swap window context.

REAL-WORLD CONTEXT (verified via web search, not fabricated — see README for
sources): on 8 June 2026, RBI circular RBI/2026-27/99 opened a USD-INR forex
swap facility for fresh FCNR(B) deposits (3-5yr tenor), absorbing banks'
currency-hedging cost so they could offer NRIs sharply higher rates without
extra cost to the bank. Inflows (~$52.3bn FCNR-specific, ~$56bn+ across all
three windows) far exceeded the comparable 2013 scheme, prompting RBI to move
the mobilisation deadline forward to 31 August 2026 (from 30 September).
Banks can still settle dollar-rupee swaps against these deposits with RBI
until 11 September 2026. After that, banks bear the FULL hedging cost
themselves on any fresh or rolled-over FCNR(B) funding. A meaningful share of
inflows came through leveraged structures yielding depositors ~14-15%
effective returns — flagged by bankers as unlikely to stay once yields
normalise, creating real rollover/retention risk once the subsidy ends.

Everything BELOW this docstring — the specific book (notional, tenor, rate),
the retention-multiplier values, and the funding-gap/hedging-cost formulas —
is a synthetic illustration built on top of that real regulatory backdrop,
not real GIFT City IBU data. Retention risk is modeled on only 2 of the 9
shock themes (Fed Rate Move, India Sovereign Spread) — the two macro drivers
that plausibly move NRI deposit retention; the other 7 themes are left at
retention_multiplier = 1.0 (no modeled effect), stated openly, not hidden.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

MOBILISATION_DEADLINE = _dt.date(2026, 8, 31)
SWAP_UNWIND_DEADLINE = _dt.date(2026, 9, 11)
CIRCULAR_REF = "RBI/2026-27/99 (FMOD.MAOG.No.S-56/01.06.016/2026-27), 8 June 2026"

FCNR_DEFAULTS = {
    "notional": 300_000_000.0,       # USD
    "tenor_years": 3,
    "rate_paid": 0.065,               # 6.5% to depositors
    "swap_booked_date": _dt.date(2026, 7, 15),  # within the Jun 8 - Aug 31 window
}

BASE_POST_WINDOW_HEDGING_COST_BPS = 350

# Retention multipliers — SCOPED to exactly these two themes, by design (see
# module docstring). Anchor values are as specified; intermediate severities
# within each theme are linearly interpolated between the given anchors.
_FED_RETENTION = {
    "Fed -50bps": 1.15,
    "Fed -25bps": 1.05,
    "Fed +25bps": 0.92,   # interpolated between -25bps(1.05) and +50bps(0.85)
    "Fed +50bps": 0.85,
    "Fed +75bps": 0.78,   # interpolated between +50bps(0.85) and +100bps(0.70)
    "Fed +100bps": 0.70,
}
_SPREAD_RETENTION = {
    "Spread +50bps": 0.90,
    "Spread +100bps": 0.8167,
    "Spread +150bps": 0.7333,
    "Spread +200bps": 0.65,
}
RETENTION_SCOPE_THEMES = {"fed_rate", "sovereign_spread"}


def retention_multiplier(theme_key: str, severity_label: str) -> float:
    """1.0 (no modeled effect) for every theme except Fed Rate Move and
    India Sovereign Spread, where NRI deposit retention plausibly reacts to
    the rate/spread move."""
    if theme_key == "fed_rate":
        return _FED_RETENTION.get(severity_label, 1.0)
    if theme_key == "sovereign_spread":
        return _SPREAD_RETENTION.get(severity_label, 1.0)
    return 1.0


def funding_gap(notional: float, ret_multiplier: float) -> float:
    """Dollar amount of FCNR(B) funding that could walk under this scenario
    and would need replacing."""
    return notional * (1 - ret_multiplier)


def post_window_hedging_cost_bps(fed_severity_value_bps: float) -> float:
    """Cost (bps) of self-funded hedging on fresh/rolled-over FCNR(B)
    funding raised after the 11 Sept 2026 swap-unwind deadline, once RBI
    stops absorbing the hedge cost. Base 350bps, adjusted by the Fed
    scenario's own implied rate-differential shift (a hike widens the
    USD-INR rate differential a bank must hedge against; a cut narrows it)."""
    return BASE_POST_WINDOW_HEDGING_COST_BPS + fed_severity_value_bps


def days_remaining(today: _dt.date | None = None) -> int:
    """Days remaining until the mobilisation deadline (31 Aug 2026);
    negative once the deadline has passed."""
    today = today or _dt.date.today()
    return (MOBILISATION_DEADLINE - today).days


def days_to_swap_unwind(today: _dt.date | None = None) -> int:
    today = today or _dt.date.today()
    return (SWAP_UNWIND_DEADLINE - today).days


@dataclass
class FcnrStatus:
    days_to_mobilisation: int
    days_to_swap_unwind: int
    window_open: bool


def fcnr_status(today: _dt.date | None = None) -> FcnrStatus:
    today = today or _dt.date.today()
    return FcnrStatus(
        days_to_mobilisation=days_remaining(today),
        days_to_swap_unwind=days_to_swap_unwind(today),
        window_open=today <= MOBILISATION_DEADLINE,
    )
