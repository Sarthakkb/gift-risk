"""Portfolio & hedge-ratio engine for GIFT Risk.

Hardcoded (but UI-editable) portfolio: three books, each with a notional and
a TARGET hedge ratio the desk wants to maintain. Each book's CURRENT hedge
ratio (existing forward cover) comes from the position metadata already used
by the Hedge Ratio tab (src/metadata.py / data/positions_metadata.json).

For every (theme, severity variant) the shock ladder produces a VaR estimate
per pair (src/shocks.py). This module turns that VaR into:

  0. Starting point — the desk rebalances to its target hedge ratio at the
     close of every trading day, so every shock is measured from that
     target, not from some other pre-existing gap.

  1. P&L impact — the shock's mark-to-market hit, assuming only the
     UNHEDGED fraction of the book takes the loss (the hedged fraction is
     locked in by existing forwards, so its INR value doesn't move):

         pnl_usd = -(1 - target_hedge_ratio) * var_pct * notional

     var_pct here is the quantum (or quantum-scaled) VaR estimate, per the
     brief — classical VaR is kept alongside for the per-cell detail view.

  2. Required hedge ratio to hold risk constant with severity: scales
     LINEARLY from today's target (at baseline volatility, vol_multiplier
     = 1.0) toward full cover (at the shock ladder's most extreme
     configured severity, vol_multiplier = VOL_MULTIPLIER_MAX = 3.0, GIFT
     City Liquidity — Extreme):

         required_ratio = target_h + (1 - target_h)
                           * (vol_multiplier - 1) / (VOL_MULTIPLIER_MAX - 1)

     This was chosen over a reciprocal VaR-ratio version (required_ratio =
     1 - (1-target_h)*baseline_var/var_pct) that was tried first: at a
     70-80% target hedge ratio the unhedged fraction is small enough that
     the reciprocal formula puts EVERY variant — even the mildest — past a
     5-point drift, flagging the whole ladder as action-required and
     destroying the mild-vs-severe signal the heatmap exists to show.
     Scaling off vol_multiplier directly — each variant's own authored
     severity knob — gives a transparent, well-graded scale instead:
     mild variants (vol ~1.1-1.3) land a few points from target; severe
     ones (vol ~2.5-3.0) approach full cover.

  3. Drift = required_ratio - target_hedge_ratio, and — if drift exceeds
     the threshold — a trade recommendation via src/hedge.py's
     compute_hedge_trade(), reusing its existing direction logic
     (payable/receivable, buy/sell) unchanged, fed the REQUIRED ratio as
     the "current" ratio to restore from.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.hedge import compute_hedge_trade

DEFAULT_DRIFT_THRESHOLD = 0.05
VOL_MULTIPLIER_MAX = 3.0  # GIFT City Liquidity — Extreme; the ladder's ceiling

# Hardcoded defaults per Part B — editable via UI number inputs in app.py.
PORTFOLIO_DEFAULTS = {
    "USD/INR": {"notional": 10_000_000, "target_hedge_ratio": 0.80},
    "SGD/INR": {"notional": 5_000_000, "target_hedge_ratio": 0.75},
    "AED/INR": {"notional": 3_000_000, "target_hedge_ratio": 0.70},
}


@dataclass
class ShockImpact:
    pnl_usd: float
    pnl_pct: float               # of notional
    required_hedge_ratio: float  # ratio needed to hold baseline $ risk tolerance constant
    drift: float                 # required_hedge_ratio - target_hedge_ratio
    action_required: bool
    trade_text: str | None       # formatted recommendation, or None


def shock_impact(
    var_pct: float,
    vol_multiplier: float,
    target_hedge_ratio: float,
    notional: float,
    exposure_type: str,
    pair: str,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> ShockImpact:
    """Compute P&L, hedge-ratio drift, and (if needed) a trade recommendation
    for one (pair, variant) combination. See module docstring for the
    required-hedge-ratio model."""
    h = target_hedge_ratio
    pnl_pct = -(1 - h) * var_pct
    pnl_usd = pnl_pct * notional

    severity_frac = (vol_multiplier - 1.0) / (VOL_MULTIPLIER_MAX - 1.0)
    severity_frac = max(0.0, min(1.0, severity_frac))
    required_ratio = h + (1 - h) * severity_frac

    drift = required_ratio - target_hedge_ratio
    action_required = abs(drift) > drift_threshold

    trade_text = None
    if action_required:
        trade = compute_hedge_trade(
            unhedged_var_pct=var_pct,
            notional=notional,
            current_ratio=target_hedge_ratio,
            target_ratio=required_ratio,
            exposure_type=exposure_type,
            pair=pair,
        )
        if trade.trade_notional >= 1:
            fcy = pair.split("/")[0]
            trade_text = (
                f"{trade.trade_action.upper()} {fcy} "
                f"{trade.trade_notional / 1_000_000:.2f}M vs INR\n"
                f"Restores {pair} hedge: {target_hedge_ratio:.0%} → {required_ratio:.0%}"
            )

    return ShockImpact(
        pnl_usd=pnl_usd,
        pnl_pct=pnl_pct,
        required_hedge_ratio=required_ratio,
        drift=drift,
        action_required=action_required,
        trade_text=trade_text,
    )
