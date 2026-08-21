"""Hedge ratio decision support for GIFT Risk.

Given a target hedge ratio, computes the FX forward trade (direction +
notional) needed to move a position's forward cover from its current
hedge ratio to the desired one, and the resulting VaR impact.

Simplifying assumption (disclosed in the UI): the hedge instrument is a
forward in the SAME currency pair as the underlying exposure, so hedge
effectiveness is 1-for-1 and VaR scales linearly with the unhedged
fraction:

    hedged_VaR(h) = unhedged_VaR * (1 - h)

Real hedge accounting would additionally need forward points / cost of
carry, basis risk if the hedge instrument differs from the exposure
currency, counterparty credit limits, and mark-to-market on existing
forwards — out of scope here, and called out in the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPOSURE_LABELS = {
    "payable": "Payable — desk owes foreign currency",
    "receivable": "Receivable — desk is owed foreign currency",
}


@dataclass
class HedgeTrade:
    current_ratio: float
    target_ratio: float
    unhedged_var: float           # 95% VaR at 0% hedge, as a fraction of notional
    current_hedged_var: float
    target_hedged_var: float
    var_reduction: float          # current_hedged_var - target_hedged_var (>0 = de-risking)
    trade_notional: float         # absolute notional to trade, in the position's currency terms
    trade_action: str             # "Buy" or "Sell"
    trade_currency: str           # e.g. "USD"
    trade_instrument: str
    is_increase: bool             # True = adding hedge cover, False = unwinding


def foreign_currency(pair: str) -> str:
    return pair.split("/")[0]


def compute_hedge_trade(
    unhedged_var_pct: float,
    notional: float,
    current_ratio: float,
    target_ratio: float,
    exposure_type: str,
    pair: str,
    tenor: str = "1-month",
) -> HedgeTrade:
    """unhedged_var_pct is a fraction (e.g. 0.0154 for 1.54%); ratios are
    fractions in [0, 1]."""
    current_hedged = unhedged_var_pct * (1 - current_ratio)
    target_hedged = unhedged_var_pct * (1 - target_ratio)
    delta = target_ratio - current_ratio
    is_increase = delta > 1e-9

    fcy = foreign_currency(pair)
    if exposure_type == "payable":
        # owes FCY: adding cover = lock in a purchase rate = Buy FCY forward
        action = "Buy" if is_increase else "Sell"
    else:
        # owed FCY: adding cover = lock in a sale rate = Sell FCY forward
        action = "Sell" if is_increase else "Buy"

    return HedgeTrade(
        current_ratio=current_ratio,
        target_ratio=target_ratio,
        unhedged_var=unhedged_var_pct,
        current_hedged_var=current_hedged,
        target_hedged_var=target_hedged,
        var_reduction=current_hedged - target_hedged,
        trade_notional=abs(delta) * notional,
        trade_action=action,
        trade_currency=fcy,
        trade_instrument=f"{tenor} FX forward ({pair})",
        is_increase=is_increase,
    )
