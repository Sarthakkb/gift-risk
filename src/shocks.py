"""Progressive shock ladder for GIFT Risk — 9 macro themes, each with several
severity variants, replacing the old flat 10-scenario list.

Each theme is a single macro driver (Brent, Fed policy, USD/INR spot, India
sovereign spread, global equities, crypto, gold, China PMI, GIFT City
liquidity) walked across multiple severities, so the dashboard shows how risk
escalates along one axis rather than 10 unrelated one-off stories.

QUANTUM COMPUTE SCOPE: running IQAE on every one of the ~43 variants (x3
currency pairs = ~129 combos) is not worth doing on a simulator. Instead, for
each theme we run IQAE for real on one REPRESENTATIVE variant (the
middle-severity one), for each of the 3 currency pairs — 9 themes x 3 pairs =
27 real quantum runs. Every other variant in that theme gets its quantum VaR
by scaling that pair's representative quantum VaR by the ratio of the
variant's vol_multiplier to the representative's. Classical Monte Carlo VaR
is cheap enough to compute for real on every (pair, variant) combo — no
scaling needed there. This is disclosed openly in the dashboard's methodology
section, not hidden.
"""

from __future__ import annotations

THEMES: dict[str, dict] = {
    "brent_crude": {
        "name": "Brent Crude Oil",
        "variants": [
            {"severity_label": "Crude -30%", "severity_value": -30, "direction": "down",
             "vol_multiplier": 2.5, "mean_shift": -0.020, "distribution_type": "fat_tail",
             "narrative": "Demand collapse signal. Severe global slowdown implied. INR benefits marginally from lower import costs but EM risk-off dominates."},
            {"severity_label": "Crude -20%", "severity_value": -20, "direction": "down",
             "vol_multiplier": 2.0, "mean_shift": -0.014, "distribution_type": "fat_tail",
             "narrative": "Sharp demand-side crude drop. Signals a meaningful global slowdown; INR gets a small import-cost tailwind, outweighed by broader EM risk-off."},
            {"severity_label": "Crude -10%", "severity_value": -10, "direction": "down",
             "vol_multiplier": 1.5, "mean_shift": -0.007, "distribution_type": "skewed",
             "narrative": "Mild crude pullback on softer demand expectations. Limited direct impact on GIFT City books."},
            {"severity_label": "Crude +10%", "severity_value": 10, "direction": "up",
             "vol_multiplier": 1.5, "mean_shift": -0.006, "distribution_type": "skewed",
             "narrative": "Moderate crude rise increases India's import bill. Current account deficit widens slightly."},
            {"severity_label": "Crude +20%", "severity_value": 20, "direction": "up",
             "vol_multiplier": 2.0, "mean_shift": -0.012, "distribution_type": "fat_tail",
             "narrative": "Sizeable crude spike. Import bill pressure builds materially; INR depreciation pressure intensifies."},
            {"severity_label": "Crude +30%", "severity_value": 30, "direction": "up",
             "vol_multiplier": 2.5, "mean_shift": -0.018, "distribution_type": "fat_tail",
             "narrative": "Severe crude spike. India import costs surge. GIFT City trade finance desks face elevated settlement risk."},
        ],
        "representative": "Crude -20%",
    },
    "fed_rate": {
        "name": "Fed Rate Move",
        "variants": [
            {"severity_label": "Fed -50bps", "severity_value": -50, "direction": "cut",
             "vol_multiplier": 1.4, "mean_shift": 0.006, "distribution_type": "skewed",
             "narrative": "Emergency Fed cut. Risk-on, EM inflows, INR strengthens."},
            {"severity_label": "Fed -25bps", "severity_value": -25, "direction": "cut",
             "vol_multiplier": 1.2, "mean_shift": 0.003, "distribution_type": "normal",
             "narrative": "Routine Fed cut, broadly expected. Mild risk-on tone, modest INR support."},
            {"severity_label": "Fed +25bps", "severity_value": 25, "direction": "hike",
             "vol_multiplier": 1.3, "mean_shift": -0.004, "distribution_type": "skewed",
             "narrative": "Routine Fed hike. Dollar firms modestly; limited direct stress on GIFT City books."},
            {"severity_label": "Fed +50bps", "severity_value": 50, "direction": "hike",
             "vol_multiplier": 1.6, "mean_shift": -0.008, "distribution_type": "skewed",
             "narrative": "Fed tightens 50bps. Dollar rallies. Capital outflows from GIFT City IBUs toward USD assets accelerate."},
            {"severity_label": "Fed +75bps", "severity_value": 75, "direction": "hike",
             "vol_multiplier": 2.0, "mean_shift": -0.013, "distribution_type": "fat_tail",
             "narrative": "Unusually large hike. Markets reassess the tightening path; EM currencies come under sustained pressure."},
            {"severity_label": "Fed +100bps", "severity_value": 100, "direction": "hike",
             "vol_multiplier": 2.4, "mean_shift": -0.018, "distribution_type": "fat_tail",
             "narrative": "Aggressive 100bps hike shocks markets. EM currencies in freefall. GIFT City positions face severe losses."},
        ],
        "representative": "Fed +50bps",
    },
    "usdinr_spot": {
        "name": "USD/INR Spot Move",
        "variants": [
            {"severity_label": "INR +3%", "severity_value": 3, "direction": "appreciation",
             "vol_multiplier": 1.3, "mean_shift": 0.008, "distribution_type": "normal",
             "narrative": "Sharp INR appreciation. Unhedged receivable books gain; payable books see a favourable mark-to-market."},
            {"severity_label": "INR +2%", "severity_value": 2, "direction": "appreciation",
             "vol_multiplier": 1.2, "mean_shift": 0.005, "distribution_type": "normal",
             "narrative": "Meaningful INR strength on portfolio inflows. Hedge ratios drift as unhedged legs gain value."},
            {"severity_label": "INR +1%", "severity_value": 1, "direction": "appreciation",
             "vol_multiplier": 1.1, "mean_shift": 0.002, "distribution_type": "normal",
             "narrative": "Modest INR strength within normal daily range."},
            {"severity_label": "INR -1%", "severity_value": -1, "direction": "depreciation",
             "vol_multiplier": 1.1, "mean_shift": -0.002, "distribution_type": "normal",
             "narrative": "Modest INR weakness within normal daily range."},
            {"severity_label": "INR -2%", "severity_value": -2, "direction": "depreciation",
             "vol_multiplier": 1.2, "mean_shift": -0.005, "distribution_type": "skewed",
             "narrative": "Meaningful INR depreciation. Unhedged payable books see a direct mark-to-market hit."},
            {"severity_label": "INR -3%", "severity_value": -3, "direction": "depreciation",
             "vol_multiplier": 1.3, "mean_shift": -0.008, "distribution_type": "skewed",
             "narrative": "Sharp INR depreciation. Unhedged GIFT City positions take direct mark-to-market losses. Hedge ratio drift accelerates."},
        ],
        "representative": "INR -2%",
    },
    "sovereign_spread": {
        "name": "India Sovereign Spread",
        "variants": [
            {"severity_label": "Spread +50bps", "severity_value": 50, "direction": "widen",
             "vol_multiplier": 1.3, "mean_shift": -0.004, "distribution_type": "skewed",
             "narrative": "Modest spread widening. Early signal of funding-cost pressure, contained for now."},
            {"severity_label": "Spread +100bps", "severity_value": 100, "direction": "widen",
             "vol_multiplier": 1.7, "mean_shift": -0.009, "distribution_type": "fat_tail",
             "narrative": "Spread widens 100bps. Capital flight risk rises. IBU funding costs increase."},
            {"severity_label": "Spread +150bps", "severity_value": 150, "direction": "widen",
             "vol_multiplier": 2.1, "mean_shift": -0.014, "distribution_type": "fat_tail",
             "narrative": "Sharp spread widening. Sustained capital outflow pressure; funding costs rise materially across GIFT City IBUs."},
            {"severity_label": "Spread +200bps", "severity_value": 200, "direction": "widen",
             "vol_multiplier": 2.5, "mean_shift": -0.019, "distribution_type": "fat_tail",
             "narrative": "Severe sovereign stress. Near-crisis conditions. GIFT City IBU liquidity under acute pressure."},
        ],
        "representative": "Spread +100bps",
    },
    "global_equity": {
        "name": "Global Equity (SPY)",
        "variants": [
            {"severity_label": "SPY +10%", "severity_value": 10, "direction": "up",
             "vol_multiplier": 1.2, "mean_shift": 0.004, "distribution_type": "normal",
             "narrative": "Strong risk-on rally. EM flows improve broadly, modest INR support."},
            {"severity_label": "SPY +5%", "severity_value": 5, "direction": "up",
             "vol_multiplier": 1.1, "mean_shift": 0.002, "distribution_type": "normal",
             "narrative": "Steady risk-on tone. Limited direct effect on GIFT City books."},
            {"severity_label": "SPY -5%", "severity_value": -5, "direction": "down",
             "vol_multiplier": 1.3, "mean_shift": -0.003, "distribution_type": "skewed",
             "narrative": "Ordinary risk-off pullback. Some EM FX softness, contained."},
            {"severity_label": "SPY -10%", "severity_value": -10, "direction": "down",
             "vol_multiplier": 1.6, "mean_shift": -0.007, "distribution_type": "fat_tail",
             "narrative": "Sharp correction. Correlated selling spreads into EM FX; GIFT City books see a moderate mark-to-market hit."},
            {"severity_label": "SPY -20%", "severity_value": -20, "direction": "down",
             "vol_multiplier": 2.3, "mean_shift": -0.016, "distribution_type": "fat_tail",
             "narrative": "S&P drops 20%. Risk assets sell off together. Correlations spike to 1. GIFT City positions face simultaneous losses."},
            {"severity_label": "SPY -30%", "severity_value": -30, "direction": "down",
             "vol_multiplier": 2.8, "mean_shift": -0.022, "distribution_type": "fat_tail",
             "narrative": "Equity crash. Extreme risk-off. Diversification fails."},
        ],
        "representative": "SPY -10%",
    },
    "crypto_contagion": {
        "name": "Crypto Contagion (BTC)",
        "variants": [
            {"severity_label": "BTC -20%", "severity_value": -20, "direction": "down",
             "vol_multiplier": 1.3, "mean_shift": -0.004, "distribution_type": "skewed",
             "narrative": "Ordinary crypto drawdown. Minimal spillover into traditional FX."},
            {"severity_label": "BTC -35%", "severity_value": -35, "direction": "down",
             "vol_multiplier": 1.5, "mean_shift": -0.007, "distribution_type": "fat_tail",
             "narrative": "Sizeable crypto deleveraging. Early signs of USD liquidity tightening at the margin."},
            {"severity_label": "BTC -50%", "severity_value": -50, "direction": "down",
             "vol_multiplier": 1.8, "mean_shift": -0.011, "distribution_type": "fat_tail",
             "narrative": "BTC drops 50%. Stablecoin redemptions tighten USD liquidity globally. INR under pressure."},
            {"severity_label": "BTC -70%", "severity_value": -70, "direction": "down",
             "vol_multiplier": 2.2, "mean_shift": -0.016, "distribution_type": "fat_tail",
             "narrative": "Systemic crypto deleveraging. USD liquidity crunch spreads to EM FX."},
        ],
        "representative": "BTC -35%",
    },
    "gold_safety": {
        "name": "Gold (Flight-to-Safety)",
        "variants": [
            {"severity_label": "Gold +5%", "severity_value": 5, "direction": "up",
             "vol_multiplier": 1.2, "mean_shift": -0.002, "distribution_type": "normal",
             "narrative": "Mild safe-haven bid. Ordinary risk-management flows, no particular stress signal."},
            {"severity_label": "Gold +10%", "severity_value": 10, "direction": "up",
             "vol_multiplier": 1.4, "mean_shift": -0.005, "distribution_type": "skewed",
             "narrative": "Notable flight-to-safety bid in gold. A moderate stress signal; some EM outflow pressure builds."},
            {"severity_label": "Gold +20%", "severity_value": 20, "direction": "up",
             "vol_multiplier": 1.7, "mean_shift": -0.010, "distribution_type": "fat_tail",
             "narrative": "Strong flight to safety. Serious market stress signal. EM positions see material outflows."},
        ],
        "representative": "Gold +10%",
    },
    "china_pmi": {
        "name": "China PMI Miss",
        "variants": [
            {"severity_label": "China PMI -1pt", "severity_value": -1, "direction": "miss",
             "vol_multiplier": 1.2, "mean_shift": -0.003, "distribution_type": "normal",
             "narrative": "Small PMI miss. Limited spillover into Asia EM FX."},
            {"severity_label": "China PMI -2pt", "severity_value": -2, "direction": "miss",
             "vol_multiplier": 1.5, "mean_shift": -0.007, "distribution_type": "skewed",
             "narrative": "Meaningful PMI miss. Early Asia EM contagion via trade channels."},
            {"severity_label": "China PMI -3pt", "severity_value": -3, "direction": "miss",
             "vol_multiplier": 1.8, "mean_shift": -0.011, "distribution_type": "fat_tail",
             "narrative": "Asia EM sell-off accelerates. INR, SGD, AED all under pressure via trade and capital contagion."},
            {"severity_label": "China PMI -5pt", "severity_value": -5, "direction": "miss",
             "vol_multiplier": 2.2, "mean_shift": -0.016, "distribution_type": "fat_tail",
             "narrative": "Severe slowdown signal. Asia-wide contagion."},
        ],
        "representative": "China PMI -3pt",
    },
    "gift_liquidity": {
        "name": "GIFT City Liquidity",
        "variants": [
            {"severity_label": "Mild", "severity_value": 1, "direction": "mild",
             "vol_multiplier": 1.5, "mean_shift": -0.006, "distribution_type": "skewed",
             "narrative": "Bid-ask spreads widen modestly. Settlement delays of 1-2 hours."},
            {"severity_label": "Moderate", "severity_value": 2, "direction": "moderate",
             "vol_multiplier": 2.0, "mean_shift": -0.012, "distribution_type": "fat_tail",
             "narrative": "Multiple IBUs drawing simultaneously on liquidity."},
            {"severity_label": "Severe", "severity_value": 3, "direction": "severe",
             "vol_multiplier": 2.5, "mean_shift": -0.018, "distribution_type": "bimodal",
             "narrative": "Settlement bottlenecks across IBUs. Spreads blow out."},
            {"severity_label": "Extreme", "severity_value": 4, "direction": "extreme",
             "vol_multiplier": 3.0, "mean_shift": -0.025, "distribution_type": "bimodal",
             "narrative": "Systemic stress. Full settlement paralysis. Unique to GIFT City's concentrated institutional structure."},
        ],
        "representative": "Severe",
    },
}


def variant_key(theme_key: str, severity_label: str) -> str:
    return f"{theme_key}::{severity_label}"


def all_variants():
    """Yield (theme_key, theme_name, variant_dict, is_representative) for every
    variant across all 9 themes, in ladder order."""
    for theme_key, theme in THEMES.items():
        rep_label = theme["representative"]
        for variant in theme["variants"]:
            yield theme_key, theme["name"], variant, (variant["severity_label"] == rep_label)


def representative_variant(theme_key: str) -> dict:
    theme = THEMES[theme_key]
    rep_label = theme["representative"]
    for v in theme["variants"]:
        if v["severity_label"] == rep_label:
            return v
    raise KeyError(f"no representative variant found for {theme_key}")


def variant_count() -> int:
    return sum(len(t["variants"]) for t in THEMES.values())
