"""GIFT Risk — FX Treasury Dashboard.

A morning dashboard for a GIFT IFSC IBU treasury officer: open it and see,
immediately, how the FX portfolio moves under a ladder of macro shocks, and
what trades are needed today to hold target hedge ratios. Quantum
computation (Qiskit Aer IQAE) feeds the risk numbers in the background — it
is supporting infrastructure here, not the primary interface; there is no
manual scenario selector, everything renders on load.

Run:  streamlit run app.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.aws_services import (
    generate_portfolio_briefing,
    generate_risk_commentary,
    load_metadata,
)
from src.benchmark import plot_benchmark
from src.portfolio import PORTFOLIO_DEFAULTS, DEFAULT_DRIFT_THRESHOLD, shock_impact
from src.shocks import THEMES

DATA_DIR = Path(__file__).resolve().parent / "data"
PAIRS = ["USD/INR", "SGD/INR", "AED/INR"]

st.set_page_config(page_title="GIFT Risk", page_icon="🏦", layout="wide")

# ------------------------------------------------------------------ header
st.title("GIFT Risk — FX Treasury Dashboard")
st.caption("GIFT IFSC | Quantum-Accelerated Tail Risk")
st.warning(
    "Synthetic data. Quantum circuit on simulator. Not financial advice.",
    icon="⚠️",
)
st.caption(f"Last updated: {datetime.datetime.now():%A, %d %b %Y — %H:%M} (session load)")


# ------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def _load_shock_results() -> pd.DataFrame:
    raw = json.loads((DATA_DIR / "shock_results.json").read_text())
    return raw["baseline_var"], pd.DataFrame(raw["rows"])


@st.cache_data(show_spinner=False)
def _load_meta():
    return load_metadata()


@st.cache_data(show_spinner=False)
def _severity_dots(theme_key: str) -> dict:
    """Map each variant's severity_label to a colour dot, ranked by
    vol_multiplier WITHIN the theme (vol is U-shaped across many themes'
    index order — e.g. Brent runs extreme-negative...mild...extreme-positive
    — so ranking by list position would be wrong; rank by actual magnitude)."""
    variants = THEMES[theme_key]["variants"]
    vols = sorted(v["vol_multiplier"] for v in variants)
    dots = {}
    for v in variants:
        pct = vols.index(v["vol_multiplier"]) / max(len(vols) - 1, 1)
        if pct < 0.25:
            dots[v["severity_label"]] = "🟢"
        elif pct < 0.5:
            dots[v["severity_label"]] = "🟡"
        elif pct < 0.75:
            dots[v["severity_label"]] = "🟠"
        else:
            dots[v["severity_label"]] = "🔴"
    return dots


baseline_var, shock_df = _load_shock_results()
meta_all, meta_source = _load_meta()

# ------------------------------------------------------- portfolio state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        pair: dict(cfg) for pair, cfg in PORTFOLIO_DEFAULTS.items()
    }

# ============================================================ SECTION 1
st.subheader("Portfolio Positions")
cols = st.columns(3)
for col, pair in zip(cols, PAIRS):
    meta = meta_all[pair]
    cfg = st.session_state.portfolio[pair]
    with col:
        st.markdown(f"**{pair}** — {meta['entity_name']}")
        st.caption(meta["desk_name"])
        notional = st.number_input(
            "Notional (USD)", min_value=100_000, value=cfg["notional"],
            step=500_000, format="%d", key=f"notional_{pair}",
        )
        target = st.slider(
            "Target hedge ratio", 0, 100, int(cfg["target_hedge_ratio"] * 100),
            step=5, key=f"target_{pair}",
        )
        st.session_state.portfolio[pair] = {
            "notional": notional, "target_hedge_ratio": target / 100.0,
        }
        # status vs. the ladder's worst-case drift for this pair, at current settings
        worst_drift = 0.0
        for _, row in shock_df[shock_df["pair"] == pair].iterrows():
            imp = shock_impact(
                row["quantum_var_pct"], row["vol_multiplier"], target / 100.0,
                notional, meta["exposure_type"], pair,
            )
            worst_drift = max(worst_drift, abs(imp.drift))
        if worst_drift <= DEFAULT_DRIFT_THRESHOLD:
            st.success(f"On target across the shock ladder (max drift {worst_drift:.0%})")
        elif worst_drift <= 0.15:
            st.warning(f"Drifts up to {worst_drift:.0%} under stress")
        else:
            st.error(f"Drifts up to {worst_drift:.0%} under stress — review")

st.divider()

# ---------------------------------------------------- recompute all impacts
def compute_impacts(portfolio: dict) -> pd.DataFrame:
    """Pure arithmetic over the precomputed VaR table — recomputed live on
    every rerun (fast: 129 rows) so notional/target edits reflect instantly
    without touching the quantum-computed cache."""
    records = []
    for _, row in shock_df.iterrows():
        pair = row["pair"]
        meta = meta_all[pair]
        cfg = portfolio[pair]
        imp = shock_impact(
            row["quantum_var_pct"], row["vol_multiplier"], cfg["target_hedge_ratio"],
            cfg["notional"], meta["exposure_type"], pair,
        )
        records.append({
            **row.to_dict(),
            "notional": cfg["notional"],
            "target_hedge_ratio": cfg["target_hedge_ratio"],
            "pnl_usd": imp.pnl_usd,
            "pnl_pct": imp.pnl_pct,
            "required_hedge_ratio": imp.required_hedge_ratio,
            "drift": imp.drift,
            "action_required": imp.action_required,
            "trade_text": imp.trade_text,
        })
    return pd.DataFrame(records)


impacts = compute_impacts(st.session_state.portfolio)

# ============================================================ SECTION 2
st.subheader("Progressive Shock Heatmap")
st.caption(
    "Rows: macro themes, mild → extreme. Columns: currency pair. "
    "Cell: P&L impact ($K / % of notional / hedge-ratio drift Δ). "
    "Click a cell for full detail."
)

flat_rows = []
for theme_key, theme in THEMES.items():
    dots = _severity_dots(theme_key)
    for variant in theme["variants"]:
        flat_rows.append((theme_key, theme["name"], variant["severity_label"], dots[variant["severity_label"]]))

row_labels = [f"{name} · {sev}" for _, name, sev, _ in flat_rows]
severity_dots_col = [dot for *_, dot in flat_rows]

display = pd.DataFrame(index=row_labels)
display.insert(0, "⚑", severity_dots_col)
colors = pd.DataFrame("", index=row_labels, columns=["⚑"] + PAIRS)

for i, (theme_key, theme_name, sev, dot) in enumerate(flat_rows):
    for pair in PAIRS:
        sub = impacts[
            (impacts["theme_key"] == theme_key)
            & (impacts["severity_label"] == sev)
            & (impacts["pair"] == pair)
        ]
        r = sub.iloc[0]
        marker = "▸ " if r["action_required"] else ""
        display.loc[row_labels[i], pair] = (
            f"{marker}{r['pnl_usd']/1000:+.0f}K "
            f"({r['pnl_pct']:+.1%}) Δ{r['drift']:+.0%}"
        )
        # diverging red (loss) -> white -> green (gain), clipped at +-3%
        t = max(-1.0, min(1.0, r["pnl_pct"] / 0.03))
        if t < 0:
            red, green, blue = 255, int(255 + t * 90), int(255 + t * 90)
        else:
            red, green, blue = int(255 - t * 90), 255, int(255 - t * 90)
        bg = f"rgb({red},{green},{blue})"
        border = "2px solid #a23b2c" if r["action_required"] else "1px solid #ddd"
        colors.loc[row_labels[i], pair] = f"background-color:{bg}; border:{border};"

styled = display.style.apply(lambda _: colors, axis=None)
event = st.dataframe(
    styled,
    height=(len(row_labels) + 1) * 36,
    on_select="rerun",
    selection_mode="single-cell",
    key="heatmap",
)

st.caption("▸ marks a cell where the hedge-ratio drift exceeds the action threshold (±5 points).")

selected_detail = None
cells = event.selection.get("cells", []) if hasattr(event, "selection") else []
if cells:
    row_idx, col_name = cells[0]
    if col_name in PAIRS:
        theme_key, theme_name, sev, _ = flat_rows[row_idx]
        selected_detail = (theme_key, theme_name, sev, col_name)

if selected_detail:
    theme_key, theme_name, sev, pair = selected_detail
    r = impacts[
        (impacts["theme_key"] == theme_key)
        & (impacts["severity_label"] == sev)
        & (impacts["pair"] == pair)
    ].iloc[0]
    with st.expander(f"Detail — {theme_name} · {sev} · {pair}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Classical VaR", f"{r['classical_var_pct']:.3%}")
        c2.metric("Quantum VaR", f"{r['quantum_var_pct']:.3%}",
                   "measured" if r["quantum_source"] == "measured" else "scaled from theme baseline")
        c3.metric("Hedge ratio: target → required", f"{r['target_hedge_ratio']:.0%} → {r['required_hedge_ratio']:.0%}")
        c4.metric("P&L impact", f"${r['pnl_usd']:,.0f}", f"{r['pnl_pct']:+.2%} of notional", delta_color="inverse")
        if r["trade_text"]:
            st.error(f"**ACTION REQUIRED**\n\n{r['trade_text']}\n\nTrigger: {theme_name} — {sev}")
        else:
            st.success("No trade required — within drift tolerance.")

        @st.cache_data(show_spinner="Drafting commentary…")
        def _cell_commentary(theme_key, sev, pair, var_pct, narrative, name):
            meta = meta_all[pair]
            text, source = generate_risk_commentary(
                meta["entity_name"], pair, f"{name} — {sev}", narrative,
                var_pct, f"${var_pct * r['notional']:,.0f}", True, 1.0,
            )
            return text, source

        commentary, comm_source = _cell_commentary(
            theme_key, sev, pair, r["quantum_var_pct"], r["narrative"], theme_name
        )
        label = "Bedrock · Claude Opus 5" if comm_source == "bedrock" else "Offline (Bedrock unavailable)"
        st.info(f"**{label}** — not financial advice\n\n{commentary}")

st.divider()

# ============================================================ SECTION 3
col_blotter, col_stress = st.columns([2, 1])

agg = (
    impacts.groupby(["theme_key", "theme_name", "severity_label"])
    .agg(total_pnl=("pnl_usd", "sum"))
    .reset_index()
)
worst = agg.loc[agg["total_pnl"].idxmin()]

with col_stress:
    st.subheader("Most Stressed Scenario")
    st.error(
        f"**{worst['theme_name']} — {worst['severity_label']}**\n\n"
        f"Aggregate portfolio impact: ${worst['total_pnl']:,.0f}"
    )
    exposure_by_pair = (
        impacts.groupby("pair")["pnl_usd"].sum().sort_values()
    )
    st.caption("Cumulative P&L across the full ladder, by pair:")
    for pair, val in exposure_by_pair.items():
        st.caption(f"{pair}: ${val:,.0f}")

with col_blotter:
    st.subheader("Trade Blotter — Action Required")
    blotter = impacts[impacts["action_required"]].copy()
    blotter["abs_drift"] = blotter["drift"].abs()
    blotter = blotter.sort_values("abs_drift", ascending=False)
    if blotter.empty:
        st.success("No trades required across the current shock ladder.")
    else:
        for _, r in blotter.head(15).iterrows():
            action = r["trade_text"].split()[0]
            icon = "🔴" if action == "SELL" else "🔵"
            st.markdown(
                f"{icon} **{r['trade_text'].splitlines()[0]}**  \n"
                f"{r['trade_text'].splitlines()[1]}  \n"
                f"*Trigger: {r['theme_name']} — {r['severity_label']} · "
                f"drift {r['drift']:+.0%}*"
            )
        if len(blotter) > 15:
            st.caption(f"...and {len(blotter) - 15} more action-required trades.")

st.divider()

# ============================================================ SECTION 4
st.subheader("Portfolio Stress Summary")
agg_sorted = agg.sort_values("total_pnl")
fig, ax = plt.subplots(figsize=(10, max(4, len(agg_sorted) * 0.16)))
labels = [f"{t} · {s}" for t, s in zip(agg_sorted["theme_name"], agg_sorted["severity_label"])]
colors_bar = ["#A23B2C" if v < 0 else "#45633A" for v in agg_sorted["total_pnl"]]
ax.barh(labels, agg_sorted["total_pnl"], color=colors_bar)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Total portfolio P&L impact (USD)")
ax.set_title("All 43 scenario variants, aggregated across the 3-pair portfolio")
fig.tight_layout()
st.pyplot(fig)

st.divider()

# ============================================================ SECTION 5
st.subheader("AI Morning Risk Briefing")
if "morning_briefing" not in st.session_state:
    top_trade_row = blotter.iloc[0] if not blotter.empty else None
    top_trade_text = (
        top_trade_row["trade_text"].splitlines()[0] if top_trade_row is not None
        else "No trades required today."
    )
    exposed_pair = exposure_by_pair.index[0]
    with st.spinner("Drafting morning briefing…"):
        text, source = generate_portfolio_briefing(
            f"{worst['theme_name']} — {worst['severity_label']}",
            f"${worst['total_pnl']:,.0f}",
            exposed_pair,
            top_trade_text,
        )
    st.session_state.morning_briefing = (text, source)

briefing_text, briefing_source = st.session_state.morning_briefing
label = (
    "AI Morning Risk Briefing — Claude Opus 5 via Amazon Bedrock. Not financial advice."
    if briefing_source == "bedrock"
    else "AI Morning Risk Briefing — offline template (Bedrock unavailable). Not financial advice."
)
st.info(f"**{label}**\n\n{briefing_text}")

st.divider()

# ============================================================ SECTION 6
with st.expander("Quantum Methodology", expanded=False):
    st.markdown(
        "**Representative-variant scaling.** Running IQAE on every one of "
        "the 43 severity variants × 3 pairs (129 combinations) is not worth "
        "doing on a simulator. Instead: IQAE is computed for real on ONE "
        "representative (middle-severity) variant per theme, per currency "
        "pair — 9 themes × 3 pairs = **27 real quantum runs**. Every other "
        "variant's quantum VaR is that pair's representative quantum VaR, "
        "scaled by the ratio of vol_multipliers. This is disclosed here "
        "openly, not hidden — the classical VaR shown in each cell's detail "
        "is computed independently and for real on all 129 combinations, so "
        "you can see where the scaling approximation and the real classical "
        "estimate diverge."
    )
    st.markdown(
        "**No wall-clock speed claim.** Everything runs on a classical "
        "simulator (Qiskit Aer). The result demonstrated is quantum "
        "amplitude estimation's query-complexity advantage — O(1/ε) vs. "
        "classical Monte Carlo's O(1/ε²) — never a claim that quantum "
        "hardware ran anything faster in real time."
    )
    st.markdown("**Backend:** Qiskit Aer (IQAE) for all 27 representative runs.")
    st.caption(
        "Baseline (unstressed) classical VaR, for reference: "
        + " · ".join(f"{p}: {v:.3%}" for p, v in baseline_var.items())
    )

    st.markdown("**Classical vs. quantum: samples needed to reach a target precision**")
    bench_path = DATA_DIR / "benchmark_results.csv"
    if bench_path.exists():
        bench_df = pd.read_csv(bench_path)
        st.pyplot(plot_benchmark(bench_df))
    else:
        st.caption("Benchmark data not found — run `python -m src.benchmark` to regenerate.")

st.caption(
    "GIFT Risk is a hackathon prototype (Track 2 — Quantum Tech in Financial Services). "
    "Query-complexity comparison only; no wall-clock speed claim."
)
