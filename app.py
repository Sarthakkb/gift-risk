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
import numpy as np
import pandas as pd
import streamlit as st

from src.aws_services import (
    generate_portfolio_briefing,
    generate_risk_commentary,
    load_metadata,
    load_position_csv,
)
from src.benchmark import plot_benchmark
import src.fcnr as fcnr
from src.market_data import PAIR_PARAMS
from src.portfolio import PORTFOLIO_DEFAULTS, DEFAULT_DRIFT_THRESHOLD, shock_impact
from src.shocks import THEMES

DATA_DIR = Path(__file__).resolve().parent / "data"
PAIRS = ["USD/INR", "SGD/INR", "AED/INR"]


def md_safe(text: str) -> str:
    """Escape literal $ before handing dynamic text (LLM output, f-strings
    with dollar amounts) to a markdown-rendering call. Streamlit's markdown
    treats a $...$ pair as inline LaTeX math — with 2+ dollar amounts in one
    block (very common in both our own f-strings and Bedrock's prose, e.g.
    "roughly $194k... $105M of funding"), everything between the first and
    second $ silently renders as a garbled math expression instead of text."""
    return text.replace("$", "\\$")

st.set_page_config(page_title="GIFT Risk", page_icon="🏦", layout="wide")

# ------------------------------------------------------------------ header
st.title("GIFT Risk — FX Treasury Dashboard")
st.caption("GIFT IFSC | Quantum-Accelerated Tail Risk")
st.warning(
    "Synthetic data. Quantum circuit on simulator. Not financial advice.",
    icon="⚠️",
)
st.caption(f"Last updated: {datetime.datetime.now():%A, %d %b %Y — %H:%M} (session load)")


tab_dashboard, tab_hdfc = st.tabs(["Portfolio Dashboard", "HDFC Case Study — Live Demo"])

with tab_dashboard:

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
    if "fcnr" not in st.session_state:
        st.session_state.fcnr = dict(fcnr.FCNR_DEFAULTS)


    @st.cache_data(show_spinner=False)
    def _rate_ticker() -> pd.DataFrame:
        """FX spot & forward rates — realistic mock values, generated once.
        Forward points reflect a plausible rate differential (INR carries a
        higher rate than USD/SGD/AED, so INR forwards trade at a premium),
        reusing the pipeline's own synthetic spot levels (src/market_data.py)
        rather than inventing a separate set of numbers."""
        diffs = {"USD/INR": 0.020, "SGD/INR": 0.032, "AED/INR": 0.020}
        daily_change = {"USD/INR": 0.0007, "SGD/INR": -0.0004, "AED/INR": 0.0005}
        rows = []
        for pair in PAIRS:
            spot = PAIR_PARAMS[pair]["spot"]
            d = diffs[pair]
            rows.append({
                "Pair": pair,
                "Spot": spot,
                "1M Forward": spot * (1 + d / 12),
                "3M Forward": spot * (1 + d / 4),
                "Daily Change": daily_change[pair],
            })
        return pd.DataFrame(rows)


    st.subheader("FX Spot & Forward Rates")
    ticker_df = _rate_ticker()
    st.dataframe(
        ticker_df.style.format({
            "Spot": "{:.4f}", "1M Forward": "{:.4f}", "3M Forward": "{:.4f}",
            "Daily Change": "{:+.2%}",
        }),
        hide_index=True, width=560,
    )
    st.caption("Illustrative rates — generated once per session, not a live feed.")
    st.divider()

    # ============================================================ SECTION 0
    st.subheader("FCNR(B) Funding Base")
    st.caption(
        f"Real regulatory context: {fcnr.CIRCULAR_REF} opened a USD-INR forex "
        "swap facility for fresh FCNR(B) deposits, absorbing banks' hedging "
        "cost. Inflows far exceeded expectations, so RBI moved the mobilisation "
        "deadline forward to 31 Aug 2026 (from 30 Sep) and the swap-settlement "
        "deadline to 11 Sep 2026. This book and its numbers below are a "
        "synthetic illustration on top of that real backdrop."
    )
    fc1, fc2, fc3 = st.columns([1.4, 1, 1])
    with fc1:
        fcnr_notional = st.number_input(
            "FCNR(B) notional raised (USD)", min_value=1_000_000,
            value=int(st.session_state.fcnr["notional"]), step=10_000_000,
            format="%d", key="fcnr_notional",
        )
        st.caption(
            f"Tenor {st.session_state.fcnr['tenor_years']} years · "
            f"rate paid to depositors {st.session_state.fcnr['rate_paid']:.1%} · "
            f"swap booked {st.session_state.fcnr['swap_booked_date']:%d %b %Y}"
        )
        st.session_state.fcnr["notional"] = fcnr_notional

    status = fcnr.fcnr_status()
    with fc2:
        if status.days_to_mobilisation >= 0:
            st.metric("Days to mobilisation deadline", status.days_to_mobilisation,
                       help="RBI swap window mobilisation deadline: 31 Aug 2026")
        else:
            st.metric("Mobilisation window", "CLOSED", f"{-status.days_to_mobilisation}d ago")
        st.caption(f"Swap settlement deadline: 11 Sep 2026 ({status.days_to_swap_unwind}d)")
    with fc3:
        # baseline retention status (theme=fed_rate, "baseline" isn't in the
        # ladder itself — use the mildest Fed variant as the closest proxy to
        # "ordinary conditions" for a quick-glance status)
        mild_gap = fcnr.funding_gap(fcnr_notional, fcnr.retention_multiplier("fed_rate", "Fed +25bps"))
        mild_gap_pct = mild_gap / fcnr_notional
        if mild_gap_pct < 0.03:
            st.success(f"Retention risk: on track (~${mild_gap/1e6:.0f}M at mild stress)")
        elif mild_gap_pct < 0.08:
            st.warning(f"Retention risk: watch (~${mild_gap/1e6:.0f}M at mild stress)")
        else:
            st.error(f"Retention risk: elevated (~${mild_gap/1e6:.0f}M at mild stress)")
        post_cost = fcnr.post_window_hedging_cost_bps(0)
        st.caption(f"Post-window self-funded hedging cost (illustrative, no shock): ~{post_cost:.0f}bps")

    st.divider()

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
    def compute_impacts(portfolio: dict, fcnr_notional: float) -> pd.DataFrame:
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
            theme_key = row["theme_key"]
            ret_mult = fcnr.retention_multiplier(theme_key, row["severity_label"])
            in_scope = theme_key in fcnr.RETENTION_SCOPE_THEMES
            gap = fcnr.funding_gap(fcnr_notional, ret_mult) if in_scope else None
            hedge_cost = (
                fcnr.post_window_hedging_cost_bps(row["severity_value"])
                if theme_key == "fed_rate" else None
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
                "fcnr_in_scope": in_scope,
                "fcnr_retention": ret_mult if in_scope else None,
                "fcnr_gap": gap,
                "fcnr_hedge_cost_bps": hedge_cost,
            })
        return pd.DataFrame(records)


    impacts = compute_impacts(st.session_state.portfolio, st.session_state.fcnr["notional"])
    agg = (
        impacts.groupby(["theme_key", "theme_name", "severity_label"])
        .agg(total_pnl=("pnl_usd", "sum"))
        .reset_index()
    )
    worst = agg.loc[agg["total_pnl"].idxmin()]
    gap_rows = (
        impacts[(impacts["fcnr_in_scope"]) & (impacts["fcnr_gap"] > 0)]
        .drop_duplicates(subset=["theme_key", "severity_label"])
        .sort_values("fcnr_gap", ascending=False)
    )

    # ============================================================ SECTION 1.5
    st.subheader("VaR Limit Utilization")
    st.caption(
        "Daily 95% VaR against configurable limits — baseline (ordinary "
        "conditions) vs. this book's own worst-case variant in the shock ladder."
    )
    VAR_LIMIT_DEFAULTS = {"USD/INR": 400_000, "SGD/INR": 200_000, "AED/INR": 150_000}
    PORTFOLIO_VAR_LIMIT_DEFAULT = 650_000

    if "var_limits" not in st.session_state:
        st.session_state.var_limits = dict(VAR_LIMIT_DEFAULTS)
    if "portfolio_var_limit" not in st.session_state:
        st.session_state.portfolio_var_limit = PORTFOLIO_VAR_LIMIT_DEFAULT


    def _limit_status(used: float, limit: float) -> tuple[str, str]:
        pct = used / limit if limit > 0 else 0.0
        if pct < 0.70:
            return "🟢", "normal"
        elif pct < 0.90:
            return "🟡", "warning"
        return "🔴", "error"


    limit_cols = st.columns(4)
    var_usage = {}
    for col, pair in zip(limit_cols[:3], PAIRS):
        with col:
            limit = st.number_input(
                f"{pair} VaR limit (USD)", min_value=10_000,
                value=st.session_state.var_limits[pair], step=10_000,
                key=f"varlimit_{pair}",
            )
            st.session_state.var_limits[pair] = limit
            notional = st.session_state.portfolio[pair]["notional"]
            base_usd = baseline_var[pair] * notional
            worst_usd = impacts[impacts["pair"] == pair]["quantum_var_pct"].max() * notional
            icon, level = _limit_status(worst_usd, limit)
            st.progress(min(worst_usd / limit, 1.0), text=f"{icon} worst-case {worst_usd/limit:.0%} of limit")
            st.caption(f"Baseline: \\${base_usd:,.0f} ({base_usd/limit:.0%})  |  Worst-case: \\${worst_usd:,.0f}")

    with limit_cols[3]:
        p_limit = st.number_input(
            "Portfolio VaR limit (USD)", min_value=50_000,
            value=st.session_state.portfolio_var_limit, step=50_000,
            key="varlimit_portfolio",
        )
        st.session_state.portfolio_var_limit = p_limit
        base_total = sum(baseline_var[p] * st.session_state.portfolio[p]["notional"] for p in PAIRS)
        worst_total = -worst["total_pnl"]  # most stressed scenario's aggregate loss
        icon, level = _limit_status(worst_total, p_limit)
        st.progress(min(worst_total / p_limit, 1.0), text=f"{icon} worst-case {worst_total/p_limit:.0%} of limit")
        st.caption(f"Baseline: \\${base_total:,.0f} ({base_total/p_limit:.0%})  |  Worst-case: \\${worst_total:,.0f}")

    st.divider()

    # ============================================================ SECTION 1.6
    st.subheader("Unified Risk Alerts")
    st.caption("Hedge drift flags, VaR limit breaches, and FCNR funding gaps in one feed — most severe and most recent first.")

    _now = datetime.datetime.now()
    alerts = []

    # (a) hedge drift flags from the trade blotter
    _drift_sorted = impacts[impacts["action_required"]].sort_values("drift", key=lambda s: s.abs(), ascending=False)
    for i, (_, r) in enumerate(_drift_sorted.iterrows()):
        sev = "critical" if abs(r["drift"]) > 0.15 else "warning"
        alerts.append({
            "ts": _now - datetime.timedelta(minutes=2 * i),
            "severity": sev,
            "message": f"Hedge drift {r['drift']:+.0%} on {r['pair']} under {r['theme_name']} — {r['severity_label']}",
        })

    # (b) VaR limit breaches (baseline and worst-case, per pair + portfolio)
    for pair in PAIRS:
        notional = st.session_state.portfolio[pair]["notional"]
        limit = st.session_state.var_limits[pair]
        worst_usd = impacts[impacts["pair"] == pair]["quantum_var_pct"].max() * notional
        pct = worst_usd / limit if limit else 0
        if pct > 0.70:
            worst_row = impacts[impacts["pair"] == pair].sort_values("quantum_var_pct", ascending=False).iloc[0]
            alerts.append({
                "ts": _now - datetime.timedelta(minutes=5),
                "severity": "critical" if pct > 0.90 else "warning",
                "message": f"{pair} VaR limit at {pct:.0%} under {worst_row['theme_name']} — {worst_row['severity_label']}",
            })
    p_pct = worst_total / p_limit if p_limit else 0
    if p_pct > 0.70:
        alerts.append({
            "ts": _now - datetime.timedelta(minutes=1),
            "severity": "critical" if p_pct > 0.90 else "warning",
            "message": f"Portfolio VaR limit at {p_pct:.0%} under {worst['theme_name']} — {worst['severity_label']}",
        })

    # (c) FCNR funding gap warnings
    for i, (_, r) in enumerate(gap_rows.iterrows()):
        alerts.append({
            "ts": _now - datetime.timedelta(minutes=10 + 3 * i),
            "severity": "critical" if r["fcnr_gap"] > 50_000_000 else "warning",
            "message": f"FCNR funding gap of ${r['fcnr_gap']/1e6:.0f}M under {r['severity_label']} ({r['theme_name']})",
        })

    _sev_rank = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (_sev_rank[a["severity"]], -a["ts"].timestamp()))

    _icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    with st.container(height=260, border=True):
        if not alerts:
            st.success("No alerts — all limits, hedges, and funding gaps within tolerance.")
        for a in alerts[:40]:
            st.markdown(f"{_icon[a['severity']]} `{a['ts']:%H:%M:%S}` **{a['severity'].upper()}** — {a['message']}")

    st.divider()

    # ============================================================ SECTION 1.7
    st.subheader("Currency Correlation Matrix")
    st.caption("Pearson correlation between the three synthetic daily return series.")


    @st.cache_data(show_spinner=False)
    def _load_return_series():
        series = {}
        for pair in PAIRS:
            df, _ = load_position_csv(pair)
            series[pair] = df["log_return"].to_numpy()
        return pd.DataFrame(series)


    returns_df = _load_return_series()
    corr = returns_df.corr()
    corr_styled = corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1).format("{:.2f}")
    st.dataframe(corr_styled, width=420)

    diag_mask = pd.DataFrame(
        [[i == j for j in range(len(PAIRS))] for i in range(len(PAIRS))],
        index=corr.index, columns=corr.columns,
    )
    off_diag = corr.mask(diag_mask)
    strongest_pair = off_diag.abs().stack().idxmax()
    strongest_val = corr.loc[strongest_pair]
    st.caption(
        f"Highest pairwise correlation: **{strongest_pair[0]} / {strongest_pair[1]}** "
        f"at {strongest_val:+.2f} — "
        + ("high positive correlation suggests limited diversification benefit under simultaneous shocks."
           if strongest_val > 0.3 else
           "correlation is modest, so a shock to one book doesn't strongly imply the same move in another.")
    )
    st.caption(
        "Note: the three return series are generated independently (different "
        "random seeds), so low correlation here reflects the synthetic data "
        "construction, not a real market finding — the shock ladder's shared "
        "vol/mean parameters per theme are what actually correlate the books "
        "under stress, not their baseline daily returns."
    )

    st.divider()

    # ============================================================ SECTION 2
    st.subheader("Progressive Shock Heatmap")
    st.caption(
        "Rows: macro themes, mild → extreme. Columns: currency pair. "
        "Cell: P&L impact ($K / % of notional / hedge-ratio drift Δ). "
        "Click a cell for full detail. The FCNR columns only populate for Fed "
        "Rate Move and India Sovereign Spread — the two themes retention risk "
        "is modeled on (see FCNR(B) Funding Base above); other themes show —."
    )

    flat_rows = []
    for theme_key, theme in THEMES.items():
        dots = _severity_dots(theme_key)
        for variant in theme["variants"]:
            flat_rows.append((theme_key, theme["name"], variant["severity_label"], dots[variant["severity_label"]]))

    row_labels = [f"{name} · {sev}" for _, name, sev, _ in flat_rows]
    severity_dots_col = [dot for *_, dot in flat_rows]

    EXTRA_COLS = ["FCNR Retention", "Funding Gap ($)"]
    all_cols = ["⚑"] + PAIRS + EXTRA_COLS
    display = pd.DataFrame(index=row_labels)
    display.insert(0, "⚑", severity_dots_col)
    for c in PAIRS + EXTRA_COLS:
        display[c] = ""
    colors = pd.DataFrame("", index=row_labels, columns=all_cols)

    for i, (theme_key, theme_name, sev, dot) in enumerate(flat_rows):
        row_sub = impacts[(impacts["theme_key"] == theme_key) & (impacts["severity_label"] == sev)]
        for pair in PAIRS:
            r = row_sub[row_sub["pair"] == pair].iloc[0]
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

        # FCNR columns — only meaningful for fed_rate / sovereign_spread themes
        r0 = row_sub.iloc[0]
        if r0["fcnr_in_scope"]:
            display.loc[row_labels[i], "FCNR Retention"] = f"{r0['fcnr_retention']:.0%}"
            display.loc[row_labels[i], "Funding Gap ($)"] = f"${r0['fcnr_gap']/1e6:+.0f}M"
            gap_bg = "background-color:#F4E1DA;" if r0["fcnr_gap"] > 0 else "background-color:#E1E9DB;"
            colors.loc[row_labels[i], "FCNR Retention"] = gap_bg
            colors.loc[row_labels[i], "Funding Gap ($)"] = gap_bg
        else:
            display.loc[row_labels[i], "FCNR Retention"] = "—"
            display.loc[row_labels[i], "Funding Gap ($)"] = "—"
            colors.loc[row_labels[i], "FCNR Retention"] = "background-color:#EEEEEE; color:#999;"
            colors.loc[row_labels[i], "Funding Gap ($)"] = "background-color:#EEEEEE; color:#999;"

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
            st.info(f"**{label}** — not financial advice\n\n{md_safe(commentary)}")

    st.divider()

    # ============================================================ SECTION 3
    col_blotter, col_stress = st.columns([2, 1])

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

        if blotter.empty and gap_rows.empty:
            st.success("No trades or funding gaps flagged across the current shock ladder.")
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

            if not gap_rows.empty:
                st.markdown("**FCNR(B) funding gap warnings**")
                for _, r in gap_rows.head(6).iterrows():
                    st.markdown(
                        f"🟠 **Funding Gap: ${r['fcnr_gap']/1e6:.0f}M at risk under "
                        f"{r['severity_label']}** ({r['theme_name']})  \n"
                        f"Replace via fresh FCNR at higher self-funded hedge cost"
                        + (f" (~{r['fcnr_hedge_cost_bps']:.0f}bps)" if pd.notna(r['fcnr_hedge_cost_bps']) else "")
                        + ", or alternative wholesale funding."
                    )

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

    # ============================================================ SECTION 4.5
    wc1, wc2 = st.columns(2)

    with wc1:
        st.subheader("P&L Attribution — Most Stressed Scenario")
        st.caption(f"{worst['theme_name']} — {worst['severity_label']}, broken down by currency pair contribution.")
        worst_by_pair = impacts[
            (impacts["theme_key"] == worst["theme_key"])
            & (impacts["severity_label"] == worst["severity_label"])
        ].set_index("pair")["pnl_usd"]

        steps = ["Start"] + PAIRS + ["Total"]
        values = [0.0] + [worst_by_pair[p] for p in PAIRS] + [worst_by_pair.sum()]
        cum = [0.0]
        for v in values[1:-1]:
            cum.append(cum[-1] + v)
        bottoms = [0.0] + cum[:-1] + [0.0]
        heights = [0.0] + values[1:-1] + [values[-1]]
        bar_colors = ["#999"] + ["#A23B2C" if v < 0 else "#45633A" for v in values[1:-1]] + ["#2A4494"]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(steps, heights, bottom=bottoms, color=bar_colors)
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_ylabel("P&L impact (USD)")
        for i, (s, h, b) in enumerate(zip(steps, heights, bottoms)):
            if s not in ("Start",):
                label_y = b + h if s != "Total" else h
                ax2.annotate(f"${h:,.0f}" if s == "Total" else f"${h:+,.0f}", (i, label_y),
                             ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

    with wc2:
        st.subheader("30-Day VaR Trend")
        st.caption("Illustrative historical trend — synthetic data, not live.")

        @st.cache_data(show_spinner=False)
        def _var_trend(center_usd: float) -> pd.DataFrame:
            noise = np.random.default_rng(42).normal(0, center_usd * 0.06, size=30)
            walk = np.cumsum(np.random.default_rng(7).normal(0, center_usd * 0.015, size=30))
            series = center_usd + walk - walk.mean() + noise * 0.3
            return pd.DataFrame({"Day": range(-29, 1), "Portfolio VaR (USD)": series})

        trend_df = _var_trend(base_total)
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(trend_df["Day"], trend_df["Portfolio VaR (USD)"], color="#2A4494", linewidth=1.6)
        ax3.axhline(base_total, color="#999", linestyle="--", linewidth=1, label="Today's baseline VaR")
        ax3.set_xlabel("Days ago")
        ax3.set_ylabel("Portfolio VaR (USD)")
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        st.pyplot(fig3)

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
        if not gap_rows.empty:
            top_gap = gap_rows.iloc[0]
            funding_note = (
                f"RBI's FCNR(B) swap window closes for fresh mobilisation in "
                f"{status.days_to_mobilisation} days (31 Aug 2026); under "
                f"{top_gap['severity_label']} ({top_gap['theme_name']}), an "
                f"estimated ${top_gap['fcnr_gap']/1e6:.0f}M of FCNR(B) funding "
                f"is at retention risk."
            )
        else:
            funding_note = (
                f"FCNR(B) funding base stable across the ladder; RBI's swap "
                f"window closes for fresh mobilisation in {status.days_to_mobilisation} days."
            )
        with st.spinner("Drafting morning briefing…"):
            text, source = generate_portfolio_briefing(
                f"{worst['theme_name']} — {worst['severity_label']}",
                f"${worst['total_pnl']:,.0f}",
                exposed_pair,
                top_trade_text,
                funding_note,
            )
        st.session_state.morning_briefing = (text, source)

    briefing_text, briefing_source = st.session_state.morning_briefing
    label = (
        "AI Morning Risk Briefing — Claude Opus 5 via Amazon Bedrock. Not financial advice."
        if briefing_source == "bedrock"
        else "AI Morning Risk Briefing — offline template (Bedrock unavailable). Not financial advice."
    )
    st.info(f"**{label}**\n\n{md_safe(briefing_text)}")

    st.divider()

    # ============================================================ SECTION 5.5
    st.subheader("Hedge & Funding Maturity Calendar")
    st.caption(
        "Existing FX hedge tranches (mock, spread over the next 90 days) and "
        "the real FCNR(B) regulatory deadlines, in one calendar."
    )

    _today = datetime.date.today()
    calendar_rows = []
    for pair in PAIRS:
        cfg = st.session_state.portfolio[pair]
        hedged_notional = cfg["notional"] * cfg["target_hedge_ratio"]
        for i, days_out in enumerate([30, 60, 90]):
            calendar_rows.append({
                "Position": pair,
                "Instrument": "1-month FX forward" if days_out == 30 else f"{days_out // 30}-month FX forward",
                "Notional": hedged_notional / 3,
                "Maturity / Deadline": _today + datetime.timedelta(days=days_out),
            })
    calendar_rows.append({
        "Position": "FCNR(B) Funding Book", "Instrument": "RBI swap mobilisation deadline",
        "Notional": st.session_state.fcnr["notional"], "Maturity / Deadline": fcnr.MOBILISATION_DEADLINE,
    })
    calendar_rows.append({
        "Position": "FCNR(B) Funding Book", "Instrument": "RBI swap unwind deadline",
        "Notional": st.session_state.fcnr["notional"], "Maturity / Deadline": fcnr.SWAP_UNWIND_DEADLINE,
    })
    cal_df = pd.DataFrame(calendar_rows).sort_values("Maturity / Deadline")
    st.dataframe(
        cal_df.style.format({"Notional": "${:,.0f}"}),
        hide_index=True, width=720,
    )

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
        st.markdown(
            "**FCNR(B) retention scoping.** Retention risk (funding gap) is "
            "modeled on exactly 2 of the 9 shock themes — **Fed Rate Move** and "
            "**India Sovereign Spread** — the two macro drivers that plausibly "
            "move NRI deposit retention. All other 7 themes carry a retention "
            "multiplier of 1.0 (no modeled effect), by design, stated here "
            "openly rather than applying an ungrounded effect everywhere. "
            f"Real regulatory backdrop: {fcnr.CIRCULAR_REF}."
        )
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

    st.divider()

    # ============================================================ SECTION 7
    st.subheader("Export Morning Report")


    def _build_report_text() -> str:
        lines = [
            "GIFT RISK — MORNING REPORT",
            f"{datetime.datetime.now():%A, %d %B %Y — %H:%M}",
            "Synthetic data. Quantum circuit on simulator. Not financial advice.",
            "",
            "PORTFOLIO STATUS",
        ]
        for pair in PAIRS:
            cfg = st.session_state.portfolio[pair]
            worst_drift_p = impacts[impacts["pair"] == pair]["drift"].abs().max()
            status_word = (
                "ON TARGET" if worst_drift_p <= DEFAULT_DRIFT_THRESHOLD
                else "WATCH" if worst_drift_p <= 0.15 else "REVIEW"
            )
            lines.append(
                f"  {pair}: ${cfg['notional']:,.0f} notional, "
                f"{cfg['target_hedge_ratio']:.0%} target hedge — {status_word} "
                f"(max drift {worst_drift_p:+.0%})"
            )
        lines += [
            "",
            "FCNR(B) FUNDING BASE",
            f"  ${st.session_state.fcnr['notional']:,.0f} raised, "
            f"{st.session_state.fcnr['rate_paid']:.1%} paid, "
            f"{st.session_state.fcnr['tenor_years']}yr tenor",
            f"  Mobilisation deadline: 31 Aug 2026 ({status.days_to_mobilisation} days)",
            f"  Swap unwind deadline: 11 Sep 2026 ({status.days_to_swap_unwind} days)",
            "",
            "TOP FLAGGED TRADES / GAPS",
        ]
        top_items = []
        for _, r in blotter.head(3).iterrows():
            top_items.append((r["abs_drift"], f"  TRADE: {r['trade_text'].splitlines()[0]} "
                                                f"— {r['theme_name']} / {r['severity_label']} (drift {r['drift']:+.0%})"))
        for _, r in gap_rows.head(3).iterrows():
            top_items.append((r["fcnr_gap"] / 1e8, f"  FUNDING GAP: ${r['fcnr_gap']/1e6:.0f}M — "
                                                     f"{r['theme_name']} / {r['severity_label']}"))
        top_items.sort(key=lambda t: t[0], reverse=True)
        if top_items:
            lines += [item[1] for item in top_items[:3]]
        else:
            lines.append("  None — all within tolerance.")
        lines += [
            "",
            "AI MORNING RISK BRIEFING",
            f"  {briefing_text}",
        ]
        return "\n".join(lines)


    st.download_button(
        "Download morning report (.txt)",
        data=_build_report_text(),
        file_name=f"GIFT_Risk_Morning_Report_{datetime.date.today():%Y-%m-%d}.txt",
        mime="text/plain",
    )

    st.caption(
        "GIFT Risk is a hackathon prototype (Track 2 — Quantum Tech in Financial Services). "
        "Query-complexity comparison only; no wall-clock speed claim."
    )

# ================================================================= HDFC TAB
with tab_hdfc:
    shock_baseline_var, hdfc_shock_df = _load_shock_results()
    usd_rows = hdfc_shock_df[hdfc_shock_df["pair"] == "USD/INR"].copy()

    st.subheader("HDFC Bank — GIFT City Bond Book")
    st.caption(
        "A live worked example: real, current facts about an actual GIFT City "
        "transaction, run through the same engine as the portfolio dashboard."
    )

    st.info(
        "**What actually happened, 21 Aug 2026:** HDFC Bank, through its GIFT "
        "City IBU, priced **US\\$1.75 billion** in senior unsecured bonds — "
        "**\\$500M at 5.159%, 3-year tenor** and **\\$1.25B at 5.4%, 5-year "
        "tenor**. Both tranches settle **26 Aug 2026** and list on India INX "
        "and NSE IX. Rated **Baa3 (Moody's) / BBB (S&P)**. Proceeds are "
        "earmarked for overseas lending and other banking activities.\n\n"
        "Sources: [Business Standard](https://www.business-standard.com/finance/news/hdfc-bank-raises-1-75-bn-through-overseas-bond-issue-to-fund-business-126082100360_1.html), "
        "[TipRanks](https://www.tipranks.com/news/company-announcements/hdfc-bank-raises-us1-75-billion-via-gift-city-bond-issuance)"
    )

    st.markdown(
        "**Why this book is genuinely interesting for treasury:** the bond "
        "is a USD *liability* — HDFC owes principal and coupon in dollars. "
        "Proceeds funding overseas (USD-denominated) lending gives a natural "
        "partial hedge, so the desk isn't starting from zero cover — but "
        "tenor mismatches between the bond (3yr/5yr fixed maturities) and "
        "the loans it funds, plus the coupon reset/refinancing risk when "
        "each tranche matures, leave a real residual FX and rollover "
        "exposure worth watching on the same shock ladder as every other book."
    )

    st.divider()

    st.markdown("##### How an HDFC treasury officer would open this")
    st.markdown(
        "1. **See the new book flagged.** It shows up in Portfolio Positions "
        "the same morning it settles — no separate system to check.\n"
        "2. **Check today's VaR.** Same quantum-verified shock ladder as the "
        "rest of the desk's books — no new methodology to trust.\n"
        "3. **Check hedge status.** Given the natural USD-asset offset, is "
        "the residual exposure inside policy, or does today's shock push it "
        "past target?\n"
        "4. **Read the AI note**, then **check the refinancing calendar** — "
        "the 3-year tranche isn't due until 2029, but a rates or spread "
        "shock *today* still repriced what refinancing it will cost."
    )

    st.divider()

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("##### Book setup")
        hdfc_notional = st.number_input(
            "Total notional (USD)", min_value=100_000_000, value=1_750_000_000,
            step=50_000_000, format="%d", key="hdfc_notional",
        )
        st.caption(
            "Tranche A: \\$500M · 3yr · 5.159% coupon · "
            "Tranche B: \\$1.25B · 5yr · 5.4% coupon · "
            "Settles 26 Aug 2026 · Rated Baa3 / BBB"
        )
        hdfc_target = st.slider(
            "Target hedge ratio (illustrative — HDFC's actual policy isn't public)",
            0, 100, 85, step=5, key="hdfc_target",
        )
        st.caption(
            "85% assumes a large natural offset from USD-denominated overseas "
            "lending funded by these proceeds, with the residual actively "
            "monitored for timing mismatches."
        )
    with c2:
        st.markdown("##### Refinancing calendar")
        hdfc_cal = pd.DataFrame([
            {"Event": "Bond settlement", "Date": datetime.date(2026, 8, 26), "Amount": "$1.75B"},
            {"Event": "3yr tranche matures", "Date": datetime.date(2029, 8, 26), "Amount": "$500M"},
            {"Event": "5yr tranche matures", "Date": datetime.date(2031, 8, 26), "Amount": "$1.25B"},
        ])
        st.dataframe(hdfc_cal, hide_index=True, width="stretch")

    st.divider()

    st.markdown("##### VaR and hedge status — same shock ladder, new book")
    hdfc_records = []
    for _, row in usd_rows.iterrows():
        imp = shock_impact(
            row["quantum_var_pct"], row["vol_multiplier"], hdfc_target / 100.0,
            hdfc_notional, "payable", "USD/INR",
        )
        hdfc_records.append({**row.to_dict(), "pnl_usd": imp.pnl_usd, "drift": imp.drift,
                               "action_required": imp.action_required, "trade_text": imp.trade_text})
    hdfc_impacts = pd.DataFrame(hdfc_records)
    hdfc_worst = hdfc_impacts.loc[hdfc_impacts["pnl_usd"].idxmin()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline VaR (95%, 1-day)", f"${shock_baseline_var['USD/INR'] * hdfc_notional:,.0f}")
    m2.metric("Worst-case scenario", f"{hdfc_worst['theme_name']} — {hdfc_worst['severity_label']}")
    m3.metric("Worst-case impact", f"${hdfc_worst['pnl_usd']:,.0f}")
    m4.metric("Hedge drift (worst-case)", f"{hdfc_worst['drift']:+.0%}")

    if hdfc_worst["action_required"] and hdfc_worst["trade_text"]:
        st.error(f"**ACTION REQUIRED**\n\n{hdfc_worst['trade_text']}\n\nTrigger: {hdfc_worst['theme_name']} — {hdfc_worst['severity_label']}")
    else:
        st.success("No trade required at current target hedge ratio — worst case stays within drift tolerance.")

    with st.expander("All 43 scenarios for this book"):
        display_cols = hdfc_impacts[["theme_name", "severity_label", "classical_var_pct", "quantum_var_pct", "pnl_usd", "drift", "action_required"]].copy()
        display_cols.columns = ["Theme", "Severity", "Classical VaR", "Quantum VaR", "P&L ($)", "Drift", "Action"]
        st.dataframe(display_cols, hide_index=True, width="stretch")

    st.divider()

    st.markdown("##### AI morning note — HDFC GIFT City book")
    if "hdfc_briefing" not in st.session_state:
        with st.spinner("Drafting note…"):
            hdfc_text, hdfc_source = generate_risk_commentary(
                "HDFC Bank GIFT City IBU",
                "USD/INR",
                f"{hdfc_worst['theme_name']} — {hdfc_worst['severity_label']}",
                "New $1.75B senior unsecured bond issuance (Baa3/BBB), $500M 3yr "
                "tranche at 5.159% and $1.25B 5yr tranche at 5.4%, settling "
                "26 Aug 2026, proceeds for overseas lending. This is a fresh USD "
                "liability being monitored on day one against the standard shock ladder.",
                hdfc_worst["quantum_var_pct"],
                f"${hdfc_worst['pnl_usd']:,.0f}",
                abs(hdfc_worst["quantum_var_pct"] - hdfc_worst["classical_var_pct"]) / hdfc_worst["classical_var_pct"] <= 0.15,
                1.0,
            )
        st.session_state.hdfc_briefing = (hdfc_text, hdfc_source)
    hdfc_text, hdfc_source = st.session_state.hdfc_briefing
    hdfc_label = (
        "Bedrock · Claude Opus 5" if hdfc_source == "bedrock" else "Offline template (Bedrock unavailable)"
    )
    st.info(f"**{hdfc_label}** — not financial advice\n\n{md_safe(hdfc_text)}")

    st.caption(
        "Everything above this line reuses the exact same shock ladder, quantum-verified "
        "VaR, and hedge engine as the Portfolio Dashboard tab — only the position "
        "(notional, exposure type, hedge ratio) is new. The real-world facts (amounts, "
        "coupons, dates, ratings) are sourced and cited above; the hedge ratio and "
        "resulting VaR/trade figures are illustrative, since HDFC's actual internal "
        "hedging policy is not public information."
    )
