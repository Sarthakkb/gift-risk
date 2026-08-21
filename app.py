"""GIFT Risk — Quantum-Accelerated Treasury VaR dashboard.

Primary product interface for a treasury risk officer at a GIFT IFSC IBU:
position -> today's VaR (classical, quantum-verified) -> stress scenarios ->
AI morning-report commentary. The QAE-vs-MC benchmark sits below as
supporting evidence, not as the headline.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.aws_services import generate_risk_commentary, load_metadata, load_position_csv
from src.benchmark import plot_benchmark, run_benchmark
from src.classical_var import monte_carlo_var
from src.quantum_var import discretize, quantum_var
from src.scenarios import SCENARIOS, apply_scenario

AGREEMENT_TOL = 0.15  # relative diff below which classical/quantum "agree"

st.set_page_config(page_title="GIFT Risk", page_icon="⚛️", layout="wide")

# ------------------------------------------------------------------ header
st.title("GIFT Risk — Quantum-Accelerated Treasury VaR")
st.warning(
    "**Disclosure:** All positions and market data are synthetic. The quantum "
    "circuit runs on a classical simulator (Aer / Braket local). This tool "
    "demonstrates quantum algorithm efficiency, not production risk "
    "management. Not financial advice.",
    icon="⚠️",
)


# ------------------------------------------------------------- cached data
@st.cache_data(show_spinner=False)
def _load(pair: str):
    df, source = load_position_csv(pair)
    return df["log_return"].to_numpy(), source


@st.cache_data(show_spinner=False)
def _meta():
    return load_metadata()


@st.cache_data(show_spinner="Running classical + quantum VaR…")
def _analyze(pair: str, scenario_key: str, backend: str):
    returns, data_source = _load(pair)
    stressed = apply_scenario(returns, scenario_key, seed=1)
    mc = monte_carlo_var(stressed, target_error=0.005, seed=2)
    if backend == "Braket local":
        from src.braket_var import braket_quantum_var

        q = braket_quantum_var(stressed)
        q_var, q_queries, q_backend = q.var_estimate, q.oracle_queries, "Braket LocalSimulator (MLAE)"
    else:
        q = quantum_var(stressed, epsilon=0.01, seed=7)
        q_var, q_queries, q_backend = q.var_estimate, q.oracle_queries, "Qiskit Aer (IQAE)"
    return stressed, mc, q_var, q_queries, q_backend, data_source


@st.cache_data(show_spinner="Measuring benchmark sweep (one-time)…")
def _benchmark(pair: str, scenario_key: str):
    returns, _ = _load(pair)
    stressed = apply_scenario(returns, scenario_key, seed=1)
    return run_benchmark(stressed)


@st.cache_data(show_spinner=False)
def _baseline_var(pair: str) -> float:
    returns, _ = _load(pair)
    stressed = apply_scenario(returns, "baseline", seed=1)
    return monte_carlo_var(stressed, target_error=0.005, seed=2).var_estimate


@st.cache_data(show_spinner="Running all 10 scenarios…")
def _scenario_table(pair: str):
    returns, _ = _load(pair)
    rows = []
    for key, sc in SCENARIOS.items():
        stressed = apply_scenario(returns, key, seed=1)
        mc = monte_carlo_var(stressed, target_error=0.005, seed=2)
        q = quantum_var(stressed, epsilon=0.01, seed=7)
        rows.append(
            {
                "Scenario": sc["name"],
                "Classical VaR": f"{mc.var_estimate:.3%}",
                "Quantum VaR": f"{q.var_estimate:.3%}",
                "MC samples": mc.samples_to_converge,
                "QAE queries": q.oracle_queries,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------- left inputs
meta_all, meta_source = _meta()
with st.sidebar:
    st.header("Position")
    pair = st.selectbox("Currency pair", list(meta_all.keys()))
    meta = meta_all[pair]
    st.caption(
        f"**{meta['entity_name']}**  \n{meta['desk_name']}  \n"
        f"Position {meta['position_id']} · valued {meta['valuation_date']}  \n"
        f"Counterparty: {meta['counterparty']}"
    )

    st.header("Scenario")
    scenario_key = st.selectbox(
        "Macro stress scenario",
        list(SCENARIOS.keys()),
        format_func=lambda k: SCENARIOS[k]["name"],
    )
    st.caption(SCENARIOS[scenario_key]["description"])

    notional = st.number_input(
        "Notional (USD)", min_value=100_000, value=10_000_000, step=1_000_000,
        format="%d",
    )
    backend = st.radio("Quantum backend", ["Qiskit Aer", "Braket local"])
    run = st.button("Run Analysis", type="primary", use_container_width=True)

sc = SCENARIOS[scenario_key]

if not run and "ran_once" not in st.session_state:
    st.info("Select a position and scenario, then click **Run Analysis**.")
    st.stop()
st.session_state["ran_once"] = True

# ------------------------------------------------------------ main results
stressed, mc, q_var, q_queries, q_backend, data_source = _analyze(
    pair, scenario_key, backend
)
rel_diff = abs(q_var - mc.var_estimate) / mc.var_estimate
agree = rel_diff <= AGREEMENT_TOL
base_var = _baseline_var(pair)
delta_abs = (mc.var_estimate - base_var) * notional
delta_pct = (mc.var_estimate / base_var - 1) if base_var else 0.0

st.subheader(f"{sc['name']} — {pair}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Classical VaR (95%, 1-day)",
    f"${mc.var_estimate * notional:,.0f}",
    f"{mc.var_estimate:.3%} of notional",
    delta_color="off",
)
c2.metric(
    "Quantum VaR (95%, 1-day)",
    f"${q_var * notional:,.0f}",
    f"{q_var:.3%} of notional",
    delta_color="off",
)
c3.metric(
    "vs. Baseline scenario",
    f"${delta_abs:+,.0f}",
    f"{delta_pct:+.1%}",
    delta_color="inverse",
)
if agree:
    c4.success(f"✓ Agree within tolerance\n\n(diff {rel_diff:.1%} ≤ {AGREEMENT_TOL:.0%})")
else:
    c4.warning(
        f"⚠ Diverging: diff {rel_diff:.1%} > {AGREEMENT_TOL:.0%} — "
        "8-bucket discretisation is too coarse for this tail shape"
    )

st.caption(
    f"Quantum backend: **{q_backend}** · oracle queries: {q_queries:,} · "
    f"MC samples: {mc.samples_to_converge:,} · data source: **{data_source}**"
    + (" (S3)" if data_source == "s3" else " (local fallback)")
)

# ---------------------------------------------------------- AI commentary
st.subheader("Morning risk commentary")
with st.spinner("Generating commentary…"):
    speedup_now = mc.samples_to_converge / max(q_queries, 1)
    commentary, comm_source = generate_risk_commentary(
        meta["entity_name"], pair, sc["name"], sc["narrative"],
        mc.var_estimate, f"${mc.var_estimate * notional:,.0f}",
        agree, max(speedup_now, 1.0),
    )
label = (
    "AI-generated risk commentary (Bedrock · Claude Opus 5) — not financial advice"
    if comm_source == "bedrock"
    else "Offline template commentary (Bedrock unavailable) — not financial advice"
)
st.info(f"**{label}**\n\n{commentary}")

st.divider()

# ------------------------------------------------ supporting evidence panel
st.subheader("Supporting evidence: why quantum")
st.caption(
    "Query complexity of Quantum Amplitude Estimation vs. classical Monte "
    "Carlo — measured counts on this position/scenario. Both run on classical "
    "simulators; the advantage shown is algorithmic sample efficiency "
    "(O(1/ε) vs. O(1/ε²)), never wall-clock speed."
)

bench = _benchmark(pair, scenario_key)
col_a, col_b = st.columns([3, 1])
with col_a:
    st.pyplot(plot_benchmark(bench))
with col_b:
    eps_choice = st.selectbox(
        "Target error ε", bench["epsilon"].tolist(),
        index=len(bench) - 1, format_func=lambda e: f"{e:g}",
    )
    row = bench[bench["epsilon"] == eps_choice].iloc[0]
    st.metric("Speedup at ε", f"{row['speedup']:.1f}×")
    st.caption(
        f"classical: {int(row['classical_samples']):,} samples  \n"
        f"quantum: {int(row['quantum_queries']):,} queries"
    )

st.subheader("All scenarios — stress summary")
table = _scenario_table(pair)
st.dataframe(table, use_container_width=True, hide_index=True)
st.caption(
    f"MC samples measured at ε=0.005; QAE queries at ε=0.01 (production run "
    f"settings). Like-for-like query advantage at matched ε is shown in the "
    f"benchmark panel above — {bench.iloc[-1]['speedup']:.1f}× at ε=1e-4 on "
    "this position/scenario."
)

st.subheader("Stressed return distribution")
probs, edges = discretize(stressed)
hist_df = pd.DataFrame(
    {
        "loss bucket": [f"{edges[i]:.3%} – {edges[i+1]:.3%}" for i in range(len(probs))],
        "probability": probs,
    }
)
st.bar_chart(hist_df, x="loss bucket", y="probability")
st.caption(
    f"8-bucket discretisation of the scenario-stressed loss distribution — "
    f"exactly what the quantum state-preparation circuit encodes. "
    f"Distribution type: {sc['distribution_type']}."
)
