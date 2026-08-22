"""Precompute every (theme, severity, pair) combination for the shock-ladder
dashboard and cache it to data/shock_results.json.

Why precompute rather than compute on each Streamlit page load: the
dashboard's whole premise is "everything runs and displays at once, no
button click" — but quantum IQAE genuinely takes real wall-clock time (even
cut down to one representative variant per theme), and a judge opening a
fresh session shouldn't wait minutes for 27 quantum subprocess runs. So this
script runs the real pipeline ONCE, and app.py loads the cached JSON
(re-running only the pure-arithmetic hedge/trade layer live, which is what
actually needs to react to notional/target-ratio edits).

Classical Monte Carlo VaR is cheap: computed for real on every one of the
129 (variant x pair) combinations, no scaling. Quantum IQAE is computed for
real on the 9 theme representatives x 3 pairs = 27 combinations (batched
into a single subprocess call to pay qiskit's import cost once, not 27
times); every other variant's quantum VaR is that pair's representative
quantum VaR scaled by the ratio of vol_multipliers — disclosed in the
dashboard's methodology section.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.classical_var import monte_carlo_var
from src.isolate import run_isolated
from src.quantum_var import quantum_var
from src.scenarios import apply_stress
from src.shocks import THEMES, representative_variant

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAIRS = ["USD/INR", "SGD/INR", "AED/INR"]


def _load_base_returns():
    out = {}
    for pair in PAIRS:
        slug = pair.replace("/", "_").lower()
        out[pair] = pd.read_csv(DATA_DIR / f"{slug}.csv")["log_return"].to_numpy()
    return out


def _compute_all_representative_quantum(base_by_pair):
    """Runs inside the isolated subprocess: real IQAE for every
    (theme representative, pair) combination, batched in one process."""
    results = {}
    for theme_key, theme in THEMES.items():
        rep = representative_variant(theme_key)
        for pair in base_by_pair:
            stressed = apply_stress(
                base_by_pair[pair], rep["vol_multiplier"], rep["mean_shift"],
                rep["distribution_type"], seed=1,
            )
            q = quantum_var(stressed, epsilon=0.01, seed=7)
            results[f"{theme_key}::{pair}"] = {
                "var_pct": q.var_estimate,
                "oracle_queries": q.oracle_queries,
            }
    return results


def main() -> None:
    base_by_pair = _load_base_returns()

    print("Baseline (unstressed) classical VaR per pair...")
    baseline_var = {}
    for pair in PAIRS:
        mc = monte_carlo_var(base_by_pair[pair], target_error=0.005, seed=2)
        baseline_var[pair] = mc.var_estimate
        print(f"  {pair}: {mc.var_estimate:.4%}")

    print(f"\nRunning {len(THEMES)} themes x {len(PAIRS)} pairs = "
          f"{len(THEMES) * len(PAIRS)} real quantum IQAE runs (one subprocess)...")
    rep_quantum = run_isolated(_compute_all_representative_quantum, base_by_pair)
    for k, v in rep_quantum.items():
        print(f"  {k}: quantum VaR {v['var_pct']:.4%}  ({v['oracle_queries']:,} queries)")

    print("\nClassical MC VaR for all 129 (variant, pair) combos, "
          "quantum VaR real for representatives / scaled otherwise...")
    rows = []
    for theme_key, theme in THEMES.items():
        rep = representative_variant(theme_key)
        for pair in PAIRS:
            rep_q = rep_quantum[f"{theme_key}::{pair}"]
            for variant in theme["variants"]:
                stressed = apply_stress(
                    base_by_pair[pair], variant["vol_multiplier"], variant["mean_shift"],
                    variant["distribution_type"], seed=1,
                )
                mc = monte_carlo_var(stressed, target_error=0.005, seed=2)
                is_rep = variant["severity_label"] == rep["severity_label"]
                if is_rep:
                    quantum_pct = rep_q["var_pct"]
                    queries = rep_q["oracle_queries"]
                    quantum_source = "measured"
                else:
                    ratio = variant["vol_multiplier"] / rep["vol_multiplier"]
                    quantum_pct = rep_q["var_pct"] * ratio
                    queries = rep_q["oracle_queries"]  # representative's measured count
                    quantum_source = "scaled"

                rows.append({
                    "theme_key": theme_key,
                    "theme_name": theme["name"],
                    "pair": pair,
                    "severity_label": variant["severity_label"],
                    "severity_value": variant["severity_value"],
                    "direction": variant["direction"],
                    "vol_multiplier": variant["vol_multiplier"],
                    "distribution_type": variant["distribution_type"],
                    "narrative": variant["narrative"],
                    "is_representative": is_rep,
                    "classical_var_pct": mc.var_estimate,
                    "mc_samples": mc.samples_to_converge,
                    "quantum_var_pct": quantum_pct,
                    "quantum_source": quantum_source,
                    "oracle_queries": queries,
                })

    out = {
        "baseline_var": baseline_var,
        "rows": rows,
    }
    out_path = DATA_DIR / "shock_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
