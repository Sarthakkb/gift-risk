"""Precompute a genuine three-way VaR comparison for one headline scenario:
classical Monte Carlo, quantum IQAE on Qiskit Aer, and quantum MLAE on
Amazon Braket's LocalSimulator — run independently, not derived from each
other, so agreement between all three is a real check, not a display trick.

Scenario: USD/INR under GIFT City Liquidity — Extreme (the same "Most
Stressed Scenario" headline used throughout the dashboard).

Braket's LocalSimulator (amazon-braket-default-simulator) is classical
simulation, same as Aer — no wall-clock speed claim, query-complexity
comparison only, consistent with the rest of the app's disclosed
methodology.

Run:  python -m src.precompute_braket   (writes data/braket_comparison.json)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.braket_var import braket_quantum_var
from src.classical_var import monte_carlo_var
from src.isolate import run_isolated
from src.quantum_var import quantum_var
from src.scenarios import apply_stress
from src.shocks import THEMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PAIR = "USD/INR"
THEME_KEY = "gift_liquidity"
SEVERITY_LABEL = "Extreme"


def _run_qiskit(stressed):
    """Runs inside the isolated subprocess — same convention as
    precompute_shocks.py (qiskit Aer's Rust extensions can segfault inside
    a thread that already imported certain macOS frameworks)."""
    q = quantum_var(stressed, epsilon=0.01, seed=7)
    return {"var_pct": q.var_estimate, "oracle_queries": q.oracle_queries}


def main() -> None:
    slug = PAIR.replace("/", "_").lower()
    base_returns = pd.read_csv(DATA_DIR / f"{slug}.csv")["log_return"].to_numpy()

    variant = next(
        v for v in THEMES[THEME_KEY]["variants"] if v["severity_label"] == SEVERITY_LABEL
    )
    stressed = apply_stress(
        base_returns, variant["vol_multiplier"], variant["mean_shift"],
        variant["distribution_type"], seed=1,
    )

    print(f"Scenario: {THEMES[THEME_KEY]['name']} — {SEVERITY_LABEL}, {PAIR}")

    print("Classical Monte Carlo VaR...")
    t0 = time.time()
    mc = monte_carlo_var(stressed, target_error=0.005, seed=2)
    t_classical = time.time() - t0
    print(f"  VaR {mc.var_estimate:.4%}, {mc.samples_to_converge:,} samples, {t_classical:.2f}s")

    print("Quantum IQAE on Qiskit Aer (isolated subprocess)...")
    t0 = time.time()
    qiskit_result = run_isolated(_run_qiskit, stressed)
    t_qiskit = time.time() - t0
    print(f"  VaR {qiskit_result['var_pct']:.4%}, "
          f"{qiskit_result['oracle_queries']:,} oracle queries, {t_qiskit:.2f}s")

    print("Quantum MLAE on Amazon Braket LocalSimulator...")
    t0 = time.time()
    braket_result = braket_quantum_var(stressed)
    t_braket = time.time() - t0
    print(f"  VaR {braket_result.var_estimate:.4%}, "
          f"{braket_result.oracle_queries:,} oracle queries, {t_braket:.2f}s")

    out = {
        "pair": PAIR,
        "theme_name": THEMES[THEME_KEY]["name"],
        "severity_label": SEVERITY_LABEL,
        "classical": {
            "var_pct": mc.var_estimate,
            "samples": mc.samples_to_converge,
        },
        "qiskit_aer": {
            "var_pct": qiskit_result["var_pct"],
            "oracle_queries": qiskit_result["oracle_queries"],
        },
        "braket_local": {
            "var_pct": braket_result.var_estimate,
            "oracle_queries": braket_result.oracle_queries,
        },
    }
    out_path = DATA_DIR / "braket_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
