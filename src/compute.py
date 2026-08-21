"""Aggregated compute entry points, designed to run via src.isolate in a
single spawned subprocess per dashboard action (module-level, picklable)."""

from __future__ import annotations

import numpy as np

from src.classical_var import monte_carlo_var
from src.quantum_var import quantum_var
from src.scenarios import SCENARIOS, apply_scenario


def scenario_table_rows(base_returns: np.ndarray) -> list[dict]:
    """Classical + quantum VaR for all 10 scenarios (one subprocess call)."""
    rows = []
    for key, sc in SCENARIOS.items():
        stressed = apply_scenario(base_returns, key, seed=1)
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
    return rows
