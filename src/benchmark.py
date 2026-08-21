"""Benchmark: classical MC samples vs. quantum IQAE oracle queries.

For a sweep of target errors (epsilon), measure the ACTUAL resource count
each method needs to estimate the same tail probability P(loss > VaR95):
  - classical: bootstrap MC samples until binomial standard error < eps
  - quantum:   IQAE oracle queries reported for epsilon_target = eps

Theoretical overlays: N_cl ~ p(1-p)/eps^2 and N_q ~ (1/eps)*log(2/alpha)
scaled constants. We compare QUERY COMPLEXITY only — both run on classical
hardware here, so wall-clock time is meaningless and never shown.

Run:  python -m src.benchmark    (writes data/benchmark_results.csv + chart)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classical_var import estimate_tail_probability
from src.quantum_var import discretize, iqae_tail_probability
from src.scenarios import apply_scenario

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EPSILONS = [0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]


def run_benchmark(
    stressed_returns: np.ndarray, epsilons: list[float] = EPSILONS, seed: int = 3
) -> pd.DataFrame:
    """Measure actual classical samples and quantum oracle queries per epsilon."""
    var95 = -np.percentile(stressed_returns, 5)
    probs, edges = discretize(stressed_returns)
    # threshold index whose edge is closest to the empirical VaR
    t_idx = int(np.clip(np.searchsorted(edges, var95), 1, len(probs) - 1))

    rows = []
    for eps in epsilons:
        # batch sized so the convergence floor never dominates the count
        batch = max(100, int(0.05 / eps**2 / 200))
        _, n_cl, _ = estimate_tail_probability(
            stressed_returns, var95, target_error=eps,
            batch=batch, max_samples=20_000_000, seed=seed,
        )
        _, n_q = iqae_tail_probability(probs, t_idx, epsilon=eps, seed=seed)
        rows.append(
            {
                "epsilon": eps,
                "classical_samples": n_cl,
                "quantum_queries": n_q,
                "speedup": n_cl / n_q,
            }
        )
    return pd.DataFrame(rows)


def plot_benchmark(df: pd.DataFrame, out_path: Path | None = None):
    """Samples/queries vs. target error — log-y, epsilon decreasing rightward."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    eps = df["epsilon"].to_numpy()

    ax.plot(eps, df["classical_samples"], "o-", color="#1f77b4", lw=2,
            label="Classical Monte Carlo (measured samples)")
    ax.plot(eps, df["quantum_queries"], "s-", color="#2ca02c", lw=2,
            label="Quantum IQAE (measured oracle queries)")

    # theoretical scaling overlays, anchored at the largest epsilon point
    c_cl = df["classical_samples"].iloc[0] * eps[0] ** 2
    c_q = df["quantum_queries"].iloc[0] * eps[0]
    ax.plot(eps, c_cl / eps**2, "--", color="#1f77b4", alpha=0.5,
            label=r"theoretical $O(1/\varepsilon^2)$")
    ax.plot(eps, c_q / eps, "--", color="#2ca02c", alpha=0.5,
            label=r"theoretical $O(1/\varepsilon)$")

    for _, r in df.iterrows():
        ax.annotate(
            f"{r['speedup']:.0f}×",
            (r["epsilon"], np.sqrt(r["classical_samples"] * r["quantum_queries"])),
            fontsize=8, ha="center", color="#444",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()  # tighter precision to the right
    ax.set_xlabel(r"target error $\varepsilon$ (tighter →)")
    ax.set_ylabel("samples / oracle queries (log scale)")
    ax.set_title(
        "Query complexity: classical MC vs. quantum IQAE\n"
        "(simulator execution — algorithmic sample efficiency, not wall-clock time)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def main() -> None:
    base = pd.read_csv(DATA_DIR / "usd_inr.csv")["log_return"].to_numpy()
    stressed = apply_scenario(base, "baseline", seed=1)
    df = run_benchmark(stressed)
    df.to_csv(DATA_DIR / "benchmark_results.csv", index=False)
    plot_benchmark(df, DATA_DIR / "benchmark_chart.png")
    print(df.to_string(index=False))
    print("\nwrote data/benchmark_results.csv and data/benchmark_chart.png")


if __name__ == "__main__":
    main()
