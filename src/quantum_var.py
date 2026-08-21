"""Quantum VaR estimation via Iterative Quantum Amplitude Estimation (IQAE).

Pipeline:
  1. Discretise the scenario-stressed return distribution into 2^n buckets
     (n=3 -> 8 buckets by default).
  2. Build a state-preparation circuit A encoding bucket probabilities into
     qubit amplitudes:  A|0> = sum_i sqrt(p_i) |i>.
  3. Mark "loss exceeds threshold" buckets with an objective qubit; the
     amplitude of the objective qubit's |1> state equals P(loss > L).
  4. Run IQAE (qiskit_algorithms) on the Aer simulator to estimate that
     amplitude to target precision epsilon — using O(1/eps) oracle queries
     vs. the classical O(1/eps^2) samples.
  5. Sweep thresholds across bucket boundaries and invert to the 95% VaR.

Runs on a classical SIMULATOR (Aer). This demonstrates the query-complexity
advantage of the QAE algorithm family; it makes no wall-clock speed claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# NOTE: qiskit imports are deferred into the functions that need them.
# Importing/using qiskit's Rust extensions inside Streamlit ScriptRunner
# threads can SIGSEGV on macOS; the app calls these functions through
# src.isolate.run_isolated (a spawned subprocess), and lazy imports keep
# `from src.quantum_var import discretize` safe in the UI process.

CONFIDENCE = 0.95
DEFAULT_NUM_QUBITS = 3  # 8 buckets


@dataclass
class QAEResult:
    var_estimate: float           # 95% VaR (positive loss fraction)
    oracle_queries: int           # total oracle queries used by IQAE
    epsilon: float                # target precision on tail probability
    tail_prob_at_var: float       # IQAE-estimated P(loss > VaR)
    num_buckets: int
    thresholds_tested: int
    backend: str


def discretize(returns: np.ndarray, num_qubits: int = DEFAULT_NUM_QUBITS):
    """Bucket the LOSS distribution (loss = -return) into 2^n equal-width bins.

    Returns (probs, bin_centers) where probs sum to 1. Bin range is clipped
    at the 0.1/99.9 percentiles so a single extreme draw can't stretch the
    grid and starve the interior buckets of resolution.
    """
    losses = -returns
    lo, hi = np.percentile(losses, [0.1, 99.9])
    edges = np.linspace(lo, hi, 2**num_qubits + 1)
    probs, _ = np.histogram(np.clip(losses, lo, hi), bins=edges)
    probs = probs / probs.sum()
    return probs, edges


def build_problem(probs: np.ndarray, threshold_idx: int):
    """State-prep circuit + objective marking buckets >= threshold_idx.

    The comparator is implemented at the amplitude level: an extra objective
    qubit is rotated to |1> exactly for basis states representing losses at
    or beyond the threshold bucket. P(objective=1) = sum_{i>=t} p_i.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit_algorithms import EstimationProblem

    n = int(np.log2(len(probs)))
    amplitudes = np.sqrt(probs)

    state_prep = QuantumCircuit(n + 1)
    state_prep.append(StatePreparation(amplitudes), range(n))
    # mark loss buckets >= threshold: flip objective qubit for those states
    for i in range(threshold_idx, len(probs)):
        bits = format(i, f"0{n}b")[::-1]  # little-endian qubit order
        zeros = [q for q, b in enumerate(bits) if b == "0"]
        for q in zeros:
            state_prep.x(q)
        state_prep.mcx(list(range(n)), n)
        for q in zeros:
            state_prep.x(q)

    return EstimationProblem(
        state_preparation=state_prep,
        objective_qubits=[n],
    )


class TranspilingSampler:
    """SamplerV2 wrapper that transpiles circuits to basis gates before
    execution — IQAE submits circuits containing opaque StatePreparation /
    Grover-Q instructions that Aer cannot assemble directly."""

    def __init__(self, inner):
        self._inner = inner

    def run(self, pubs, **kwargs):
        from qiskit import transpile

        fixed = []
        for pub in pubs:
            if isinstance(pub, tuple):
                circuit, *rest = pub
                circuit = transpile(
                    circuit, basis_gates=["u", "cx"], optimization_level=1,
                    seed_transpiler=7,
                )
                fixed.append((circuit, *rest))
            else:
                fixed.append(
                    transpile(pub, basis_gates=["u", "cx"], optimization_level=1,
                              seed_transpiler=7)
                )
        return self._inner.run(fixed, **kwargs)


def _count_oracle_queries(iqae_result) -> int:
    """Total oracle (A / Grover) queries reported by IQAE."""
    return int(iqae_result.num_oracle_queries)


def iqae_tail_probability(
    probs: np.ndarray,
    threshold_idx: int,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    shots: int = 64,
    seed: int = 7,
) -> tuple[float, int]:
    """IQAE-estimate P(loss bucket >= threshold_idx). Returns (p, queries)."""
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_algorithms import IterativeAmplitudeEstimation

    problem = build_problem(probs, threshold_idx)
    sampler = TranspilingSampler(AerSampler(default_shots=shots, seed=seed))
    iqae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, sampler=sampler
    )
    res = iqae.estimate(problem)
    return float(res.estimation), _count_oracle_queries(res)


def quantum_var(
    stressed_returns: np.ndarray,
    epsilon: float = 0.01,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    seed: int = 7,
    backend_name: str = "aer_simulator",
) -> QAEResult:
    """Estimate 95% VaR by sweeping IQAE tail probabilities over buckets.

    Walk thresholds from the far tail inward. Marking buckets >= t means the
    estimated amplitude is P(loss >= edges[t]), so each threshold index maps
    to a bucket EDGE. The VaR is found where the exceedance probability
    crosses 1 - CONFIDENCE, refined by linear interpolation between the two
    bracketing edges.
    """
    probs, edges = discretize(stressed_returns, num_qubits)
    target_p = 1 - CONFIDENCE  # 0.05

    total_queries = 0
    tested = 0
    prev = (len(probs), 0.0)  # threshold beyond last bucket: p = 0
    var_est = float(edges[-1])
    p_at_var = 0.0

    for t in range(len(probs) - 1, 0, -1):
        p_est, queries = iqae_tail_probability(
            probs, t, epsilon=epsilon, seed=seed + t
        )
        total_queries += queries
        tested += 1
        if p_est >= target_p:
            # crossed 0.05 between edges[t] (p_est) and edges[t_prev] (p_prev)
            t_prev, p_prev = prev
            if p_est > p_prev:
                frac = (p_est - target_p) / (p_est - p_prev)
                var_est = float(edges[t] + frac * (edges[t_prev] - edges[t]))
            else:
                var_est = float(edges[t])
            p_at_var = p_est
            break
        prev = (t, p_est)
    else:
        var_est = float(edges[0])
        p_at_var = 1.0

    return QAEResult(
        var_estimate=var_est,
        oracle_queries=total_queries,
        epsilon=epsilon,
        tail_prob_at_var=p_at_var,
        num_buckets=2**num_qubits,
        thresholds_tested=tested,
        backend=backend_name,
    )
