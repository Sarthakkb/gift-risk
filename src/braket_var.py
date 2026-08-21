"""Braket-native amplitude estimation backend for GIFT Risk.

Qiskit circuits are not directly executable on Braket, so this is a
PARALLEL, Braket-native implementation of the same amplitude-estimation
pipeline, clearly labelled as such. It runs on the Braket LocalSimulator
(amazon-braket-default-simulator) — still classical simulation, no
wall-clock claims.

Approach: same discretisation as src/quantum_var.py, state preparation via
explicit unitary (fine at 4 qubits), Grover-style amplification with
maximum-likelihood amplitude estimation (MLAE, Suzuki et al. 2020) across a
power schedule — an amplitude-estimation variant with the same O(1/eps)
query scaling as IQAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    BRAKET_AVAILABLE = True
except Exception:  # pragma: no cover
    BRAKET_AVAILABLE = False

from src.quantum_var import CONFIDENCE, DEFAULT_NUM_QUBITS, discretize


@dataclass
class BraketQAEResult:
    var_estimate: float
    oracle_queries: int
    tail_prob_at_var: float
    num_buckets: int
    backend: str = "braket_local"


def _state_prep_unitary(probs: np.ndarray) -> np.ndarray:
    """Unitary whose first column is sqrt(p) — prepares sum sqrt(p_i)|i>."""
    n = len(probs)
    a = np.sqrt(probs).reshape(-1, 1)
    # complete to an orthonormal basis via QR on [a | I]
    m = np.hstack([a, np.eye(n)[:, : n - 1]])
    q, _ = np.linalg.qr(m)
    # fix sign so first column is +sqrt(p)
    if q[0, 0] < 0:
        q[:, 0] *= -1
    return q


def _grover_unitary(A: np.ndarray, marked: np.ndarray) -> np.ndarray:
    """Q = A S0 A^dag S_chi for marked basis states."""
    n = A.shape[0]
    S_chi = np.eye(n)
    S_chi[marked, marked] = -1
    S0 = -np.eye(n)
    S0[0, 0] = 1
    return A @ S0 @ A.conj().T @ S_chi


def _run_circuit(U: np.ndarray, n_qubits: int, shots: int, marked: np.ndarray) -> float:
    """Apply unitary U to |0...0> on the Braket local simulator; return
    the empirical probability of measuring a marked state."""
    circ = Circuit()
    circ.unitary(matrix=U, targets=list(range(n_qubits)))
    circ.probability()
    device = LocalSimulator()
    task = device.run(circ, shots=shots)
    probs = task.result().values[0]
    return float(np.sum(probs[marked]))


def braket_tail_probability(
    probs: np.ndarray,
    threshold_idx: int,
    shots: int = 64,
    powers: tuple[int, ...] = (0, 1, 2, 4, 8),
) -> tuple[float, int]:
    """MLAE estimate of P(bucket >= threshold_idx) on Braket LocalSimulator.

    Runs Grover powers k in `powers`, measures hit rates, and maximises the
    combined likelihood over the amplitude angle. Oracle queries counted as
    shots * (2k + 1) per power — same accounting as the Qiskit IQAE path.
    """
    n_qubits = int(np.log2(len(probs)))
    marked = np.arange(threshold_idx, len(probs))
    A = _state_prep_unitary(probs)
    Q = _grover_unitary(A, marked)

    hits, total_queries = [], 0
    for k in powers:
        U = np.linalg.matrix_power(Q, k) @ A
        p_hit = _run_circuit(U, n_qubits, shots, marked)
        hits.append((k, p_hit))
        total_queries += shots * (2 * k + 1)

    # maximum-likelihood over theta: P_k(hit) = sin^2((2k+1) theta)
    thetas = np.linspace(1e-4, np.pi / 2 - 1e-4, 20_000)
    loglik = np.zeros_like(thetas)
    for k, p_hit in hits:
        m = np.sin((2 * k + 1) * thetas) ** 2
        m = np.clip(m, 1e-9, 1 - 1e-9)
        h = round(p_hit * shots)
        loglik += h * np.log(m) + (shots - h) * np.log(1 - m)
    theta_hat = thetas[np.argmax(loglik)]
    return float(np.sin(theta_hat) ** 2), total_queries


def braket_quantum_var(
    stressed_returns: np.ndarray,
    num_qubits: int = DEFAULT_NUM_QUBITS,
    shots: int = 64,
) -> BraketQAEResult:
    """95% VaR via Braket-native MLAE threshold sweep (mirrors quantum_var)."""
    if not BRAKET_AVAILABLE:
        raise RuntimeError("amazon-braket-sdk not installed")
    probs, edges = discretize(stressed_returns, num_qubits)
    target_p = 1 - CONFIDENCE

    total_queries = 0
    prev = (len(probs), 0.0)
    var_est, p_at_var = float(edges[-1]), 0.0

    for t in range(len(probs) - 1, 0, -1):
        p_est, queries = braket_tail_probability(probs, t, shots=shots)
        total_queries += queries
        if p_est >= target_p:
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
        var_est, p_at_var = float(edges[0]), 1.0

    return BraketQAEResult(
        var_estimate=var_est,
        oracle_queries=total_queries,
        tail_prob_at_var=p_at_var,
        num_buckets=2**num_qubits,
    )
