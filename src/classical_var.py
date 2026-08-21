"""Classical Monte Carlo VaR baseline with convergence tracking.

The MC engine resamples from the scenario-stressed empirical return
distribution and estimates 95% VaR. The PRIMARY comparison metric exposed
here is `samples_to_converge`: how many Monte Carlo samples are needed
before the VaR estimate is stable within a target standard error (epsilon).

Classical MC standard error scales as O(1/sqrt(N)) -> N ~ 1/eps^2.
This is the baseline the quantum IQAE pipeline is benchmarked against
(oracle queries ~ 1/eps). No wall-clock claims are made anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONFIDENCE = 0.95


@dataclass
class MCResult:
    var_estimate: float          # 95% VaR as positive loss fraction
    samples_to_converge: int     # samples needed to hit target_error
    target_error: float          # epsilon on the tail-probability estimate
    tail_prob_at_var: float      # P(loss > VaR) — should be ~0.05
    converged: bool


def estimate_tail_probability(
    stressed_returns: np.ndarray,
    loss_threshold: float,
    target_error: float = 0.01,
    batch: int = 200,
    max_samples: int = 2_000_000,
    seed: int = 0,
) -> tuple[float, int, bool]:
    """MC-estimate P(loss > threshold) to a target standard error.

    Draws bootstrap samples from the stressed distribution in batches and
    stops when the binomial standard error of the tail-probability estimate
    falls below `target_error`. Returns (p_hat, n_samples, converged).
    """
    rng = np.random.default_rng(seed)
    hits = 0
    n = 0
    while n < max_samples:
        draws = rng.choice(stressed_returns, size=batch, replace=True)
        hits += int(np.sum(-draws > loss_threshold))  # loss = -return
        n += batch
        p_hat = hits / n
        se = np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n)
        if se < target_error and n >= 10 * batch:
            return p_hat, n, True
    return hits / n, n, False


def monte_carlo_var(
    stressed_returns: np.ndarray,
    target_error: float = 0.01,
    seed: int = 0,
) -> MCResult:
    """Estimate 95% VaR by MC and report samples needed to converge.

    Approach: take the empirical VaR as the loss threshold, then MC-verify
    the tail probability at that threshold to target_error precision —
    mirroring exactly what the quantum estimator does (estimate P(loss>L)
    for a fixed threshold), so sample counts are comparable.
    """
    var_emp = -np.percentile(stressed_returns, (1 - CONFIDENCE) * 100)
    p_hat, n, converged = estimate_tail_probability(
        stressed_returns, var_emp, target_error=target_error, seed=seed
    )
    return MCResult(
        var_estimate=float(var_emp),
        samples_to_converge=n,
        target_error=target_error,
        tail_prob_at_var=float(p_hat),
        converged=converged,
    )
