# GIFT Risk — Quantum-Accelerated Treasury VaR

**Track 2 (Quantum Tech in Financial Services) · Focus: Risk Analysis & Derivatives Pricing via Quantum Amplitude Estimation**

A risk intelligence tool for a treasury officer at a GIFT IFSC (GIFT City, India) IBU:
daily 95% Value-at-Risk on cross-border FX positions — classical estimate,
quantum-verified — stressed under 10 macro scenarios, with AI-generated
morning-report commentary.

## Core thesis

Classical Monte Carlo VaR needs **O(1/ε²)** samples to reach target error ε.
Quantum Amplitude Estimation needs **O(1/ε)** oracle queries — a proven
quadratic query-complexity improvement (Brassard et al. 2002; here via
Iterative QAE, Grinko et al. 2021). We measure both counts empirically on the
same stressed distributions and show the crossover.

**What we do NOT claim:** wall-clock speed advantage. Everything runs on
classical simulators. The measured advantage is *algorithmic sample
efficiency* — the reason to care today is that the math and tooling are
mature enough to validate now, so the approach is production-ready the day
the hardware is. The barrier is hardware maturity, not the algorithm.

## Setup & run

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env        # add AWS credentials (optional — see fallbacks)
./venv/bin/streamlit run app.py
```

Without AWS credentials everything still runs: data loads from local CSVs and
commentary uses a clearly-labelled offline template.

Regenerate data / benchmarks:

```bash
./venv/bin/python -m src.generate_data
./venv/bin/python -m src.benchmark
```

## Architecture

```
data (synthetic FX returns + Faker metadata, S3 w/ local fallback)
  → scenarios.py        10 macro stress transforms (vol / mean / skew / tails)
  → classical_var.py    Monte Carlo VaR + samples-to-converge tracking
  → quantum_var.py      distribution → 8-bucket amplitude encoding → IQAE
                        (Qiskit Aer) → threshold sweep → 95% VaR
  → braket_var.py       parallel Braket-native MLAE backend (LocalSimulator)
  → benchmark.py        measured samples vs. oracle queries across ε sweep
  → aws_services.py     S3 loading · Bedrock (Claude Opus 5) commentary
  → isolate.py          quantum compute runs in a spawned subprocess,
                        keeping native SDK code out of the UI process
  → app.py              Streamlit dashboard (the product surface)
```

Both estimators target the *same* quantity — P(loss > threshold) on the same
stressed distribution — so their resource counts are directly comparable.

## AWS services used, and why

| Service | Role | Why chosen |
|---|---|---|
| **S3** | Stores the synthetic position datasets; the app reads via boto3 with graceful local fallback | Mirrors how a real IBU risk stack would centralise position data; costs nothing to demo credibly |
| **Braket** (LocalSimulator) | Alternative quantum execution backend beside Qiskit Aer, selectable in the UI | Shows the pipeline is not welded to one SDK — the same discretised distribution drives a Braket-native MLAE implementation, which is the migration path to managed QPUs when hardware matures |
| **Bedrock — Claude Opus 5** | Turns each VaR result + scenario narrative into 2–3 sentences of morning-report commentary | The treasury officer's actual deliverable is a written morning report; an LLM with the scenario context drafts it. Labelled AI-generated, not financial advice |

## What's synthetic/mocked vs. what's real

**Synthetic / mocked:**
- All market data: FX return series are generated (skewed Student-t, realistic
  vol/skew/kurtosis for USD/INR, SGD/INR, AED/INR) — no real market feed.
- All position metadata: entity names, position IDs, desks, counterparties are
  Faker-generated. Any resemblance to real institutions is a byproduct of
  using real bank names as plausible GIFT City tenants — no real positions.
- Stress scenarios: parameterised transforms designed by us, not calibrated
  to historical episodes.
- Quantum execution: **classical simulation** (Qiskit Aer, Braket
  LocalSimulator). No QPU was used.

**Real (actually measured / actually running):**
- The IQAE algorithm itself — genuine qiskit-algorithms implementation, real
  state-preparation circuits, real Grover-operator query counts as reported
  by the estimator. Benchmark counts are measured, not theoretical curves.
- The classical MC convergence counts — measured binomial-standard-error
  stopping, not formulas.
- S3 storage/retrieval and Bedrock (Claude Opus 5) calls — live AWS calls.
- The classical-vs-quantum agreement check on every result.

**What would need to be true in production:** real position/market data
feeds, distribution calibration sign-off by model validation, QPU execution
with error rates low enough for deep Grover circuits (the hardware-maturity
barrier), and IFSCA model-governance approval for regulatory VaR reporting.

## Honest read of the benchmark

Measured on USD/INR baseline: classical MC actually *wins* at loose target
error (constants beat asymptotics — at ε=5×10⁻³ MC needs ~2k samples vs ~3.7k
IQAE queries). The quantum advantage appears below ε≈3×10⁻³ and grows to
**~19× fewer queries at ε=10⁻⁴**. Tail-risk desks care about exactly this
tight-precision regime — deep-tail probabilities are where MC sample counts
explode.
