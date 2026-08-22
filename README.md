# GIFT Risk — Quantum-Accelerated Treasury VaR

**Track 2 (Quantum Tech in Financial Services) · Focus: Risk Analysis & Derivatives Pricing via Quantum Amplitude Estimation**

A morning FX treasury dashboard for a GIFT IFSC IBU treasury officer: open it
and immediately see how a 3-book portfolio (USD/INR, SGD/INR, AED/INR) —
plus a $300M FCNR(B) funding book — moves under a ladder of 9 macro shock
themes (~43 severity variants), and exactly what FX forward trades are
needed today to hold each book's target hedge ratio. Quantum amplitude
estimation (Qiskit Aer IQAE) feeds the risk numbers in the background — it's
supporting infrastructure, not the primary interface. There is no manual
scenario picker: everything computes and renders on load.

**Full feature set:** FX spot/forward rate ticker · FCNR(B) funding base
with a live countdown to RBI's real swap-window deadline · portfolio position
cards with hedge-drift status · VaR limit gauges (per-pair + portfolio,
baseline vs. worst-case) · a unified risk alerts feed (hedge drift + VaR
breaches + funding gaps, one sorted list) · a 3×3 currency correlation matrix
· a 43-row progressive shock heatmap with click-to-expand detail · a trade
blotter with FCNR funding-gap warnings · portfolio stress summary · P&L
attribution waterfall · a 30-day VaR trend · an AI morning briefing (Bedrock)
· a hedge & funding maturity calendar · a downloadable morning report · and
a collapsible quantum methodology section with the real classical-vs-quantum
benchmark.

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

Regenerate data / benchmarks / the shock-ladder cache:

```bash
./venv/bin/python -m src.generate_data
./venv/bin/python -m src.benchmark
./venv/bin/python -c "from src.precompute_shocks import main; main()"
```

## Architecture

```
data (synthetic FX returns + Faker metadata, S3 w/ local fallback)
  → shocks.py            9 macro themes x ~43 severity variants (the
                         progressive shock ladder) — vol/mean/dist params
  → scenarios.py         apply_stress(): the generic distribution-transform
                         engine underneath every variant (vol / mean /
                         skew / tails); apply_scenario() is the legacy
                         10-scenario wrapper over the same engine, kept
                         for reference, no longer driving the dashboard
  → classical_var.py     Monte Carlo VaR + samples-to-converge tracking —
                         run for REAL on all 129 (variant x pair) combos
  → quantum_var.py       distribution → 8-bucket amplitude encoding → IQAE
                         (Qiskit Aer) → threshold sweep → 95% VaR — run
                         for REAL on 27 representative (theme x pair) combos
  → precompute_shocks.py runs the 129 classical + 27 quantum combos ONCE,
                         caches to data/shock_results.json so the live
                         dashboard never blocks a judge's session on
                         quantum compute
  → portfolio.py         target hedge ratio → required hedge ratio given a
                         shock's severity → drift → trade recommendation
                         (reuses hedge.py's buy/sell direction logic)
  → hedge.py             payable/receivable buy-vs-sell direction logic,
                         shared by portfolio.py
  → fcnr.py              FCNR(B) funding book: real RBI swap-window dates,
                         retention-multiplier model (scoped to 2 themes),
                         funding-gap and post-window hedging-cost formulas
  → benchmark.py         measured samples vs. oracle queries across ε sweep
  → aws_services.py      S3 loading · Bedrock (Claude Opus 5) commentary,
                         both per-cell and portfolio-level morning briefing
  → isolate.py           quantum compute runs in a spawned subprocess,
                         keeping native SDK code out of the UI process
  → app.py               Streamlit dashboard — rate ticker, FCNR funding
                         base, portfolio cards, VaR limit gauges, unified
                         alerts feed, correlation matrix, shock heatmap,
                         trade blotter, stress summary, P&L waterfall,
                         VaR trend, AI morning briefing, maturity calendar,
                         report export, quantum methodology
```

Both estimators target the *same* quantity — P(loss > threshold) on the same
stressed distribution — so their resource counts are directly comparable.

## FCNR(B) funding book — real regulatory backdrop

On 8 June 2026, RBI circular RBI/2026-27/99 opened a USD-INR forex swap
facility for fresh FCNR(B) deposits (3-5yr tenor), absorbing banks'
currency-hedging cost so they could offer NRIs sharply higher rates without
extra cost to the bank. Inflows (~$52.3bn FCNR-specific, ~$56bn+ across all
three swap windows as of mid-August) far exceeded the comparable 2013
scheme, prompting RBI to move the mobilisation deadline forward to 31 August
2026 (from 30 September) and the swap-settlement deadline to 11 September
2026. After that, banks bear the full hedging cost themselves on any fresh
or rolled-over FCNR(B) funding. A meaningful share of inflows came through
leveraged structures yielding depositors ~14-15% effective returns after
borrowing costs — flagged as a real rollover-risk concern, since that
premium may not survive once the subsidy ends. These facts were verified
via web search before being cited here, not assumed:

- [Business Standard — FCNR(B) swap window explained: why RBI opened it, then advanced the deadline](https://www.business-standard.com/finance/news/rbi-fcnr-b-window-nri-dollar-swap-deposit-scheme-deadline-forex-126081700609_1.html)
- [Business Today — RBI limits FCNR(B) forex swap facility after $52.3bn inflows](https://www.businesstoday.in/amp/latest/economy/story/rbi-limits-fcnrb-forex-swap-facility-after-52-3-bn-inflows-ecb-ofcb-window-stays-open-549331-2026-08-14)
- [RBI/2026-27/99 circular text, 8 June 2026](https://caalley.com/rbi26/99NT1723C9DF80134A538854DA7B2DA84F37.pdf)
- [Vinod Kothari Consultants — leveraged FCNR deposits, effective yields and costs](https://vinodkothari.com/2026/07/leveraged-fcnr-deposits-higher-returns-for-nris-at-whose-cost/)

Everything built ON TOP of that backdrop — the specific $300M book, its rate
and tenor, the retention-multiplier values by scenario, and the funding-gap
formula — is a synthetic illustration (`src/fcnr.py`), not real GIFT City
IBU data. Retention risk is modeled on only 2 of the 9 shock themes (Fed
Rate Move, India Sovereign Spread) — the two macro drivers that plausibly
move NRI deposit retention; the other 7 carry a retention multiplier of 1.0
(no modeled effect), stated openly in the dashboard's methodology section.

## HDFC Case Study tab — a live worked example

A second dashboard tab runs a real, current GIFT City transaction through the
exact same engine as the portfolio dashboard: on 21 Aug 2026, HDFC Bank's
GIFT City IBU priced **US$1.75bn in senior unsecured bonds** — $500M at
5.159% (3yr) and $1.25B at 5.4% (5yr), settling 26 Aug 2026, rated Baa3
(Moody's) / BBB (S&P), proceeds for overseas lending. These facts were
verified via web search before being cited, not assumed:

- [Business Standard — HDFC Bank raises $1.75bn through overseas bond issue](https://www.business-standard.com/finance/news/hdfc-bank-raises-1-75-bn-through-overseas-bond-issue-to-fund-business-126082100360_1.html)
- [TipRanks — HDFC Bank raises US$1.75 billion via GIFT City bond issuance](https://www.tipranks.com/news/company-announcements/hdfc-bank-raises-us1-75-billion-via-gift-city-bond-issuance)

The bond itself is modeled as a new USD/INR "payable" position, reusing the
same 43-variant shock ladder, the same quantum-verified VaR (Qiskit Aer
IQAE), and the same hedge-drift/trade-recommendation engine as the main
portfolio — no new computation, no new methodology to trust. The tab is
explicit about what's real and what's illustrative: the transaction facts
(amounts, coupons, dates, ratings) are real and cited; the target hedge
ratio (85%, reflecting a natural offset from USD-denominated overseas
lending) is an illustrative assumption, since HDFC's actual internal hedging
policy is not public information.

## Progressive shock ladder & hedge engine

Running IQAE on every one of the 43 severity variants × 3 pairs (129
combinations) isn't worth doing on a simulator, so for each of the 9 themes,
IQAE runs for real on ONE representative (middle-severity) variant, per
currency pair — 27 real quantum runs. Every other variant's quantum VaR is
that pair's representative quantum VaR scaled by the ratio of
`vol_multiplier`s. Classical Monte Carlo VaR is cheap enough to run for real
on all 129 combinations — no scaling there. This is disclosed in the
dashboard's own "Quantum Methodology" section, not hidden.

The hedge engine (`portfolio.py`) assumes each book starts the day hedged
exactly at its target ratio, then asks: given this shock's severity
(`vol_multiplier`, scaled linearly toward the ladder's most extreme
variant), what hedge ratio would be needed to keep risk in check? The gap
between that required ratio and the target is the "drift" — if it exceeds
the threshold (default 5 points), a trade is recommended via `hedge.py`'s
existing payable/receivable buy-vs-sell direction logic, reused unchanged
from the single-position Hedge Ratio calculator in the previous build.

## AWS services used, and why

| Service | Role | Why chosen |
|---|---|---|
| **S3** | Stores the synthetic position datasets; the app reads via boto3 with graceful local fallback | Mirrors how a real IBU risk stack would centralise position data; costs nothing to demo credibly |
| **Braket** (LocalSimulator) | A parallel Braket-native MLAE backend (`braket_var.py`) exists in the codebase from the earlier single-scenario build, showing the pipeline isn't welded to one SDK — the current dashboard's precomputed cache uses Aer only | The migration path to managed QPUs when hardware matures |
| **Bedrock — Claude Opus 5** | Two uses: a per-cell commentary when a heatmap cell is expanded, and one portfolio-level "AI Morning Risk Briefing" generated once per session | The treasury officer's actual deliverable is a written morning report; an LLM with the scenario context drafts it. Labelled AI-generated, not financial advice |

## What's synthetic/mocked vs. what's real

**Synthetic / mocked:**
- All market data: FX return series are generated (skewed Student-t, realistic
  vol/skew/kurtosis for USD/INR, SGD/INR, AED/INR) — no real market feed.
- All position metadata: entity names, position IDs, desks, counterparties are
  Faker-generated. Any resemblance to real institutions is a byproduct of
  using real bank names as plausible GIFT City tenants — no real positions.
- The 43 shock-ladder severities: parameterised transforms designed by us,
  not calibrated to historical episodes or a real risk model.
- Quantum execution: **classical simulation** (Qiskit Aer). No QPU was used.
- Portfolio state: notionals, target hedge ratios, and exposure types
  (payable/receivable) per book are hand-set synthetic parameters, not read
  from any real forward book or treasury management system.
- The required-hedge-ratio formula (linear in `vol_multiplier` toward the
  ladder's most extreme severity) is a modeling choice we designed to make
  mild-vs-severe shocks discriminate cleanly, not an industry-standard model.
- The FCNR(B) book ($300M, 3yr, 6.5%), its retention-multiplier values, and
  the funding-gap/post-window-hedging-cost formulas are synthetic —
  illustrative math on top of the real RBI swap-window backdrop (dates,
  circular reference, and inflow figures are real and cited above).
- FX spot/forward rates on the ticker: mock values built from the pipeline's
  own synthetic spot levels plus a plausible rate-differential forward
  premium — not a live market feed.
- The hedge & funding maturity calendar's FX forward tranches (30/60/90-day)
  are illustrative, sized off the current target hedge ratio — no real
  forward-contract ledger exists behind them. The FCNR(B) deadline dates on
  the same calendar ARE the real regulatory dates.
- The 30-day VaR trend is a synthetic random walk around today's real
  computed VaR level, explicitly labeled as such in the UI — not historical
  data.
- The correlation matrix reflects independently-seeded synthetic return
  series (see the app's own caption on this) — the near-zero values are an
  artifact of that construction, not a market finding.

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
