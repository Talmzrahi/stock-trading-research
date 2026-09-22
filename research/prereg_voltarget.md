# Pre-registration: volatility-targeted equity exposure

**Committed before any backtest on the holdout. 2026-09-22.**

This governs the Phase 2 test of volatility targeting on SPY. It follows the
precedent of `prereg_midsmall.md`, `prereg_portfolio_top5.md` and
`prereg_inference.md`: the rule, the sample and the verdict are fixed in writing
first, and the result is reported whatever it says.

## What is being tested

Volatility is forecastable; short-horizon returns are not proportional to it.
Measured on this project's own data (SPY, 2002-2026):

- 21-day realised volatility, autocorrelation 21 days ahead: **0.648**
- correlation of realised volatility with the **next** 21 days' return: **0.063**
- VIX forecasts next-month realised volatility better than trailing realised
  volatility does: R² **0.517** vs **0.420**, RMSE 7.42pp vs 8.14pp

So exposure can be scaled down when risk is high without surrendering a
proportional share of expected return. The claim under test is that this
improves risk-adjusted return and reduces drawdown, net of costs, out of sample.

This is **not** a claim to have found an edge. Volatility targeting is well
known. What is being tested is whether it survives this project's gate when
implemented honestly.

## The rule

Each rebalance date, hold weight `w` in SPY and the remainder in the cash proxy
(SHY; the risk-free leg before SHY exists is 0% return, stated so it cannot be
quietly changed later):

```
w = clip(target_vol / forecast_vol, 0, cap)
```

- **Forecast volatility**, two specifications, both reported:
  - **A (primary):** annualised standard deviation of the trailing 21 daily
    returns, **lagged one day** so it is known before the weight applies.
  - **B (secondary):** VIX close, lagged one day, divided by 100.
  B is pre-specified on the forecasting measurement above, not on returns. Both
  are reported; neither may be dropped after the fact.
- **Rebalance:** monthly, on the last trading day. Weekly and daily-with-band
  are robustness checks only.
- **Risk postures**, all three reported regardless of outcome:

  | | target | cap |
  |---|---|---|
  | conservative | 10% | 1.0 (never borrows) |
  | standard | 12% | 1.5 |
  | aggressive | 15% | 2.0 |

- **Costs:** 2bps of traded notional per rebalance. Sensitivity at 5 and 10bps
  is reported. Any posture with cap > 1.0 is additionally charged **margin
  interest on borrowed notional** at the 3-month T-bill rate + 1.5%; leverage is
  not free and the result must not pretend it is.
- **Benchmark:** SPY buy-and-hold over the identical window.

## Samples

| | period | status |
|---|---|---|
| Development | 2002-01 → 2026-09 | **already examined** — the feasibility run below is known |
| **Holdout** | **1993-01 → 2001-12** | **never examined, one shot** |

The holdout exists only because Phase 0 extended SPY back to 1993; everything in
`research.db` starts in 2002. Nine years, covering the 1990s bull market, the
1998 crisis and the dot-com peak — a regime unlike anything in the development
sample. **It is spent once.** No retuning, no second look, whatever it returns.

### Already known, declared for honesty

A feasibility run was executed before this document, with target 12% / cap 1.5 /
21-day realised vol / monthly, on 2002-2026, costs 2bps, no margin charge:

| | return | vol | Sharpe | max DD |
|---|---|---|---|---|
| SPY buy and hold | 9.97% | 18.9% | 0.53 | −55.2% |
| vol-targeted, cap 1.5 | 9.87% | 15.1% | 0.65 | −38.6% |
| vol-targeted, no leverage | 8.80% | 13.0% | 0.68 | −33.7% |

By era (Sharpe, SPY vs targeted): 2002-2009 **0.07 → 0.39**; 2010-2019
0.92 → 0.96; 2020-2026 **0.78 → 0.56**.

The 2020-2026 loss is the known failure mode and is expected, not a surprise to
be explained away: volatility targeting de-risks into a spike and misses a
V-shaped rebound. It is recorded here **before** the verdict is set so that
passing cannot be achieved by discovering a reason to exclude it.

Because the development period is already seen, **the development result is not
evidence.** Only the holdout and the robustness battery carry weight.

## Verdict

**PASS** requires all four, on the **holdout**, for the standard posture
(12% / 1.5) with specification A:

1. **Sharpe improvement ≥ +0.10** over SPY buy-and-hold in the same window.
2. **Maximum drawdown reduced by ≥ 20% relative** to SPY buy-and-hold.
3. **Not a lone peak.** With lookback ∈ {10, 21, 42, 63} and target ∈ {10, 12,
   15}%, at least **8 of the 12** neighbouring combinations must also improve
   Sharpe. A single winning cell is a fitted parameter, not an effect.
4. **Cost robustness.** The Sharpe improvement survives at 10bps per rebalance.

**FAIL** on any of the four. A fail stops the build; it does not license a
search for a variant that passes.

Reported alongside, never as a substitute for the verdict: all three postures,
both forecast specifications, per-era breakdowns, and the full parameter grid
including the cells that lose.

## What would falsify the premise itself

If volatility autocorrelation on the holdout is below ~0.3, or if realised
volatility correlates with next-month returns above ~0.3 in absolute value, the
mechanism does not hold in that era and the strategy should not be expected to
work there. Both are measured on the holdout **before** returns are scored, and
reported either way.

## Inherited by reference

The gate in `CLAUDE.md` and the statistics in `research/inference_check.py`
apply unchanged. Specifically: no in-sample ranking, no full-sample statistics
used in a rule evaluated on the same sample, no fitting on data that includes
the future. Four results in this project died to exactly those three errors on
2026-09-21/22 and the corrections are recorded in `PROJECT_STATE.md` §8b and
`research/v3_drift_rule.py`.

Cost handling is first-class here for the same reason: a sign error in the
long-short book's cost accounting inverted short-side costs into gains and
overstated that strategy by about +0.9pp/yr until it was caught.
