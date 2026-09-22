# Pre-registration: volatility targeting, international replication

**Committed before any return in these markets is scored. 2026-09-22.**

## Why this exists

`prereg_voltarget.md` FAILED on its SPY holdout (Sharpe +0.06 against a +0.10
bar, plateau 7/12, +0.04 at 10bps). The drawdown criterion passed and the
premise held — volatility autocorrelation 0.604 on the holdout — so the
mechanism is real and the question is whether its magnitude is.

That holdout is **spent**. The only legitimate way to continue is replication on
data this project has never touched. This is that test, and it is also one shot.

## The rule: unchanged

The **standard posture** from the SPY pre-registration, unmodified:

```
w = clip(0.12 / realised_vol_21d, 0, 1.5)     set at each month end,
                                               applied from the next trading day
```

The conservative (10%/1.0) and aggressive (15%/2.0) postures are also reported.

**No parameter is re-chosen after the SPY result.** Specifically, the SPY test
showed the VIX specification beating trailing realised volatility (Sharpe 0.90
vs 0.82). That is **not** adopted here, for a reason that is principled rather
than convenient: there is no VIX for Malaysia or Austria. A specification that
only exists for the US is not the one worth validating, so realised volatility
is the specification, as before.

## Two deliberate differences from the SPY test, declared in advance

1. **Cash earns the 13-week T-bill rate (^IRX), not 0%.** The SPY document
   specified 0% before SHY existed, written conservatively. Using the actual
   risk-free rate is more correct, and IRX covers the whole period. This makes
   the two tests **not directly comparable**, and it is more favourable to the
   strategy than the SPY spec was. Stated now so it cannot be mistaken for a
   post-hoc improvement.
2. **No separate holdout.** The whole 1996-2026 sample is the test, because none
   of it has been examined.

## The universe: the whole 1996 cohort, not a selection

All 17 iShares MSCI single-country funds launched 1996-03-18, every one with
7,678 days of total-return history from the identical start date:

Australia, Austria, Belgium, Canada, France, Germany, Hong Kong, Italy, Japan,
Malaysia, Mexico, Netherlands, Singapore, Spain, Sweden, Switzerland, UK.

Taking the entire launch cohort rather than a chosen subset is deliberate: it
removes the option of having picked markets that suit the result. **Every one of
the 17 is reported, including those that lose.**

*Survivorship caveat:* a fund that had closed could not have been included. The
original cohort appears intact, but that is an assumption I have not verified
independently, and it is the one selection effect this design cannot rule out.

## These are not 17 independent tests

Developed equity markets correlate with each other at roughly 0.7-0.85, and the
1996-2026 window overlaps the 2002-2026 period already seen for SPY. The
effective number of independent observations is closer to **4-6 than to 17**.
The verdict is therefore framed on the *median* market and a *count*, not on
finding significance somewhere in seventeen tries. A single winning market means
nothing and will be reported as nothing.

## Verdict

**PASS** requires all five, on the standard posture:

1. **Median Sharpe improvement across the 17 markets ≥ +0.10** — the same bar
   SPY failed.
2. **At least 12 of 17 markets improve Sharpe** at all.
3. **Median maximum-drawdown reduction ≥ 20%** relative to buy-and-hold.
4. **Median Sharpe improvement ≥ +0.10 at 10bps** per rebalance.
5. **Parameter robustness:** across lookback ∈ {10, 21, 42, 63} × target ∈
   {10, 12, 15}%, the median-across-markets Sharpe improvement is positive in at
   least **8 of the 12** cells.

**FAIL** on any one. A fail ends the volatility-targeting line; it does not
license a third venue, a different cohort, or a re-specification.

Reported regardless: all 17 markets individually, all three postures, the full
parameter grid including losing cells, and per-market drawdowns.

## Inherited by reference

The gate in `CLAUDE.md` and the statistics in `research/inference_check.py`. No
in-sample ranking, no full-sample statistic used in a rule scored on the same
sample, no fit informed by the future. The corrections that taught those rules
are in `PROJECT_STATE.md` §8b, `research/v3_drift_rule.py` and the cost sign
error in `research/longshort_sim.py`.
