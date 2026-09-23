# Pre-registration: volatility targeting on emerging markets, and on the recent regime

**Committed before any return in this cohort is scored. 2026-09-23.**

## What this is for

Five pre-registered tests so far: two pass, three fail (`PROJECT_STATE.md` §8d). The one
consistent finding is that targeting reduces **drawdown** everywhere and moves **Sharpe**
inconsistently. Two questions are still open and this cohort can answer both:

1. **Does it work in markets with much larger drawdowns than any tested so far?** Emerging
   markets routinely fall 50-80%. If the tool is a drawdown tool, this is where it should
   look best.
2. **Does it work in the recent regime?** Every failure has clustered late — the
   diversified book returns +6.6%/yr against SPY's +15.1% since 2021. No test yet has
   isolated the recent period on data that had not already been seen.

## The cohort

All 14 single-country emerging-market ETFs with at least 14 years of history. Untouched:
the 17-fund cohort already spent was **developed** markets from 1996.

Brazil (EWZ, 2000), Taiwan (EWT, 2000), Korea (EWY, 2000), South Africa (EZA, 2003),
Chile (ECH, 2007), Turkey (TUR, 2008), Thailand (THD, 2008), Vietnam (VNM, 2009),
Peru (EPU, 2009), Indonesia (EIDO, 2010), Philippines (EPHE, 2010), Brazil small caps
(EWZS, 2010), China (MCHI, 2011), India (INDA, 2012).

All are total-return series, so there is no dividend adjustment and no price-only bias.
Cash earns the 13-week T-bill throughout. **All 14 exist from 2012 or earlier, so the
2015-2026 window is complete for every one of them.**

*Survivorship:* a fund that closed could not be included. Single-country EM funds do get
liquidated, so unlike the 1996 developed cohort this one has a real closure gap I cannot
size. It biases toward markets that stayed investable.

## The rule: unchanged, for the seventh time

```
w = clip(0.12 / realised_vol_21d, 0, 1.5)     set at each month end,
                                               applied from the next trading day
```

No parameter re-chosen after any earlier result. Mechanics from `research/mechanics.py`.

## Two tests

**Test D — full history.** Each fund over its own complete record. PASS requires all three:

1. median excess-Sharpe improvement ≥ **+0.10**
2. at least **10 of 14** funds improve
3. median maximum-drawdown reduction ≥ **20%**

**Test E — the recent regime.** The same 14 funds restricted to **2015-01 → 2026-09**
(11.7 years, complete for all of them). Same three criteria.

E is a sub-period of D, so these are **two looks at one sample**, not independent
evidence. Declared now rather than discovered later. D is the test of the mechanism; E is
the test of whether it still works, and E is the one I actually care about.

## Predictions, recorded in advance

Based on the pattern across the five earlier tests — drawdown consistent, Sharpe not —
I expect:

- **Drawdown criterion (3) PASSES in both D and E**, and by a wide margin in D, because EM
  drawdowns are the largest in anything tested.
- **Sharpe criterion (1) FAILS in at least one of D and E.**
- **Test E performs worse than Test D** on the Sharpe measure, because every failure so far
  has clustered in the recent period.

If instead the Sharpe criterion passes cleanly in both, the "risk tool, not a return tool"
conclusion in §8d is wrong and must be revised. If the drawdown criterion fails, the single
consistent finding of the whole programme collapses and volatility targeting should be
dropped entirely.

Either outcome is informative, which is the point of writing it down.

## Inherited by reference

The gate in `CLAUDE.md`, the statistics in `research/inference_check.py`, the primitives in
`research/mechanics.py`, and the five errors they exist to prevent (§8b, §8c).
