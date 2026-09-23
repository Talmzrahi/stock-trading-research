# Pre-registration: volatility targeting on three fresh samples

**Committed before any return in these samples is scored. 2026-09-23.**

## Why three, and why these three

The mechanism passed a pre-registered test on 17 country ETFs (1996-2026) and failed
one on SPY (1993-2001). Both are spent. Calendar span has been the binding constraint on
every test in this project — pooling 2.7× the *events* bought only 1.43× the precision
because it added no extra quarters — so the samples below are chosen for what each can
settle that the others cannot, not for volume.

| Test | Sample | Fresh? | Settles |
|---|---|---|---|
| **A** | ^GSPC, **1928-1992** | yes — 65 years never touched | does it survive regimes unlike the modern one: Depression, WWII, stagflation, 1987 |
| **B** | 9 SPDR sector ETFs, 1999-2026 | yes | does it generalise to a different **cross-section**? countries worked; sectors are a different cut of the same market |
| **C** | long national indices, **pre-1996 only** | yes | Japan 1965-1995 includes the bubble and its collapse — the largest equity drawdown in the sample |

Test C is restricted to pre-1996 because the 1996-2026 window overlaps the country-ETF
cohort already spent for the UK, Japan, Canada, Hong Kong and Germany. Only the earlier
portion is untouched.

## The rule: unchanged

The standard posture, identical to both previous tests, with no parameter re-chosen:

```
w = clip(0.12 / realised_vol_21d, 0, 1.5)     set at each month end,
                                               applied from the next trading day
```

Cash earns the 13-week T-bill (^IRX, from 1990) or **0% where IRX does not reach** — which
is most of tests A and C. That is conservative: T-bills paid well in the 1930s-80s and a
book holding cash would have earned it. Stated so it cannot later be swapped in.

Mechanics come from `research/mechanics.py`, not hand-written per script.

## The dividend problem, handled before it can be exploited

^GSPC and the national indices except the DAX are **price-only**. This biases **in favour
of the strategy**: buy-and-hold forgoes the entire dividend yield, while a book at 70%
equity forgoes only 70% of it. At a 4% yield and 0.7 average weight that is roughly
1.2%/yr of free flattery.

So an assumed yield `q` is added to the equity leg, and:

- **the verdict uses q = 4%/yr**, the conservative choice — the S&P 500's dividend yield
  averaged roughly 4-5% across 1928-1990, and a higher `q` helps buy-and-hold, not the
  strategy;
- **q = 0% and 2% are also reported**, so the sensitivity is visible rather than chosen.

The DAX needs no adjustment and gets none.

## Verdicts

Each test, on the standard posture, at q = 4% where applicable:

**Test A (^GSPC 1928-1992)** — PASS requires both:
1. excess Sharpe improvement ≥ **+0.10** over buy-and-hold
2. maximum drawdown reduced by ≥ **20%** relative

**Test B (9 sectors)** — PASS requires all three, mirroring the country test:
1. **median** excess Sharpe improvement ≥ **+0.10**
2. at least **7 of 9** sectors improve
3. median drawdown reduction ≥ **20%**

**Test C (national indices, pre-1996)** — PASS requires both:
1. **median** excess Sharpe improvement across markets ≥ **+0.10**
2. at least **3 of 5** markets improve

**Overall: the mechanism stands if at least 2 of the 3 tests pass.** All three are reported
whatever they show, and a single pass out of three will be called a failure.

These are not independent: sectors sit inside the US market, and the long indices
correlate with each other. Two passes out of three is therefore weaker evidence than it
looks, and will be described that way.

## A stated hypothesis, with a direction, testable on all three

Across everything measured so far, **volatility targeting has helped most where
buy-and-hold was worst**. SPY had the best record in the project (Sharpe 0.76 on its
holdout) and the test failed there; the 17 countries had a median Sharpe of 0.29 and all
17 improved.

**Prediction, recorded before the fact:** across the ~31 series in these three samples,
the correlation between a market's own buy-and-hold Sharpe and the Sharpe improvement
from targeting it will be **negative**, at r ≤ −0.3.

If it holds, it is a usable rule — targeting earns its keep in poor, crash-prone markets
and not in strong steady ones — and it explains the SPY failure rather than excusing it.
If the correlation comes out near zero or positive, the pattern seen so far was noise and
should be dropped from the project's conclusions.

## Inherited by reference

The gate in `CLAUDE.md`, the statistics in `research/inference_check.py`, and the timing
and accounting primitives in `research/mechanics.py`. The five errors those exist to
prevent are listed in `PROJECT_STATE.md` §8b and §8c.
