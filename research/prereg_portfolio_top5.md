# Pre-registration — re-specifying the S&P 500 strategy if the out-of-sample test passes

Written 2026-09-15, **before the result of `research/oos_midsmall.py` is known** (its data
was still downloading). Conditional on that test; if it FAILS or is INCONCLUSIVE, none of
this runs.

## Why a second pre-registration

The first portfolio gate chose top 2% + an 8-sd stop as the best of 20 cells on a
survivors-only universe; walk-forward kept only 9% of the excess return. If the
out-of-sample test supports the top-5% cutoff, the S&P 500 strategy should be
re-specified — but the selection must not become another search over 20 cells.

## Fixed procedure (only if the out-of-sample test PASSES)

1. **Cutoff is fixed at top 5%** (`cutoff = 0.95`). It is not re-optimised on S&P 500
   data; it comes from the out-of-sample test.
2. **Universe:** the repaired S&P 500 (departed firms restored, membership by firm) —
   `trader.backtest.load_market` as it stands.
3. **Stop:** the only free parameter. `research/portfolio_gate.py` runs with
   `CUTOFFS = [0.95]` and the existing stop grid `[None, 3, 5, 8, 12]`, and the existing
   rule in `research/selection.py` picks it: a multiple k qualifies only if it beats
   no-stop, **both** grid neighbours also beat no-stop, and it is no worse than no-stop
   in at least 2 of 3 eras. Otherwise: no stop.
4. **Everything else stays**: 60-session hold, fixed fraction at entry, idle cash in SPY,
   10bps round trip, $1,000, `slot_mult = 1.0`, `min_slots = 5`.
5. **The tripwire band is rebuilt** from the chosen configuration's trades, and
   `config/strategy.json` is rewritten by that gate run.
6. **A walk-forward check is reported** (`research/stress_test.py` with the fixed cutoff)
   but is **not** a gate: with the cutoff fixed and only the stop free, the selection
   search is small by construction.

## What this does NOT authorise

- No real money. Phase 4 stays deferred; the ~$100 stake is the owner's decision, made
  after live paper results, not by this document.
- No switch of the running simulated account to Alpaca. That is an owner decision too.
- If the gate's chosen configuration has a worse CAGR than SPY over the repaired
  backtest, the strategy is reported as not worth trading and the simulated account is
  left as an experiment.
