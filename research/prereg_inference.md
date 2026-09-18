# Pre-registration: do the PEAD results survive overlap-robust inference?

Written 2026-09-18, **before any of the tests below has been run on S&P 400/600 data**.
Committed to git before the script that runs them, so the history shows the order.

## Why this test

Every event-level gate in this project clusters by **entry date**. Positions are held 60
sessions, so events entered a few days apart share most of their return window. Date
clustering treats them as independent draws, which understates the uncertainty.

Two quick checks (already seen, so not blind) suggest this matters:

- S&P 500 top-5% events, one mean per cluster: p = 0.001 clustered by date, 0.013 by
  week, 0.004 by month, **0.23 by calendar quarter**.
- The live strategy's daily backtest returns regressed on SPY: beta 1.09, alpha +1.95%
  a year, **t = 0.70**.

The question is whether the evidence that fixed the strategy survives inference that
respects the overlap and the market exposure. No cutoff, horizon, cost, universe or return
definition changes. Only the statistics change.

## What is blind and what is not

- **Blind (primary):** everything on the S&P 400/600 out-of-sample data. Only its original
  date-clustered result (+2.730pp, p = 0.0009) has been seen.
- **Not blind (secondary):** S&P 500 event level (the rough quarter check above was seen)
  and the portfolio-level alpha (a daily version was seen). They are run under the same
  fixed rules, and their verdicts are labelled as not blind.

## Samples, exactly as originally tested

- **S&P 400/600:** `research/oos_midsmall.py` primary. Top 5% trailing SUE, point-in-time
  members, 60-session return minus the member's index ETF (IJH / IJR), net 20bps.
- **S&P 500:** `research/pead_trailing.py` on the repaired universe. Top 5% trailing SUE,
  60-session return minus SPY, net 10bps.

**Replication step (T0):** the script first recomputes each original date-clustered
result. It must reproduce +2.730pp / p = 0.0009 (S&P 400/600) and +1.91pp / p = 0.002
(S&P 500) to the stated precision. If it does not, the run stops and the mismatch is
investigated before anything else is reported.

## Tests

**T1: same estimate, overlap-robust standard error.** The original estimate is the mean of
per-entry-date means. It is kept exactly. Its standard error is recomputed with those
per-date means clustered by **calendar quarter of entry** (CR1, t on G−1 degrees of
freedom). A quarter is about one 60-session holding period.

**T2: calendar-time portfolio, market-adjusted.** On each trading day, form an
equal-weighted portfolio of every event inside its holding window (sessions entry+1 through
entry+60). The entry cost (20bps / 10bps) is deducted on each position's first day. A
position whose price series ends early is held flat at its last close, as in the original
test. Its benchmark is the same positions' matched ETF returns, equal-weighted (IJH/IJR
for 400/600, SPY for the 500). Days with no open position are left out. Daily returns are
compounded to calendar months, and months with fewer than 5 such days are dropped.
Monthly portfolio return is regressed on monthly benchmark return; **α is the intercept**,
with Newey-West standard errors (3 lags) and t on n−2 degrees of freedom. There is no
risk-free rate (it is not in the free data). With β > 1 that biases α slightly *down*, by
about (β−1) × the T-bill rate.

**T3: the strategy itself vs SPY (S&P 500 only, the live universe).** Take the current
`config/strategy.json` backtest (`trader.backtest.run`) daily equity and compound it to
monthly returns. Regress them on SPY monthly returns, with Newey-West standard errors
(3 lags).

## Verdicts

- **Signal (primary, blind): SURVIVES** iff on S&P 400/600 **both** T1 and T2 give a
  positive estimate with p < 0.05. Otherwise **DOES NOT SURVIVE**.
- **S&P 500 signal (secondary, not blind):** the same rule, reported.
- **Strategy vs SPY (not blind): BEATS SPY** iff T3 α > 0 with p < 0.05. Otherwise
  **NOT SHOWN**.

## What each outcome means

None of them changes `config/strategy.json` automatically, and none touches the paper
account. Nothing else has passed the gate, so there is no alternative to switch to.

- **Signal survives:** the drift is real after overlap and market exposure. The weakness
  is in turning it into a portfolio: most capital sits in SPY, and the volatile names
  carry beta. Next work would be portfolio construction, pre-registered.
- **Signal does not survive:** the recorded p = 0.0009 overstated the evidence, and the
  signal is unproven. The only clean test left is out-of-time: live data from 2026-09 on.
- **Strategy vs SPY not shown:** putting real money in would be a bet the backtest does
  not support. That remains the owner's decision.
- **Recommendation, not automatic:** future gates adopt T1 and T2 in place of date-only
  clustering. That would change the mandatory gate, so it is the owner's call.
