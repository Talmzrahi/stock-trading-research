# Pre-registration — out-of-sample test of extreme-surprise drift in S&P 400/600

Written 2026-09-15, **before any S&P MidCap 400 or SmallCap 600 price, earnings, or return
data has been loaded** by this project. Committed to git before the data download so the
timestamp shows the order.

## Why this test

On the repaired S&P 500 universe the registered test (top decile, 60 sessions) failed:
+0.767pp, p=0.063. Tighter cutoffs looked stronger (top 5% +1.91pp, p=0.002), but that
was noticed *after* seeing the data, across four cutoffs — adopting it would be a
post-hoc choice. The honest check is to test that exact specification, once, on stocks
the project has never examined.

## Hypothesis

**H1:** Among point-in-time members of the S&P MidCap 400 and SmallCap 600, earnings
announcements whose price-scaled surprise is in the top 5% of the trailing year are
followed by positive 60-session returns in excess of the member's own index.

**H0:** mean excess return ≤ 0.

## Specification (fixed)

- **Universe:** point-in-time members of the S&P 400 or S&P 600, from quarterly snapshots
  of Wikipedia revision history ("List of S&P 400 companies", "List of S&P 600
  companies"), membership matched by SEC CIK where the revision carries it, otherwise
  by ticker — the same rule as `trader.events.point_in_time`, including the one-year
  forward limit for departed tickers. The period is whatever the revision history
  supports.
- **Survivorship:** data is fetched for every symbol *ever* a member, not just current
  members. A departed ticker is kept only if its price history covers ≥50% of the
  sessions it was a member and it has ≥1 earnings report inside that span (same rule as
  `research/ingest_departed.py`).
- **Data:** yfinance adjusted closes and EPS estimate/actual, stored in a separate
  database (`data/oos_midsmall.db`) so the live S&P 500 system is untouched.
- **Signal:** SUE = (actual − estimate) / close on the last session before the
  announcement day. Percentile against point-in-time member events dated strictly
  earlier within 365 days, minimum 200 — `trader.signals.sue.trailing_percentile`,
  pool = this universe.
- **Entry timing:** as `trader/events.py` — announcement before 09:30 ET → that day's
  close, otherwise the next session's close.
- **Return:** 60-session close-to-close return minus the same-period return of the
  member's index ETF (IJH for S&P 400, IJR for S&P 600), net of **20bps round trip**
  (double the S&P 500 assumption, for smaller stocks).
- **One event per firm per announcement day.**

## Feasibility notes (added before any price/earnings data was loaded)

A check of the Wikipedia pages' structure only (no returns, no earnings) found:

- "List of S&P 400 companies" has a constituent table in revisions from roughly
  2011-2014 onward, **without a CIK column** (ticker, company, sector only).
- "List of S&P 600 companies" exists only from **2018-08**; recent revisions carry CIK;
  at least one 2020 revision lists ~1,000 rows (apparently the combined S&P 1000).

Resulting fixed rules:

- **Firm identity** for rows without a CIK: the majority CIK of the same ticker across
  every snapshot of the S&P 400, 600 and 500 pages; else `SYM:<ticker>`.
- **Index label** (which ETF adjusts the return): a member of the S&P 400 snapshot at
  entry → IJH; otherwise → IJR. This resolves any combined 1,000-row table.
- The S&P 600 portion of the sample therefore starts in 2018; the S&P 400 portion starts
  at its first parseable snapshot. Both are used as-is.

## Primary test and decision rule

Events with trailing percentile ≥ 0.95. Collapse to the mean excess return per entry
date, one-sample t-test against zero.

- **PASS** iff mean > 0 **and** p < 0.05.
- **INCONCLUSIVE** if fewer than 300 qualifying events — too few to decide either way.
- **FAIL** otherwise.

## Reported but not decisive

Top 10% / 3% / 2%; eras (split into thirds by date); 40bps round trip; excess vs SPY
instead of the size ETF; S&P 400 and 600 separately; mean by percentile bucket.

## Rules for myself

- The test runs once. The specification above is not changed after results are seen.
  If a bug is found, it is fixed, disclosed next to the result, and the primary rule
  stays the same.
- **PASS** → earnings drift at the top-5% cutoff has out-of-sample support; the S&P 500
  strategy may be re-specified to top 5% and must go through the portfolio gate with a
  selection rule written before that run.
- **FAIL or INCONCLUSIVE** → earnings drift is not traded with real money in this
  project; the simulated account may keep running as an experiment.
