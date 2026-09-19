# Roadmap

Trading system that fuses numerical data, non-standard/alternative signals, and news
sentiment to trade the gap between market sentiment and company fundamentals.

**Where it stands (2026-09-18):** a simulated paper-trading loop is built for one
signal — post-earnings drift on unusually large EPS surprises. It is a
low-frequency, long-only, event-driven strategy: buy S&P 500 stocks whose price-scaled
surprise is in the top 5% of the trailing year, hold 60 sessions or until a 12-sd
volatility trailing stop, keep idle cash in SPY. **That signal is now unproven:** it
passed the date-clustered gate but not overlap-robust, market-adjusted inference
(Phase 2e). The strategy is not shown to beat SPY after market exposure. The
original VIX-regime idea was rejected (Phase 1). The fusion thesis was tested on SEC
earnings-release text and **failed** its pre-registered holdout (Phase 3a), so
sentiment stays out of the system.

This file is the source of truth for sequencing. Update it as phases complete or
priorities change; don't let it go stale.

## Guiding decisions

2026-09-14:

- **Prove the core hypothesis before building more architecture.**
- **Target is paper trading**, not live capital, for the foreseeable future.
- **Free data sources only** until a signal proves useful enough to justify paying for
  more.

2026-09-15 (Q&A before building Phase 2):

- **The validation gate is mandatory.** No signal, exit rule, or sizing rule reaches the
  live system without passing it: date-clustered inference, point-in-time universe,
  pre-registered direction and selection rule, positive control, threshold grid with
  neighbour robustness, costs. Everything rejected so far was rejected here.
- **Gate objective is net total return (CAGR).** The project is profit-first and accepts
  high risk: concentration through a tighter entry cutoff, not diversification limits.
  No position-weight, sector, or max-count caps.
- **Must work at tiny size.** The eventual real stake is ~$100, so sizing is in dollars
  (fractional shares), not whole shares.
- **Market-crash exits are out** (exiting after a crash is pointless; *predicting* one
  would be a signal and would face the gate). **Sentiment exits are parked** until the
  NLP ensemble is shown to be reliable.
- **Tripwire alerts, never acts.** A human decides whether the signal has died.

## Phase 0 — Done

- News sentiment backfill ([Main.py](Main.py)): Finnhub company news -> 3-model NLP
  ensemble (FinBERT / fintone / Twitter-RoBERTa) -> exponential time-decay weighting ->
  weekly sentiment scores in SQLite (`data/sentiment_history.db`). Never run for real.
- Export pipeline ([Dataframe.py](Dataframe.py)).

## Phase 1 — Validate the core hypothesis — DONE (2026-09-14)

Built: [research/ingest.py](research/ingest.py), [research/backtest.py](research/backtest.py),
[research/sentiment_events.py](research/sentiment_events.py),
[research/fusion_test.py](research/fusion_test.py), [research/archive.py](research/archive.py).

### Result 1 — the VIX + earnings-beat hypothesis is REJECTED

42,191 earnings events, 504 stocks, 2002-2026. The beat x elevated-VIX interaction is
+0.013 / +0.074 / +0.009 / -0.051pp at 1/3/5/10d (p 0.50-0.94), and p ran 0.344-0.975
across all 12 threshold combinations. A precisely estimated zero. PEAD (+0.124pp/5d,
p=0.003) and the VIX regime effect are additive, not synergistic.

### Result 2 — the fusion thesis is UNTESTED, not disproven

41,569 pre-earnings articles across 1,940 events; 5-day beat x sentiment interaction
+0.243pp, 95% CI [-0.269, +0.755]. The 12-month sample fails its positive control (PEAD
comes out -0.285pp inside it), so it cannot adjudicate anything. The limit is the data
window. (Later tested properly on 8-K text: FAIL, see Phase 3a.)

### Data constraints discovered (expensive to rediscover — check here first)

- **Finnhub free news: rolling ~12 months**, so fetched news is perishable — archived
  to `data_archive/`.
- **Finnhub free earnings: 4 quarters only**; no revenue estimates.
- **yfinance: EPS estimate/actual back to ~2002**, 100 quarters per ticker. Prices and
  `^VIX` to 1980/1990. Free, no key.
- **Revenue estimates are not available free anywhere** — everything is EPS-only.
- Wikipedia 403s on urllib's default user-agent; fetch via requests.
- **Wikipedia's S&P 500 revision snapshots parse only from 2010-04**, with a gap from
  2011-10 to 2014-10 (the 2011-10 list stands in for those three years).
- **yfinance does not serve delisted tickers** (and some, like SHLD, return a different
  company). Survivorship on the losing tail needs a paid dataset such as CRSP.
- yfinance earnings timestamps are hour-rounded; 0.11% are midnight (no time) — too few
  to matter for entry timing.

### Methodology lessons worth keeping

- **Test the interaction, not the contrast.**
- **Cluster on dates, not events.**
- **Survivorship bias is asymmetric**, and look-ahead *inclusion* (trading firms before
  they joined the index) was a bigger contaminant than survivorship: 23.5% of events.
- **Run a positive control.**
- **Pre-register the direction — and the selection rule** when choosing among
  parameters.
- **Ranking breakpoints must be knowable on the day.** Cutting deciles within a calendar
  quarter ranks week-1 reporters against week-6 reporters.

## Phase 1b — PEAD re-specified — DONE (2026-09-15)

[research/pead.py](research/pead.py), [research/universe.py](research/universe.py),
[research/pead_trailing.py](research/pead_trailing.py).

- Price-scaled SUE `(actual - estimate) / price` instead of a binary >2% beat (which
  discarded magnitude and exploded on near-zero estimates); horizons to 60 days.
- Point-in-time S&P 500 membership from Wikipedia revision history. Removing
  not-yet-members cut decile 10 @60d from +2.19pp to +1.32pp and erased the "big misses
  bounce back" anomaly entirely.
- **Trailing cutoffs (pre-registered gate): PASS.** Each event ranked only against the
  prior 365 days. Top decile 60d **+1.225pp net, p=0.003**, 2010-05 → 2026-06; 90%
  overlap with the quarterly-cutoff top decile.

| era       | top decile 60d net | p     |
|-----------|--------------------|-------|
| 2010-2013 | +1.636pp           | 0.040 |
| 2014-2019 | -0.507pp           | 0.308 |
| 2020-2026 | +2.405pp           | 0.001 |

Cutoff grid (60d net): top 10% +1.225, top 5% +2.094, top 3% +1.860, top 2% +2.892 — all
p ≤ 0.019. Not monotone inside the top decile (the 97th-98th percentile bucket is weak),
which is why the cutoff choice went to the portfolio gate.

## Phase 2 — Paper-trading loop — BUILT (2026-09-15), simulated broker

### Architecture — `trader/`, one code path for backtest and live

| layer | module | status |
|---|---|---|
| 1 Data | [data.py](trader/data.py), [refresh.py](trader/refresh.py), [events.py](trader/events.py), [market_calendar.py](trader/market_calendar.py) | ✅ live refresh: universe weekly, prices daily (full monthly), earnings for due symbols daily + all weekly |
| 2 Signal | [signals/base.py](trader/signals/base.py), [signals/sue.py](trader/signals/sue.py) | ✅ one signal (SUE, trailing percentile) |
| 3 Validation | [research/pead_trailing.py](research/pead_trailing.py), [research/portfolio_gate.py](research/portfolio_gate.py) | ✅ event gate + portfolio gate |
| 4 Fusion | [fusion.py](trader/fusion.py) | ✅ interface; pass-through at n=1, agreement flag only |
| 5 Portfolio | [portfolio.py](trader/portfolio.py), [exits.py](trader/exits.py) | ✅ fixed fraction at entry, adaptive slots, idle cash in SPY; time exit, vol trailing stop |
| 6 Execution | [broker.py](trader/broker.py), [state.py](trader/state.py) | ✅ simulated MOC broker in `data/trader.db` · ⬜ Alpaca |
| 7 Monitoring | [monitor.py](trader/monitor.py), [report.py](trader/report.py), [run_daily.py](trader/run_daily.py) | ✅ daily report, bootstrap tripwire |

Engine rules ([engine.py](trader/engine.py)): the step for session t sees closes through
t-1 and events whose entry session has arrived, and submits market-on-close orders for
close t. Entry: announcement before 09:30 ET → that day's close, otherwise next session's.
Late signals (PC off, EPS posted late) are taken up to 2 sessions late. A held stock that
re-triggers has its 60-session clock reset. Slots = qualifying signals over the trailing
year x 60/252, min 5; each entry gets equity/slots, scaled down (never financed) when a
busy day exceeds cash + SPY.

### Strategy selection — [research/portfolio_gate.py](research/portfolio_gate.py)

$1,000 start, 10bps round trip on stocks, 1bp/side on SPY, 2011-05-09 → 2026-09-14.
Grid of 4 cutoffs x 5 stops; selection rule written in the script header before the
first run. **All 20 configurations beat SPY** (15.5%-21.7% CAGR vs 13.9%).

| config | CAGR | max DD | Sharpe | trades/yr | held |
|---|---|---|---|---|---|
| top 10%, no stop (baseline) | +16.10% | -48.3% | 0.84 | 117 | 30.4 |
| top 5%, no stop | +18.84% | -50.3% | 0.90 | 59 | 15.1 |
| top 2%, no stop | +18.99% | -49.1% | 0.89 | 23 | 5.9 |
| **top 2%, stop 8 sd (selected)** | **+21.74%** | **-39.4%** | **1.01** | **24** | **5.2** |
| SPY | +13.89% | -33.7% | | | |

Selected: `cutoff=0.98`, `stop_k=8` → [config/strategy.json](config/strategy.json).
Eras: 2011-2013 +13.42%, 2014-2019 +15.28%, 2020-2026 +31.65%. Mean alpha +3.74%/trade,
hit rate 51%, 111 of 361 exits by stop. Losing years vs SPY: 2011, 2012, 2018, 2019.

**Read with care:** the selected cell is the best of 20, so expect nearer the ~19% of
its neighbours. Delisted firms are missing, which flatters a 5-position portfolio more
than a 30-position one. 2026 YTD (+44.5%) is doing a lot of work in the recent era.

### Tripwire — [config/tripwire_band.csv](config/tripwire_band.csv)

Date-clustered bootstrap of the 361 backtest trades: fire when the running mean alpha
after N closed trades is below the 5th percentile (N=10: -4.63%, N=50: -0.53%, N=100:
+0.72%). Replayed on history: started 2011 → false alarm at trade 11 (2012); started
2014 → fired at trade 124 (Feb 2020); started 2020 → never. **At ~24 trades a year it
takes years to detect a dead signal** — that is the cost of concentration.

### Running it

```
.venv\Scripts\python.exe -m trader.run_daily --dry-run   # decide + report, save nothing
.venv\Scripts\python.exe -m trader.run_daily             # the real daily run
powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1   # weekdays 14:30 ET
.venv\Scripts\python.exe research\portfolio_gate.py      # re-select + rebuild band
```

Reports land in `reports/YYYY-MM-DD.md` (gitignored); the account lives in
`data/trader.db` — delete it to restart the paper account.

**Status:** simulated account opened 2026-09-15 with $1,000 (first order: SPY at that
day's close). Task Scheduler task "PEAD paper trading daily run" registered for weekdays
14:30 ET. Live data refresh verified end to end the same day.

### Known gaps

- **Alpaca not integrated.** Checked against Alpaca's docs (2026-09-15): fractional and
  notional orders accept only `time_in_force` `day` or `gtc`, so **market-on-close
  (`cls`) is unavailable for a fractional account** — use a day market order a few
  minutes before the close and track slippage against the close. Accounts under $2,000
  equity get 1x buying power (no leverage, no shorting), but can trade on unsettled
  funds, so same-day sells can fund buys as the simulated broker assumes.
  Sources: [fractional trading](https://docs.alpaca.markets/us/docs/fractional-trading),
  [margin](https://docs.alpaca.markets/us/docs/margin-and-short-selling),
  [cash accounts](https://alpaca.markets/support/alpaca-cash-accounts).
- Scheduled time drifts an hour for the weeks when US and local DST disagree (still
  before the cutoff); 13:00 early-close days run after the 12:50 cutoff and roll a day.
- Older adjusted prices drift by new dividends between monthly full refreshes (affects
  backtests at the 400-day boundary, not live signals).
- The tripwire band is built from the history the strategy was selected on, so it is if
  anything optimistic — it fires early, the safe direction.

## Phase 2b — Stress test and survivorship repair (2026-09-15) — GATE NOW FAILS

### Walk-forward on the original (survivors-only) data — [research/stress_test.py](research/stress_test.py)

Re-selecting the configuration each year from prior years only, 2014 → 2026-09:

| | CAGR | max DD |
|---|---|---|
| walk-forward selection | +14.55% | -48.3% |
| hindsight pick (top 2% / stop 8) | +23.50% | -39.4% |
| baseline (top 10% / no stop) | +14.92% | -48.3% |
| all 20 configurations averaged | +18.04% | -43.2% |
| SPY | +13.66% | -33.7% |

Pre-registered verdict PASS (beats SPY), but only **9% of the excess return survives**
without hindsight; the rule never picked the live configuration in 13 years, and
averaging all 20 beat it. Costs are not the problem (50bps round trip: 21.7% → 20.2%).

### The universe was survivors-only

`research/ingest.py` fetched data for *today's* constituents only. Of 880 symbols ever
in the index, 377 had left and had no data, so point-in-time backtests saw 52% of the
index in 2010 and 78% in 2020.

- [research/ingest_departed.py](research/ingest_departed.py): 100 departed firms
  recovered (history must cover ≥50% of their member sessions); 207 have no Yahoo data
  (acquired/delisted — 0 recovered on a one-at-a-time retry); 42 tickers now belong to
  someone else or are truncated; 28 have prices but no earnings.
- [research/universe_ids.py](research/universe_ids.py): membership matched by SEC CIK
  through ticker renames (BK→BNY lost 16 years; 59 multi-ticker firms incl. BLL→BALL,
  ABC→COR), majority CIK per ticker to survive bad revisions, and a departed ticker
  can't carry its firm forward (AA → Alcoa spin-off). Shared by the gate, backtest and
  live system via `trader.events.point_in_time`.

### Event gate re-run on the repaired universe: **FAIL**

26,464 point-in-time events, 588 firms, 2010-05 → 2026-06. Top decile 60d
**+0.767pp net, p=0.063** (was +1.225pp, p=0.003). Eras: 2010-13 +1.50pp (p=0.036),
2014-19 -0.53pp (p=0.33), 2020-26 +1.54pp (p=0.041). Cutoff grid: top 5% +1.91pp
(p=0.002), top 3% +1.58pp (p=0.039), top 2% +1.96pp (p=0.052). Inside the top decile,
the 90th-95th percentile bucket is -0.09pp — the drift lives in the extreme tail, but
choosing a tighter cutoff now would be post-hoc and unregistered.

Per the pre-registered rule the build stopped there, pending the out-of-sample test
below. Still missing and not fixable for free: 207 acquired/bankrupt S&P 500 firms.

## Phase 2c — Out-of-sample test PASSES; strategy re-specified (2026-09-15)

Pre-registered in [research/prereg_midsmall.md](research/prereg_midsmall.md) **before any
S&P 400/600 data was loaded** (commit bc09e15), because the stronger tighter-cutoff
result on S&P 500 data was noticed after the fact and could not adjudicate itself.

### The test — [research/oos_midsmall.py](research/oos_midsmall.py)

Universe built by [research/oos_midsmall_data.py](research/oos_midsmall_data.py) in its
own `data/oos_midsmall.db`: 62 quarterly S&P 400 snapshots (2011-2026) and 32 of the
S&P 600 (2018-2026, the page's whole life), 1,915 tickers ever a member, firm identity by
CIK, data for departed tickers kept only where history covers ≥50% of their member
sessions. 1,176 symbols usable; 543 have no Yahoo data at all (SIVB, IDTI, WWAV … —
genuinely delisted, confirmed by direct checks), so this sample is still survivor-tilted.

**PASS: top 5% of trailing SUE, 60 sessions, net 20bps, vs IJH/IJR — +2.730pp,
p=0.0009**, 1,480 events on 748 dates, 1,152 firms. (Re-checked in Phase 2e: it does
not survive overlap-robust, market-adjusted inference.)

| cut | n | vs ETF | @40bps | vs SPY |
|---|---|---|---|---|
| top 10% | 2,946 | +2.605pp (p=0.008) | +2.405pp | +1.930pp (p=0.058) |
| top 5% | 1,480 | +2.730pp (p=0.001) | +2.530pp | +2.140pp (p=0.016) |
| top 3% | 908 | +3.590pp (p=0.001) | +3.390pp | +3.038pp (p=0.008) |
| top 2% | 610 | +4.785pp (p=0.001) | +4.585pp | +4.138pp (p=0.007) |

S&P 400 +1.94pp (p=0.045), S&P 600 +3.59pp (p=0.003). **Era-dependent again**: by thirds,
+0.78pp (p=0.37), +6.48pp (p=0.001), +0.94pp (p=0.42) — the effect is real but lumpy.

### Re-specification — [research/prereg_portfolio_top5.md](research/prereg_portfolio_top5.md)

Also written before the result was known: cutoff **fixed** at top 5% (not re-optimised),
only the stop chosen, by the existing neighbour rule, on the repaired S&P 500 universe
(`python research/portfolio_gate.py 0.95`).

| stop | CAGR | max DD | Sharpe | tr/yr | held |
|---|---|---|---|---|---|
| none | +15.11% | -49.2% | 0.76 | 67 | 17.0 |
| 3 sd | +16.78% | -36.5% | 0.93 | 79 | 7.2 |
| 5 sd | +14.53% | -39.1% | 0.80 | 72 | 11.3 |
| 8 sd | +16.62% | -43.9% | 0.84 | 68 | 14.9 |
| **12 sd (selected)** | **+16.63%** | **-46.0%** | **0.82** | **68** | **16.6** |

3 sd scored highest but its neighbour (5 sd) is worse than no stop, so the robustness
rule rejected it; 8 sd failed the same way. **Live config: `cutoff=0.95`, `stop_k=12`**
(config/strategy.json), SPY +13.90% over the same window, +1.76pp alpha per trade.

Walk-forward with the cutoff fixed: **+15.15% vs SPY +13.66%, 51% of the excess return
survives** (9% when the cutoff was also being fitted). Costs are minor: 50bps round trip
still gives +15.06%. The tripwire band was rebuilt (1,040 trades); it would have fired
early from a 2011 or 2020 start, so read it as noisy before ~50 closed trades.

`trader/alpaca.py` is wired into `run_daily --broker alpaca` (paper-only, fake-API
tested) but switched off: it needs the owner's Alpaca paper keys, and real money remains
a Phase 4 decision.

## Phase 2d — Survivorship quantified, and where the edge actually lives (2026-09-16)

[research/survivorship_test.py](research/survivorship_test.py) — a diagnostic, not a
selection rule; it changes nothing in `config/strategy.json`.

### The missing-firms hole is much smaller than assumed

Wikipedia's index-change history (1,223 removals with a stated reason, from
"Historical components of the S&P 500" plus the 400/600 pages) says *why* each missing
firm left:

| why they left | share | examples |
|---|---|---|
| acquired | 52.7% | ADT, AET, ANDV, BMC, CBE |
| unknown | 25.6% | ACT, FO, LB, MHP, NU |
| demoted | 15.2% | BTU, RSH, BBBY, LEG, ADS |
| restructured | 5.4% | AIV, NE, SVU, ARNC |
| failed | 1.1% | SBNY, FRC, SIVB |

**Over half were acquisitions**, which close at a premium — those are missing *winners*,
pushing the bias opposite to the usual worry. Only 1% failed outright. 16.3% of
member-time is unobservable overall (38% in 2010, 1% in 2026), implying ~203 unseen
trades against 1,040 observed.

Cost to the backtest, with each position ~4.9% of equity and ~13 unseen trades a year:

| assumption | mean alpha/trade | portfolio drag |
|---|---|---|
| like firms that did leave (+0.55pp, empirical) | +1.76 → +1.56pp | **-0.78pp/yr** |
| mix by stated reason | +1.76 → +1.67pp | **-0.33pp/yr** |
| every missing trade -25% | +1.76 → -2.61pp | -17.2pp/yr |
| every missing trade -50% | +1.76 → -6.69pp | -33.3pp/yr |

The last two rows are bounds, not forecasts — they are ruled out by the classification
above. The honest number is **0.3-0.8pp a year**.

Nor is the edge propped up by doomed firms: trades in companies that left the index
within a year earned +0.55pp (p=0.44) against +1.94pp (p=0.003) for the rest — less, but
not negative, and the difference is not significant (p=0.507).

### The real limitation: the edge exists only in volatile stocks

| volatility at entry | n | mean alpha | p |
|---|---|---|---|
| low | 347 | -0.10pp | 0.846 |
| mid | 346 | +0.49pp | 0.830 |
| **high** | 347 | **+4.88pp** | **0.001** |

Same story by share price (low-priced +3.01pp, p=0.030; high-priced +1.04pp, p=0.46).
Excess return is measured against SPY, so high-beta names would beat it in a rising
market regardless — the control rules that out:

| volatility | top 5% events | all other events | difference |
|---|---|---|---|
| low | -0.86pp (n=217) | -0.31pp (n=8,605) | -0.55pp (p=0.440) |
| mid | -0.02pp (n=373) | -0.88pp (n=8,448) | +0.86pp (p=0.204) |
| high | +3.06pp (n=688) | +0.06pp (n=8,134) | **+3.00pp (p=0.001)** |

Ordinary earnings events in equally volatile stocks earn +0.06pp, so this is the signal,
not beta. But it means **two thirds of the universe contributes nothing**, and the
strategy is structurally a bet on volatile names — which is why the drawdown is -46%
against SPY's -34%.

Tempting and **not adopted**: filtering entries to high-volatility names. That was found
by looking at the data, so adopting it now would be exactly the post-hoc choice the gate
exists to prevent. It would need pre-registering and testing on data not used for it.

## Phase 2e — The PEAD evidence under overlap-robust inference — DOES NOT SURVIVE (2026-09-18)

Pre-registered in [research/prereg_inference.md](research/prereg_inference.md) (`fbddc02`),
script [research/inference_check.py](research/inference_check.py) committed before it ran.

Every gate so far clustered by entry date, but positions are held 60 sessions, so trades
entered days apart share most of their returns. This re-check changed only the statistics
and reproduced both original results exactly before doing anything else.

| | S&P 400/600 (primary, blind) | S&P 500 |
|---|---|---|
| original, date-clustered | +2.730pp, p=0.0009 | +1.912pp, p=0.002 |
| same estimate, quarter-clustered | p=**0.021** | p=0.073 |
| calendar-time α vs matched ETF | **+2.94%/yr, p=0.37**, β 1.36 | −0.90%/yr, p=0.83, β 1.29 |

The live strategy vs SPY, monthly: raw excess +3.30%/yr, but **α +1.06%/yr, p=0.75**,
β 1.16.

**Verdict: the signal is unproven, not disproven.** Point estimates on 400/600 are still
positive. But after allowing for overlapping holds and the extra market exposure of the
stocks it picks, 15 years of lumpy returns cannot separate it from zero. Nothing in
`config/strategy.json` changes. The paper account now serves as an out-of-time test, which
is the only clean test left, and it only works if the account actually runs.

Post-hoc diagnostics, which cannot change the verdict:
- Most of the gap is **beta**. With β forced to 1, 400/600 shows +7.2%/yr (p=0.065).
- Top-5% surprises **bunch in time**: 2020 alone holds 17% of 400/600 events, at +12.5pp.
- On the S&P 500, **date-weighting flattered the estimate**. Lone-date events averaged
  +3.2pp; the rest +0.2pp.

Lesson worth keeping: **with overlapping holding periods, date clustering is not enough.
Test the calendar-time, market-adjusted return too.** It is the recommended addition to
the mandatory gate; adopting it is the owner's decision.

## Phase 3a — Fusion test on SEC earnings-release text — FAIL (2026-09-18)

Design, registration and full result: [research/design_sentiment_v2.md](research/design_sentiment_v2.md).
This replaced v1 news sentiment, which could not be tested on a 12-month window, with the
8-K press release every company files when it reports. Question: does the company's own
framing of the quarter predict the 60-session return **that the earnings surprise does
not already explain**?

- [research/edgar_filings.py](research/edgar_filings.py) — 26,358 releases cached in
  `data/edgar.db` (97.6% of point-in-time events matched). Not archived; re-downloadable
  from EDGAR.
- [research/edgar_features.py](research/edgar_features.py) — eight features: tone vs own
  norm, three guidance measures, language change vs the previous release and the same
  quarter last year (*Lazy Prices*), filing latency vs own habit, non-GAAP emphasis.
- [research/sentiment_gate.py](research/sentiment_gate.py) — ridge on the surprise residual,
  fitted 2012-2019. The 2020-2026 holdout was read exactly once.

| | train, in-sample | **holdout** |
|---|---|---|
| top − bottom text-score quintile | +0.855pp (p=0.009) | **+0.431pp (p=0.384)** |
| correlation with unexplained return | +0.036 | **+0.003** |
| top-5% trades, better − worse half | — | **−0.84pp (p=0.618)** |

11,284 holdout events and a clean zero. The in-sample pattern was a fit to noise. The
fusion thesis on company-authored text is **unsupported**, and it is not retuned.

Five bugs were found and fixed *before* the look (disclosed in the design doc). Any of them
could have produced a false pass or wasted the holdout: the verdict tests were not
date-clustered; 1.4% of filings post-dated the trade decision; there was no train/holdout
embargo; the own-history z-score silently needed 12 prior releases instead of 6; and a zero
MAD was treated as missing data.

Lesson worth keeping: **check a feature's coverage by group before trusting a sample
size.** A normaliser that silently returns NaN shrinks the sample without any error, and
here it did that to a third of events.

## Phase 3b — Sentiment v3, a layered reader — IN PROGRESS (from 2026-09-18)

Design, the owner's decisions, and every choice made along the way:
[research/design_sentiment_v3.md](research/design_sentiment_v3.md).

A measured review of v1 (the news ensemble) found that its temperature and bias
adjustments are cosmetic (rank correlation 0.993-0.999 with untuned outputs). The real
problems: nothing in it was ever fitted to prices, and its news sentiment tracks the
stock's own prior 10-day return (+0.14) rather than the surprise (−0.002) or the
announcement move (−0.020). v3 therefore:

- **reads earnings releases, not news.** They are timestamped before the reaction and
  have 15 years of history.
- **lets the market label the text:** the target is the announcement reaction, not human
  sentiment labels.
- **trades the gap:** text-implied reaction minus actual reaction. The trade enters at
  the next close, because the 14:30 ET decision cannot see that day's close.
- **is built in layers,** starting with layer 0: each sentence and table row is labelled
  boilerplate / template / edited / new against the company's previous 4 releases.

Built: `research/release_text.py` (structure-preserving parser), `edgar_filings.py`
(`--html-only`, `--set midsmall`), `v3_layer0.py`, `v3_labels.py`, and `v3_prototype.py`
(a cheap end-to-end pass on development data). **Exam set:** the S&P 400/600 releases,
in their own `edgar_midsmall.db` / `v3_midsmall.db`, used once, after a pre-registration.

Compute constraint for layer 1: FinBERT reads about 18 sentences/s on this CPU. Roughly
1.3M new and edited sentences would take about 20 hours per model, so layer 1 needs a
small reader or the big-teaches-small route.

## Phase 3 — Add breadth to the fusion layer

Goal: a second signal that passes the gate, so fusion does real work.

- Every candidate goes through the event-level gate (like `pead_trailing.py`) *and* the
  portfolio gate before touching `config/strategy.json`.
- Candidates, free sources only: multi-year news for the fusion test (GDELT); attention
  proxies (same-day announcement counts, news volume); congressional/insider trading
  (House/Senate stock-watcher); prediction markets (Kalshi/Polymarket). SEC 8-K
  earnings-release language was tested in Phase 3a and failed. 10-K/10-Q language
  (where *Lazy Prices* was originally shown) is untested, and the cached-download
  machinery in `edgar_filings.py` would carry over.
- Conflict handling ("trade agreement, hold back on conflict") is a rule in its own right
  and needs testing once two signals exist.
- Deciles 8-10 at 20 days and a broader universe (S&P 400/600, where drift is typically
  stronger) are untested variants of the existing signal.

## Phase 4 — Live-trading readiness (explicitly deferred)

Not started until Phase 2 has run clean in paper trading for a meaningful stretch —
given the tripwire's speed, measured in quarters, not weeks. The intended real stake is
~$100. Will need Alpaca integration, compliance/tax considerations, and alerting.

## Working conventions

- No hardcoded secrets — environment variables only.
- SQLite for storage until it's an actual bottleneck.
- Prefer free data sources; only reach for paid ones when a free signal has already
  shown enough promise to justify it.
