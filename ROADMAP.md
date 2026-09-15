# Roadmap

Trading system that fuses numerical data, non-standard/alternative signals, and news
sentiment to trade the gap between market sentiment and company fundamentals.

**Where it stands (2026-09-15):** a simulated paper-trading loop is built and trades one
validated signal — post-earnings drift on unusually large EPS surprises. It is a
low-frequency, long-only, event-driven strategy: buy S&P 500 stocks whose price-scaled
surprise is in the top 2% of the trailing year, hold 60 sessions or until a volatility
trailing stop, keep idle cash in SPY. The original VIX-regime idea was rejected (Phase
1); sentiment is not in the system because the fusion thesis remains untested.

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
window.

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

## Phase 3 — Add breadth to the fusion layer

Goal: a second signal that passes the gate, so fusion does real work.

- Every candidate goes through the event-level gate (like `pead_trailing.py`) *and* the
  portfolio gate before touching `config/strategy.json`.
- Candidates, free sources only: multi-year news for the fusion test (GDELT); attention
  proxies (same-day announcement counts, news volume); congressional/insider trading
  (House/Senate stock-watcher); prediction markets (Kalshi/Polymarket); SEC filing
  language via EDGAR full-text search.
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
