# Sentiment Arbitrage Trading System

AI-driven system trading the gap between market sentiment and company fundamentals.
Full context, phased plan, and the reasoning behind current priorities: see
[ROADMAP.md](ROADMAP.md) — read it before proposing what to build next.

## Current phase

**Phase 2 is built: a simulated paper-trading loop on one signal** — post-earnings
drift, which passed the date-clustered gate but is **unproven** under overlap-robust,
market-adjusted inference (below). Buy S&P 500 stocks whose price-scaled EPS surprise is in the **top
5%** of the trailing year, hold 60 sessions or until a **12-sd** volatility trailing stop,
idle cash in SPY. The VIX + earnings-beat hypothesis was rejected in Phase 1. The fusion
thesis was tested on SEC 8-K earnings-release text and **failed** its one-shot 2020-2026
holdout (2026-09-18, ROADMAP Phase 3a). Sentiment is not in the system, and that test
is closed: no retuning, no second look. ROADMAP.md has the results, the decisions behind
them, and known gaps.

**How that cutoff was settled (2026-09-15, ROADMAP Phase 2b/2c):** the original top-decile
gate failed once the survivors-only universe was repaired (+0.767pp, p=0.063), and the
tighter cutoffs that looked better were a post-hoc choice. They were re-tested
out-of-sample on S&P 400/600 stocks, pre-registered first: **PASS, +2.730pp, p=0.0009**.
The cutoff is therefore fixed by that test, not fitted on S&P 500 data. The effect is
era-dependent in both samples — expect long flat stretches.

**Re-checked 2026-09-18 (ROADMAP Phase 2e, pre-registered):** that PASS does not survive
once overlapping 60-session holds and the picks' higher beta are allowed for. On S&P
400/600, the quarter-clustered p is 0.021, but the calendar-time alpha is +2.94%/yr with
p=0.37 (β 1.36). The live strategy's alpha vs SPY is +1.06%/yr, p=0.75. Treat the
signal as unproven; the paper account is its out-of-time test. It does not pass the
upgraded gate below, so it stays in paper trading only: real money would need it to pass.

**The S&P 500 figures above predate the universe repair of 2026-09-22** (PROJECT_STATE
§8b), which filled 2012-2014 — never collected, so those events had been matched against
a three-year-stale index — and extended membership back to 2007-04. The mid/small figures
are unaffected; they come from their own database. Two things follow. **Analysis starts
2010-05**, because the newly reachable 2008-2010 window is only 54% covered by prices and
shows a calendar-time alpha of +24.49%/yr on 184 trades — survivorship, not drift.
And **`research/pead_trailing.py` now prints PASS (+1.019pp, p=0.0093)** because it has no
window guard; that verdict is the contaminated window and should not be quoted.
`research/sp1500_gate.py` carries the guard: on the clean window the S&P 500 alpha is
−0.95%/yr, 95% CI [−9.2, +7.3].

Report intervals rather than verdicts where you can. T2 resolves to roughly ±8%/yr here,
so it separates a spectacular strategy from a disastrous one and nothing in between. That
is the argument for the paper accounts; it is **not** a reason to loosen the gate, which
exists because the lenient version overstated every result it was ever applied to.

The simulated account runs on this config. **Real money and the Alpaca switch are the
owner's decisions**, not automatic (Alpaca needs paper keys in ALPACA_API_KEY /
ALPACA_SECRET_KEY).

**The validation gate is mandatory**: any new signal, exit, or sizing rule must pass the
event-level gate (`research/pead_trailing.py` pattern) and the portfolio gate
(`research/portfolio_gate.py`, selection rule pre-registered in its header) before it
reaches `config/strategy.json`. **Since 2026-09-19 (owner's decision), the event-level
gate also requires**, for anything with overlapping holding periods, a positive
estimate with p < 0.05 on both:

- **T1:** the same estimate with standard errors clustered by calendar quarter of entry
- **T2:** calendar-time, market-adjusted alpha: an equal-weighted portfolio of open
  positions, monthly returns regressed on the matched benchmark, with Newey-West SEs

Both are implemented in `research/inference_check.py` (`quarter_clustered`,
`calendar_time`, `market_alpha`). Date clustering alone overstated every earlier result
(ROADMAP Phase 2e).

## What exists

- `Main.py` — news sentiment backfill: Finnhub company news -> 3-model NLP ensemble
  (FinBERT / fintone / Twitter-RoBERTa) -> exponential time-decay weighting -> weekly
  sentiment scores in SQLite (`data/sentiment_history.db`). Run with
  `python Main.py` after setting `FINNHUB_API_KEY`. Note this pipeline has never been
  run for real — the research work below uses its own tables.
- `Dataframe.py` — exports the `weekly_summary` table to CSV.
- `research/` — Phase 1 work (`ingest.py`, `backtest.py`, `sentiment_events.py`,
  `fusion_test.py`, `archive.py`), PEAD re-specification (`pead.py`, `universe.py`,
  `pead_trailing.py`), the strategy selector (`portfolio_gate.py`), and the v2 sentiment
  test (`edgar_filings.py` → `data/edgar.db`, `edgar_features.py`, `sentiment_gate.py`;
  design and result in `design_sentiment_v2.md`), and the v3 sentiment work
  (`v3_layer0.py`, `v3_labels.py`, `v3_readers.py`, `v3_readers_eval.py`,
  `v3_prototype.py`, `v3_signal_probe.py`, `v3_fit.py`; design in
  `design_sentiment_v3.md`).
- `trader/` — the trading system, one code path for backtest and live: `events.py` →
  `signals/` → `fusion.py` → `exits.py` / `portfolio.py` → `engine.py` → `broker.py`
  (simulated MOC ledger) → `state.py` (`data/trader.db`) → `monitor.py` / `report.py`.
  `run_daily.py` is the daily entry point; `refresh.py` keeps research.db current.
- `trader/text/` — the text signal in production: `fetch.py` (EDGAR, rate-limited),
  `parse.py` (exhibit HTML → paragraphs and rows), `novelty.py` (what is new against the
  company's own past releases), `readers.py` (three free models), `features.py`,
  `model.py` (loads `config/text_model.{json,npz}`), `pipeline.py` (today's releases,
  end to end), `store.py`. `trader/shadow.py` trades its scores on separate books in
  `data/shadow.db`; scores land in `trader.db` `text_scores`, written once.
- `config/strategy.json` (gate-selected parameters + evidence),
  `config/tripwire_band.csv` — regenerated by `research/portfolio_gate.py` — and
  `config/text_model.{json,npz}`, written by `research/v3_fit.py`. All committed.
- `research/mechanics.py` — the shared backtest arithmetic (holding windows, trailing
  thresholds, position books, levered returns, performance), each function guarding
  against an error this project actually made; `tests/test_mechanics.py` holds the
  regressions. New work uses it rather than rewriting it.
- Volatility targeting (a separate research track, 2026-09-22 to 09-24): `vol_ingest.py`
  → `data/vol.db`, `voltarget_*.py`, `bear_regimes.py`, `noise_search.py`;
  pre-registrations in `research/prereg_voltarget*.md`, results in PROJECT_STATE §8d-8f.
- `scripts/install_task.ps1` registers the weekday 14:30 ET run in Task Scheduler.

**The text signal is in shadow mode.** It reads every S&P 500 release each day, scores
it, and trades a separate simulated $1,000 account, so it builds its own out-of-time
record. It does not affect the live account and is not in `config/strategy.json`: on all
23,997 development releases its top-minus-bottom spread is +1.12pp at 60 sessions, but that
is the bottom falling; the long-only top 5% earns −0.051pp per trade and fails the gate
(PROJECT_STATE §3). `--text off` skips it.

Not built: a validated second signal (so fusion is still a pass-through for the live
account), non-standard signals, numerical stream beyond EPS.

## Commands

Use the venv interpreter — the system `python` has no pandas.

```
.venv\Scripts\python.exe -m trader.run_daily --dry-run      # decide + report, save nothing
.venv\Scripts\python.exe -m unittest discover -s tests -t .  # tests (stdlib unittest)
.venv\Scripts\python.exe research\portfolio_gate.py         # re-select strategy + tripwire band
.venv\Scripts\python.exe research\v3_readers.py             # score releases with the readers
.venv\Scripts\python.exe research\v3_fit.py                 # refit config/text_model.*
```

Reports go to `reports/` (gitignored). Deleting `data/trader.db` restarts the paper
account.

## Data

`data/research.db` (~183MB, gitignored) holds prices, VIX, earnings and 41,569 scored
news articles. Prices/VIX/earnings are re-downloadable; **the news articles are not** —
Finnhub's free window is a rolling ~12 months, so they age out permanently. They are
archived to `data_archive/*.csv.gz` (committed); restore with
`python research/archive.py restore`, then rerun `research/ingest.py` for prices/VIX.

## Conventions

- Secrets via environment variables only — never hardcode API keys. A Finnhub key was
  previously committed in plaintext in `Main.py`; treat it as compromised and rotated,
  don't reuse it.
- SQLite for storage until it's an actual bottleneck — don't reach for a bigger DB
  preemptively.
- Free data sources only for now (see ROADMAP.md's guiding decisions) — check before
  adding a paid data dependency.
- Target paper-trading venue: Alpaca's paper trading API — **built** (`trader/alpaca.py`,
  tests in `tests/test_alpaca.py`), selected with `--broker alpaca` or `TRADER_BROKER`.
  It implements `trader.broker.Broker`, sends market day orders shortly before the close
  (Alpaca allows fractional/notional only with `time_in_force=day`), and refuses a
  non-paper URL by design. The simulated broker remains the default; switching is the
  owner's decision and needs ALPACA_API_KEY / ALPACA_SECRET_KEY.
- Runs on Windows locally (PowerShell). Keep scripts cross-platform-safe where it's
  easy (e.g. explicit UTF-8 stdout so emoji in status output doesn't crash on Windows'
  default console encoding), but don't over-engineer for platforms not in use.
