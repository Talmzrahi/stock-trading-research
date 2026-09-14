# Roadmap

Trading system that fuses numerical data, non-standard/alternative signals, and news
sentiment to trade the gap between market sentiment and company fundamentals. Full
concept: the flagship idea is that when macro fear is high (VIX well above its trailing
average) and a company beats both earnings and revenue estimates, the market is too
busy being afraid to price the good news in — creating a short-window mispricing.

This file is the source of truth for sequencing. Update it as phases complete or
priorities change; don't let it go stale.

## Guiding decisions (2026-09-14)

- **Prove the core hypothesis before building more architecture.** Everything else in
  the original vision (fusion layer, regime/ranking/allocation/execution) is only worth
  building on top of a signal that's actually been shown to work in this codebase, not
  just asserted.
- **Target is paper trading**, not live capital, for the foreseeable future. Design
  execution around a paper-trading broker API, not real order routing.
- **Free data sources only** until a signal proves useful enough to justify paying for
  more. Applies mainly to the non-standard/alternative-data stream (congressional
  trading and prediction markets have workable free APIs; satellite/foot-traffic data
  is generally paid and stays deferred).

## Phase 0 — Done

- News sentiment backfill ([Main.py](Main.py)): Finnhub company news -> 3-model NLP
  ensemble (FinBERT / fintone / Twitter-RoBERTa) -> exponential time-decay weighting ->
  weekly sentiment scores in SQLite (`data/sentiment_history.db`).
- Export pipeline ([Dataframe.py](Dataframe.py)): `weekly_summary` table -> CSV.
- Fixed for local execution (2026-09-14): removed Colab-only code (Drive mounting,
  `google.colab.files.download`, in-notebook `pip install`), moved the Finnhub key to
  an environment variable, fixed a Windows console UTF-8 crash, dropped unused deps.

This covers roughly one-third of one of the three intended data streams (news
sentiment only — no earnings-call tone or SEC filing language yet). Nothing else from
the original architecture exists in code yet: no numerical/financials stream, no
non-standard signals, no fusion layer, no regime/ranking/allocation/execution, no
scheduling.

## Phase 1 — Validate the core hypothesis (current)

Goal: reproduce the VIX-regime + earnings-beat signal as real, runnable, testable code
— not just a claim — before anything else gets built on top of it.

1. **Historical data ingestion**
   - OHLCV price history for the stock universe — yfinance (free, no key)
   - VIX historical series + trailing average — yfinance `^VIX` or FRED `VIXCLS` (free)
   - Historical earnings surprises (EPS + revenue, actual vs. estimate) — Finnhub
     earnings-surprises endpoint first (key already in hand), yfinance earnings history
     as a fallback if free-tier limits bite
   - Store alongside the existing sentiment DB (SQLite) — no new infra needed yet
2. **Signal construction** — regime flag (VIX vs. trailing average, thresholded),
   beat flag (EPS actual > estimate AND revenue actual > estimate), combined into an
   event list of `(stock, date, regime_flag, beat_flag)`.
3. **Backtest engine** — forward returns at multiple horizons (1/3/5/10 trading days)
   from each event, compared across: regime+beat vs. beat-only vs. regime-only vs.
   baseline. Robustness battery:
   - out-of-sample holdout (time-based split, not random)
   - permutation testing (null distribution from shuffled event dates/labels)
   - regression controls (isolate the interaction effect from a plain beat effect and
     market/sector beta)
   - transaction-cost adjustment (realistic slippage/commission haircut)
4. **Decision point** — if the signal holds up, it becomes the flagship signal driving
   Phase 2. If it doesn't reproduce, that's the moment to find out and revisit, before
   more is built on it.

## Phase 2 — Minimal paper-trading loop

Goal: take the validated signal from a backtest to a running simulated system.

- Regime score module (from Phase 1's data)
- Universe ranking: rank the stock list by earnings-surprise magnitude, gated by regime
- Simple allocation: equal-weight top N, position size caps, stop-loss rule
- Order generation against a paper broker — Alpaca paper trading API (free) is the
  default choice unless something rules it out
- Daily report: what fired, why, current positions
- Scheduling: nightly analysis + pre-market check, run manually or via a simple
  scheduled task at this stage. Full intraday risk monitoring is deferred until this
  loop is proven out.

## Phase 3 — Add breadth to the fusion layer

Goal: bring in the other data streams and start actually fusing them, now that there's
a working loop and a proven anchor signal to fuse against.

- Numerical stream: technicals, more macro (rates, yield curve)
- Non-standard signals (free sources only):
  - Congressional/insider trading — House/Senate stock-watcher public data
  - Prediction markets — Kalshi / Polymarket public APIs
  - Narrative lifecycle — derived from the existing sentiment DB over time, no new
    source needed
  - Supply-chain / satellite / foot-traffic — deferred, generally paid; revisit only if
    the free signals prove insufficient
- Sentiment stream expansion: SEC filing language via EDGAR full-text search API
  (free); earnings-call tone deferred (transcripts are mostly paywalled)
- Fusion layer: agreement/conflict logic across streams, confidence-weighted
  combination — "when they agree, trade with high confidence; when they conflict, hold
  back"

## Phase 4 — Live-trading readiness (explicitly deferred)

Not started until Phase 2 has run clean in paper trading for a meaningful stretch.
Will need real broker integration, compliance/tax considerations, real risk controls,
and monitoring/alerting. No design work here yet on purpose.

## Working conventions

- No hardcoded secrets — environment variables only.
- SQLite for storage until it's an actual bottleneck.
- Prefer free data sources; only reach for paid ones when a free signal has already
  shown enough promise to justify it.
