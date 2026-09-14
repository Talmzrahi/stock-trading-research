# Sentiment Arbitrage Trading System

AI-driven system trading the gap between market sentiment and company fundamentals.
Full context, phased plan, and the reasoning behind current priorities: see
[ROADMAP.md](ROADMAP.md) — read it before proposing what to build next.

## Current phase

Phase 1 is complete. **The core VIX + earnings-beat hypothesis was tested and
rejected** (interaction ~0 across 42,191 events). PEAD is confirmed but thin
(+0.124pp/5d, ~2bps net of costs). The fusion thesis is untested — its 12-month sample
failed a positive control. No phase is currently in flight; see ROADMAP.md for the open
options and don't assume the original hypothesis is still live.

## What exists

- `Main.py` — news sentiment backfill: Finnhub company news -> 3-model NLP ensemble
  (FinBERT / fintone / Twitter-RoBERTa) -> exponential time-decay weighting -> weekly
  sentiment scores in SQLite (`data/sentiment_history.db`). Run with
  `python Main.py` after setting `FINNHUB_API_KEY`. Note this pipeline has never been
  run for real — the research work below uses its own tables.
- `Dataframe.py` — exports the `weekly_summary` table to CSV.
- `research/` — the Phase 1 work: `ingest.py` (prices/VIX/EPS), `backtest.py` (the
  hypothesis test), `sentiment_events.py` (pre-earnings news scoring), `fusion_test.py`,
  `archive.py` (export/restore the perishable data).

Not built: numerical/financials stream beyond EPS, non-standard signals, fusion layer,
regime/ranking/allocation/execution, scheduling.

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
- Target paper-trading venue: Alpaca's paper trading API — not yet integrated, but
  that's the default choice for Phase 2 unless something rules it out.
- Runs on Windows locally (PowerShell). Keep scripts cross-platform-safe where it's
  easy (e.g. explicit UTF-8 stdout so emoji in status output doesn't crash on Windows'
  default console encoding), but don't over-engineer for platforms not in use.
