# Sentiment Arbitrage Trading System

AI-driven system trading the gap between market sentiment and company fundamentals.
Full context, phased plan, and the reasoning behind current priorities: see
[ROADMAP.md](ROADMAP.md) — read it before proposing what to build next.

## Current phase

Phase 1 — validating the core hypothesis (VIX-regime + earnings-beat signal) with a
real backtest, before building any more architecture on top of it.

## What exists

- `Main.py` — news sentiment backfill: Finnhub company news -> 3-model NLP ensemble
  (FinBERT / fintone / Twitter-RoBERTa) -> exponential time-decay weighting -> weekly
  sentiment scores in SQLite (`data/sentiment_history.db`). Run with
  `python Main.py` after setting `FINNHUB_API_KEY`.
- `Dataframe.py` — exports the `weekly_summary` table to CSV.

Nothing else from the original vision (numerical/financials stream, non-standard
signals, fusion layer, regime/ranking/allocation/execution, scheduling) exists in code
yet — see ROADMAP.md for sequencing.

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
