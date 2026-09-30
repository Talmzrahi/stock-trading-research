# Stock trading research

Code for finding, testing and paper-trading stock-market signals, built with an AI coding
agent on free data in 11 days (14–24 September 2026). It has three tracks: an earnings-surprise
strategy running on a paper account, language models reading 23,997 SEC earnings
releases, and a century-long study of volatility targeting.

**The short version:** the engineering works and the testing is strict. Nothing tested
beats the market once overlapping trades, market risk, survivorship and hindsight are
accounted for. The record of how each result was checked, including the ones that failed,
is the main output.

This repository is the project as of 2026-09-24 12:22 (tag `snapshot-2026-09-24`), with a
documentation cleanup on top.

## What was built

| Part | What it does |
|---|---|
| `trader/` | One code path for backtest and live: earnings events → signal → exits and sizing → engine → broker (simulated market-on-close ledger, or Alpaca paper) → state, monitoring, daily report. Runs on a weekday schedule. |
| `trader/text/` | Reads each day's SEC 8-K earnings releases: finds the filing, checks it was public before the decision, separates what is new from boilerplate, reads it with three free language models (FinBERT, DistilRoBERTa-finance, MiniLM) and scores it. Trades a separate shadow account. |
| `research/` | Data ingestion, a point-in-time index universe, the validation gate, pre-registrations, and every experiment with its result in its header. |
| `tests/` | 94 stdlib `unittest` tests, including regressions for five past calculation errors. |

## What it found

| Idea | Result |
|---|---|
| Earnings beats pay more when the VIX is high (the founding idea) | No effect on 42,191 events |
| Post-earnings drift, top 5% of surprises (the live paper strategy) | Backtest +16.6%/yr vs SPY +13.9% (2011–2026), but +1.06%/yr (p=0.75) after adjusting for its higher market risk |
| The same, out of sample on S&P 400/600, pre-registered | Passed (+2.73pp per trade, p=0.0009), then failed a stricter test for overlapping trades (alpha +2.94%/yr, p=0.37) |
| Sentiment of SEC 8-K text, one-shot 2020–2026 holdout | Failed (+0.43pp, p=0.38) |
| Language models reading earnings releases | Explain the earnings-day move better than the numbers alone (R² 0.0716 → 0.0853) but predict nothing after it; the long-only trade fails the gate |
| Volatility targeting, 7 pre-registered tests over ~100 years | One clean pass (S&P 500 1928–1992, mostly the Great Depression). Drawdowns smaller in 98% of 61 series, returns lower in 67% |
| Volatility targeting in bear markets | +17.7%/yr over buy-and-hold in bear markets dated with hindsight; −6.4% and −7.5%/yr when they must be recognised in real time |
| The gate fed 1,000 coin-flip rules | 20 passed (best p=0.0009), which is why a broad search needs a stricter bar |

![With hindsight the model turns $1 into $88; in real time, $29](docs/charts/1-hindsight-vs-real-time-growth.png)

The volatility-targeting result in one picture: knowing the bear markets in advance would
have beaten the S&P 500 by 3.5 points a year; switching on a signal you could see at the
time did not beat it at all.

The write-up with all five charts: [docs/FINDINGS.md](docs/FINDINGS.md). Every number, with
its caveats: [PROJECT_STATE.md](PROJECT_STATE.md). The history and the decisions behind it:
[ROADMAP.md](ROADMAP.md).

## How results are checked

- **Pre-registration:** the rule and pass mark are committed before any return is computed
  (`research/prereg_*.md`).
- **One-shot holdouts:** data used for a verdict is never reused.
- **An overlap-robust gate:** quarter-clustered standard errors (T1) and a calendar-time,
  market-adjusted alpha with Newey-West errors (T2), in `research/inference_check.py`.
- **A point-in-time universe:** index membership by quarter, matched by SEC company ID
  through ticker changes. Survivorship is measured, not assumed.
- **Shared, tested mechanics:** `research/mechanics.py`, where each function guards against
  an error this project actually made.

## Layout

```
trader/        the system (backtest and live share it)
trader/text/   the earnings-release reader
research/      ingestion, gates, pre-registrations, experiments
config/        live parameters with their evidence, the text model, the tripwire band
data_archive/  perishable data kept in git: index snapshots, earnings, reader outputs
docs/          the findings write-up, its charts, and the commit map
tests/         stdlib unittest
scripts/       Windows Task Scheduler setup
```

## Running it

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -t .
.venv\Scripts\python.exe research\archive.py restore     # perishable data from data_archive/
.venv\Scripts\python.exe research\ingest.py              # prices, VIX, earnings
.venv\Scripts\python.exe research\vol_ingest.py          # index and ETF prices for volatility targeting
.venv\Scripts\python.exe research\edgar_filings.py       # SEC earnings releases (hours, resumable)
```

Data lives in SQLite under `data/` (gitignored, several GB with the SEC filings). Free
sources only: Yahoo Finance via yfinance, SEC EDGAR, Wikipedia, and Finnhub's free tier.
`python -m trader.run_daily --dry-run` decides and reports without trading, but still
refreshes prices into `data/research.db`.

## Notes

- Paper trading only. Nothing here is investment advice.
- The Finnhub news used in the first experiment is not included: Finnhub's terms forbid
  redistributing it, so it was removed from the history ([data_archive/README.md](data_archive/README.md)).
  Commit IDs changed as a result; [docs/COMMIT_MAP.md](docs/COMMIT_MAP.md) maps the old ones.
- Built with Claude Code as a pair programmer; the commits are co-authored.
