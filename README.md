# PEAD stock trader

A low-frequency, long-only equity strategy that buys large-cap US stocks after unusually
large positive earnings surprises and holds them for about three months — plus the
validation machinery that decides whether a signal is allowed to trade at all.

The strategy itself is ordinary: post-earnings-announcement drift has been documented
since 1968. The part worth reading is [ROADMAP.md](ROADMAP.md), which records what was
tested, what failed, and why the surviving specification is what it is.

## The strategy

| | |
|---|---|
| Trigger | Quarterly earnings announcement, S&P 500 members |
| Signal | Price-scaled surprise `(actual EPS − estimate) / price`, ranked against the trailing 365 days |
| Entry | Top 5% of that ranking, at the first close a trader could act on |
| Exit | 60 trading sessions, or a 12-sd volatility trailing stop |
| Idle cash | Held in SPY |
| Sizing | Fixed fraction of equity at entry, never rebalanced |

Backtest on a point-in-time, survivorship-repaired universe (2011-2026): **+16.6% a year
vs SPY's +13.9%**, max drawdown −46%, ~68 trades a year, +1.76pp alpha per trade.
Walk-forward, re-choosing the stop each year from prior data only: **+15.15% vs +13.66%**.

## What was rejected

- **The founding hypothesis.** "Earnings beats pay extra when VIX is elevated" — tested on
  42,191 events, interaction ≈ 0 at every horizon (p 0.50-0.94). Rejected.
- **The first passing version of this strategy.** Its universe contained only firms still
  in the index today. With 100 departed firms restored and ticker renames matched by SEC
  CIK, the pre-registered test failed (+0.767pp, p=0.063) and the build stopped.
- **Hindsight-fitted parameters.** Re-selecting settings each year without hindsight kept
  only 9% of the original excess return.

The tighter cutoff that replaced it was pre-registered ([research/prereg_midsmall.md](research/prereg_midsmall.md),
committed before the data existed) and tested once on 1,152 S&P 400/600 firms the project
had never touched: **+2.73pp per trade, p=0.0009**.

## Honest caveats

- The effect is **era-dependent** in every sample: mid/small-cap returns concentrate in
  2019-2022; the S&P 500 version lost to SPY in 2014, 2017-2020, 2023 and 2024.
- **The whole edge lives in volatile stocks.** Split by volatility at entry, the top third
  earns +4.88pp per trade (p=0.001) while the bottom two thirds earn nothing (-0.10pp and
  +0.49pp). It is the signal rather than beta — in equally volatile stocks, ordinary
  earnings events earn +0.06pp — but the strategy only works where price swings are large,
  so drawdowns are the price of entry.
- **Delisted firms are still missing** (207 S&P 500, 543 mid/small — Yahoo drops them), but
  the damage is now measured rather than feared: 53% of them were acquired (deals close at
  a premium, so those are missing *winners*) and only 1% failed outright. Estimated cost
  to the backtest: **0.3-0.8pp a year**. See [research/survivorship_test.py](research/survivorship_test.py).
- Paper trading only. Nothing here is investment advice.

## Layout

```
trader/      the system: events -> signals -> fusion -> portfolio/exits -> engine
             -> broker (simulated or Alpaca paper) -> monitoring/report
research/    data ingestion, the gates, pre-registrations, stress tests
config/      gate-selected parameters + the tripwire band (committed)
data_archive/ perishable data: scored news, index membership snapshots
tests/       stdlib unittest
```

The backtest and the live run share the same modules, so there is no "the backtest did
something the live code doesn't" gap.

## Running it

```
python -m venv .venv && .venv\Scripts\pip install -r requirements.txt
python research/archive.py restore     # perishable data (news, membership)
python research/ingest.py              # prices, VIX, earnings (re-downloadable)
python -m trader.run_daily --dry-run   # decide and report, save nothing
python -m unittest discover -s tests -t .
```

Data lives in SQLite under `data/` (gitignored: ~500MB, all re-downloadable except the
archives above). `scripts/install_task.ps1` registers the weekday run on Windows.
