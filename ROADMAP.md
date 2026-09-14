# Roadmap

Trading system that fuses numerical data, non-standard/alternative signals, and news
sentiment to trade the gap between market sentiment and company fundamentals.

The original flagship idea was that when macro fear is high (VIX well above its
trailing average) and a company beats earnings, the market is too busy being afraid to
price the good news in. **That hypothesis was tested and rejected** — see Phase 1 below
for the numbers. The broader fusion thesis (cross-referencing streams to find where
they disagree) is untested rather than disproven, and remains the more promising idea.

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

## Phase 1 — Validate the core hypothesis — DONE (2026-09-14)

Built: [research/ingest.py](research/ingest.py) (S&P 500 prices, VIX, ~20y EPS
estimate/actual -> SQLite), [research/backtest.py](research/backtest.py) (the test),
[research/sentiment_events.py](research/sentiment_events.py) (pre-earnings news
sentiment), [research/fusion_test.py](research/fusion_test.py),
[research/archive.py](research/archive.py).

### Result 1 — the core hypothesis is REJECTED

Tested on 42,191 earnings events, 504 stocks, 2002-2026. The hypothesis needs beats to
be rewarded *extra* during elevated VIX, which is the beat x elevated interaction:

| horizon | interaction | p     |
|---------|-------------|-------|
| 1d      | +0.013pp    | 0.846 |
| 3d      | +0.074pp    | 0.502 |
| 5d      | +0.009pp    | 0.939 |
| 10d     | -0.051pp    | 0.767 |

Zero at every horizon, and across all 12 beat/VIX threshold combinations p ran
0.344-0.975 with signs scattered around zero. With 42k events this is a precisely
estimated zero, not an underpowered null.

What is actually there are two **additive**, not synergistic, effects:

- **PEAD (plain earnings drift): +0.124pp at 5d, p=0.003 — real.**
- **VIX regime effect: ~+0.23pp at 5d — real in-sample, but see the caveat below.**
- **Their interaction: +0.009pp, p=0.939 — zero.**

Elevated VIX lifted beats (+0.239pp) and non-beats (+0.229pp) essentially equally, so
the earnings filter is redundant with the VIX filter. The proposed mechanism — good
news going unpriced because the market is distracted by fear — is not visible.

### Result 2 — the fusion thesis is UNTESTED, not disproven

Scored 41,569 news articles published strictly before 1,940 earnings events, then
tested whether numbers/narrative disagreement predicts drift (direction pre-registered:
H1 needs a negative beat x sentiment interaction).

5-day interaction: **+0.243pp, 95% CI [-0.269, +0.755]** — indistinguishable from zero,
and the wrong sign for H1. But the result should not be read either way, because the
sample fails its positive control: PEAD, which is +0.124pp (p=0.003) over the full
history, comes out at **-0.285pp (p=0.301)** inside the 12-month news window. A sample
that cannot reproduce a confirmed stronger effect cannot adjudicate an unconfirmed
weaker one. The limit is the data window, not the hypothesis.

### Data constraints discovered (expensive to rediscover — check here first)

- **Finnhub free news: rolling ~12 months.** 18 months back returns zero articles. This
  is what makes the fusion test underpowered, and it means fetched news is
  **perishable** — hence [research/archive.py](research/archive.py) and `data_archive/`.
- **Finnhub free earnings: 4 quarters only**, and the historical earnings calendar
  returns empty. No revenue estimates at all.
- **yfinance: EPS estimate/actual back to ~2002**, capped at 100 quarters per ticker.
  Prices and `^VIX` go back to 1980/1990. All free, no key.
- **Revenue estimates are not available free anywhere** — actuals are (SEC/yfinance),
  estimates are the paywalled part (IBES/Zacks). So the literal "beats BOTH earnings
  and revenue" version of the hypothesis was never testable; everything above is
  EPS-only.
- Wikipedia's S&P 500 table 403s on urllib's default user-agent; fetch via requests.

### Methodology lessons worth keeping

- **Test the interaction, not the contrast.** "Beats during high VIX vs. beats during
  normal VIX" reads +0.281pp (p=0.009) while the interaction it gets mistaken for is
  +0.009pp (p=0.939). The contrast silently includes the regime effect. The same trap
  appeared again in the sentiment terciles.
- **Cluster on dates, not events.** VIX regime is a date-level property, so every stock
  reporting that day shares it; event-level t-tests overstate significance badly.
- **Survivorship bias is asymmetric.** The universe is today's S&P 500, which inflates
  the regime effect but cancels in a difference-in-differences. So the null is the
  trustworthy number here and the positive result is the contaminated one. A
  point-in-time universe would fix this if the regime effect is ever pursued.
- **Run a positive control.** Checking whether a known effect (PEAD) shows up in a
  subsample is what separated "hypothesis is wrong" from "sample can't answer".
- **Pre-register the direction** when two plausible stories predict opposite signs.

### Where this leaves the project

The one confirmed effect, PEAD, is ~+0.124pp over 5 days — roughly **2bps net** of a
10bps round-trip cost assumption. That is thin enough that execution quality could
erase it, which is worth weighing before committing to Phase 2 as originally scoped.

Open options, no decision made yet:

1. **Get multi-year news** and re-run the fusion test — the pipeline is built and
   validated, only the data is missing. GDELT (BigQuery, needs a Google Cloud account)
   or a paid provider.
2. **Try a different attention proxy** — VIX measures market-wide fear, not attention
   to a given stock. Same-day announcement counts or news volume are closer.
3. **Build Phase 2 on PEAD** as a thin anchor, accepting the margin is small.

## Phase 2 — Minimal paper-trading loop (blocked on a signal worth trading)

Goal: take a validated signal from a backtest to a running simulated system. Written
when the VIX hypothesis was expected to supply that signal; it did not. PEAD is the
only confirmed candidate and its margin is ~2bps net, so decide whether that is worth
building on before starting this phase.

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
