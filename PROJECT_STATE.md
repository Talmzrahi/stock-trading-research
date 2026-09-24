# Project state — 2026-09-21

A handoff for a fresh session. `CLAUDE.md` holds the standing rules and `ROADMAP.md` the
phase-by-phase history; this file is the current picture, the evidence behind it, and the
one direction worth exploring next (long-short, at the end).

---

## 1. What this is

A paper-trading system that buys S&P 500 stocks after unusually large earnings surprises
and holds them for 60 sessions. One code path serves both the backtest and the live run
(`trader/`). A second, newer piece reads the companies' earnings press releases with three
language models and scores them (`trader/text/`), but it does not trade real decisions.

**The honest one-liner:** the plumbing is finished and reliable; the evidence that any of
it beats simply holding SPY is not there yet.

---

## 2. What is running right now

| | Live account (`data/trader.db`) | Shadow account (`data/shadow.db`) |
|---|---|---|
| Opened | 2026-09-15 with $1,000 | 2026-09-21 with $1,000 |
| Holds | 1.3136 SPY (idle cash), equity $1,006.74 at the 09-17 close | all cash |
| Closed trades | 0 | 0 |
| Signal | top 5% price-scaled EPS surprise | text score (top 5% of the model's training predictions) |
| Purpose | out-of-time test of the earnings signal | out-of-time test of the text signal |

**The daily run** (`python -m trader.run_daily`) is registered in Task Scheduler for
weekdays at 14:30 New York time. It refreshes prices and earnings, settles fills, decides
and submits market-on-close orders, reads any earnings releases filed that day, scores
them, advances the shadow account, and writes `reports/YYYY-MM-DD.md`.

**Scheduler caveat, learned the hard way:** the task was silently dead for three days
because Windows' defaults refuse to start it on battery and kill it when the power source
changes. It now runs on battery, retries three times, and logs each start and exit code to
`data/run_daily.log`. It still cannot wake a sleeping laptop — `StartWhenAvailable` catches
up when the machine wakes. The first full run with everything wired is the evening of
2026-09-21.

---

## 3. Evidence ledger

What has been tested, what it showed, and whether that data is now spent.

| Test | Result | Data |
|---|---|---|
| VIX regime × earnings beat (Phase 1) | rejected, a precisely estimated zero | — |
| PEAD top-decile, repaired universe | FAIL: +0.767pp, p=0.063 | S&P 500 |
| PEAD top-5%, out-of-sample, pre-registered | PASS: +2.730pp, p=0.0009, date-clustered | S&P 400/600 prices — **spent** |
| **The same, re-checked properly (Phase 2e)** | **does not survive**: quarter-clustered p=0.021, calendar-time alpha +2.94%/yr p=0.37, beta 1.36 | same |
| v2 sentiment (word lists on 8-K text) | **FAIL**: +0.431pp, p=0.384; correlation +0.003 | S&P 500 2020-26 holdout — **spent, never reuse** |
| v3 reader, explaining the announcement move | **works**: R² 0.0853 vs 0.0716 from the surprise alone, 23,997 releases | S&P 500 development |
| v3 as a long-only trade | **fails the gate**: top 5% −0.051pp/trade (p=0.95), calendar-time alpha −1.82%/yr (p=0.58) | same |
| **PEAD gate on the pooled S&P 1500** | T1 +2.45pp (p=0.014); **T2 alpha +0.29%/yr, 95% CI [−5.5, +6.0]**, beta 1.39 | S&P 500 + 400/600 prices — both already examined |
| Correlation-aware weighting (ERC, min-variance) | **REJECTED**: ERC excess Sharpe 0.36 vs inverse-vol 0.62; min-variance 0.49. Shrinking the covariance to its diagonal restores 0.62 — the correlation information is the harm | construction |
| Trend filter on the diversified book | **REJECTED**: Sharpe 0.72 → 0.54 at 200d, worse at every window, and does not fix 2021-26 | construction |
| Cost of leverage | futures-style financing (rf+0.3%) vs retail margin (rf+1.5%) buys **+1.7%/yr, +0.09 Sharpe** at target 20% — the largest single improvement found | construction |
| **Vol targeting, S&P 500 1928-1992 (pre-registered)** | **PASS**: ΔexSharpe +0.19, drawdown −85% → −56% (+34%). Monotone in the dividend assumption | ^GSPC 65 untouched years — **spent** |
| **Vol targeting, 9 SPDR sectors (pre-registered)** | **FAIL by 0.0022**: median ΔexSharpe 0.09778 vs a 0.10 bar — but **9/9 sectors improved** and drawdowns fell 48% | sectors 1999-2026 — **spent** |
| **Vol targeting, national indices pre-1996 (pre-registered)** | **FAIL**: median ΔexSharpe −0.00, only 2/5 improve (Japan +0.20, HK +0.38; Canada −0.10, UK −0.00, Germany −0.19) | 5 indices — **spent** |
| "Helps most where buy-and-hold is worst" (pre-registered prediction) | **NOT CONFIRMED**: r = −0.089 across 15 series against a −0.30 bar. The pattern was noise and leaves the conclusions | same |
| Switch targeting on only in real-time bear markets | **REJECTED**: targeted only below the 200-day average ΔexSharpe −0.020, only when down 20% −0.034, against +0.046 left on always. Real-time signals arrive after 31-59% of the fall | 61 series, post-hoc |
| Is the S&P 1928-1992 pass real timing? (permutation) | **YES**: 0 of 1,000 books with the same weights shuffled across months matched +0.186 (best +0.120, mean −0.064) | post-hoc |
| Diversified risk-parity book (**construction, not a test**) | 2005+, margin charged: target 20% gives **12.51%/yr, excess Sharpe 0.54, DD −42%, 12.9× growth** vs SPY 10.94% / 0.48 / −55% / 9.5× — **beats SPY on return, excess Sharpe and drawdown**. With futures financing (rf+0.3%): **14.21%/yr, 0.63, −41%, 17.8×**. Lags 2021-2026 (+7.6%/yr vs SPY +15.1%) | data already seen |
| **Volatility targeting, 17 markets (pre-registered)** | **PASS as executed, but on RAW Sharpe** (+0.144). On excess Sharpe, the standard, the median is **+0.067 — below the +0.10 bar**; 17/17 still improve and drawdown still falls 42%. Raw Sharpe flatters the strategy holding more cash, which is exactly what this one does | 17 country ETFs 1996-2026 — **spent** |
| **Volatility targeting, SPY (pre-registered)** | **FAIL** on the 1993-2001 holdout: Sharpe +0.06 (needed +0.10), plateau 7/12 (needed 8), +0.04 at 10bps. Drawdown criterion passed (−35.6% → −26.9%). Premise held (vol autocorrelation 0.604) | SPY 1993-2001 — **spent, never reuse** |
| **Long-short grid, 27 configs** | best is PEAD 10% / 20-session: **+3.57%/yr, Sharpe 0.36, p=0.207**; −2.60%/yr in 2014-19, +8.83%/yr in 2020-26; +0.67%/yr at 20bps a side | S&P 500 development, fitted search |
| **Survivorship in 2008-2010** | that window alone: **T2 alpha +24.49%/yr, CI [+1.6, +47.4]** off 184 trades — excluded from the analysis window | S&P 500, 54% price coverage |
| v3 score ranking the PEAD picks (reaction-trained) | **fails, backwards**: worst-read fifth +3.58pp, best-read +0.04pp | S&P 500 development |
| The same score refitted on 60-session drift | **unresolved**: T1 +3.37pp p=0.062, T2 +6.05%/yr p=0.276, once nothing sees the future | same |
| S&P 400/600 release text (the v3 exam) | **never touched** — 27,164 releases downloaded, returns never read | **unspent** |

**The live strategy's own backtest**, 15.3 years: 16.66%/yr against SPY's 13.91%, Sharpe
0.82 against 0.85, worst drawdown −46% against −34%. After adjusting for its higher market
exposure (beta 1.16) the edge is +1.06%/yr with p=0.75. It is not distinguishable from
holding a bit more market risk.

---

## 4. Rules that govern changes

These are the owner's, recorded in `CLAUDE.md`, and they are why the project has thrown
away three ideas instead of talking itself into them.

1. **The gate is mandatory.** Nothing reaches `config/strategy.json` without passing the
   event-level gate and the portfolio gate. Since 2026-09-19, anything with overlapping
   holding periods must also clear **T1** (standard errors clustered by calendar quarter)
   and **T2** (calendar-time alpha against the matched benchmark, Newey-West), both
   implemented in `research/inference_check.py`. Date clustering alone overstated every
   earlier result.
2. **Pre-register before looking.** Write the rule and the verdict condition into a
   committed file first. `research/prereg_*.md` are the precedents.
3. **One-shot data stays one-shot.** The S&P 500 2020-26 sentiment holdout is spent. The
   S&P 400/600 exam set is not, and nothing that touches its returns may run before a
   pre-registration is committed. Layer 0 (pure text bookkeeping) is allowed on it.
4. **The project is free** — no paid APIs, data or models unless the owner says otherwise.
5. **Scores are written once.** `text_scores` rows are never revised; each carries the
   model that produced it. A score is evidence about what was knowable that afternoon.

---

## 5. The text signal (v3), in one page

**Pipeline**, `trader/text/`, same code in research and live:

1. `fetch.py` — find the company's 8-K with Item 2.02 within two days of the announcement;
   refuse to score anything whose acceptance time is later than the decision.
2. `parse.py` — the EX-99 exhibit's HTML into paragraphs and table rows. Handles every
   generator from 2010 to 2026 and repairs Windows-1252 quotes and dashes.
3. `novelty.py` — every sentence and row against the company's own previous four releases:
   *boilerplate* (identical), *template* (same words, new numbers), *edited*, *new*. In a
   mature release, 44% of sentences are verbatim repeats and 15% only change numbers.
4. `readers.py` — the first 20 new or edited sentences, read by three free models. The cap
   is measured, not guessed: reading is flat from the first 10 sentences upward, because
   the news sits at the top of a release.
5. `features.py` / `model.py` — 14 features into a ridge fitted on 23,997 releases
   (`config/text_model.{json,npz}`, written by `research/v3_fit.py`).

**What it predicts:** the announcement reaction — the stock's move against SPY from the
last close before the release to the first close the system could trade — from the text
**and** the earnings surprise, so text is only rewarded for what the numbers do not say.

**Reader comparison** (out-of-fold R² on the reaction, 23,997 releases):

| | R² |
|---|---|
| Earnings surprise alone | 0.0716 |
| + word lists | 0.0748 |
| + FinBERT | 0.0798 |
| + DistilRoBERTa-finance | 0.0813 |
| + **MiniLM** (384-number "what is this about" fingerprint) | **0.0819** |
| + everything | **0.0853** |

Two things worth carrying forward. **MiniLM went from worst to best** between 8,000 and
24,000 releases — its map needs data. And the **model-disagreement feature faded**: +0.0011
at 8,000 releases, +0.0001 at 24,000. That is what a noise finding does when tested on more
data.

**Speeds on this machine** (Snapdragon X Plus, 8 cores, native ARM64 Python): MiniLM ~85
sentences/s, DistilRoBERTa ~33/s, FinBERT ~18/s. Running the three side by side is *slower*
than one after another. 8-bit quantisation is slower still and disagrees with the full
model.

---

## 6. Long-short — the direction worth exploring

### Why it came up

The text score's top-minus-bottom fifth is **+1.12pp over 60 sessions (p<0.005)** on the
full development set. That looks tradable until you see where it comes from:

| Text-score decile | 60-session return vs SPY |
|---|---|
| 1 (worst-read) | −0.79pp |
| 2 | −0.62pp |
| 3 | −0.75pp |
| 4 | −0.70pp |
| 5 | −0.23pp |
| 6 | −0.53pp |
| 7 | −0.21pp |
| 8 | −0.20pp |
| **9** | **+0.60pp** |
| 10 (best-read) | +0.23pp |
| *all releases* | *−0.32pp* |
| **top 5% — what a long-only rule buys** | **−0.46pp** |

**The spread is the bottom falling, not the top rising.** A long-only system cannot reach
it. That is the whole reason long-short is on the table.

Note the base rate: the average release underperforms SPY by −0.32pp over 60 sessions,
because equal-weighted stocks lagged the cap-weighted index across 2011-2026. Every
long-only number above fights that drag; a dollar-neutral book cancels it, which is the
second reason long-short is the natural shape for this signal.

### What it would have earned

`research/v3_longshort_probe.py`, development data, dollar-neutral, equal-weighted, each
event held 60 sessions from the next close, 10bps a side plus 1%/yr borrow on the shorts:

| | |
|---|---|
| Decile 10 minus decile 1, per trade | +1.02pp (p=0.066, quarter-clustered) |
| **Decile 9 minus decile 1** | **+1.38pp (p=0.007)** |
| Portfolio return | **+0.58%/yr**, volatility 4.2%, **Sharpe 0.14** |
| Significance | t = 0.53, **p = 0.59 — not significant** |
| Alpha vs SPY | +0.62%/yr (p=0.55), **beta −0.00** |
| Book size | ~36 long and ~36 short positions at a time |
| Years positive | 9 of 16 (worst 2020 −8%, best 2024 +7%) |

**Corrected 2026-09-22.** The portfolio rows previously read +1.51%/yr, Sharpe 0.36,
p=0.16. `book()` applied the long/short sign *after* subtracting costs, which turns every
short-side cost into a gain — worth about +0.9pp/yr at a 60-session hold and more at
shorter ones. The per-trade decile spreads were never affected. Fixed in
`research/v3_longshort_probe.py`.

### How to read that honestly

- **It is the best shape the signal has taken**, and the only one where the money is
  reachable. Market-neutral by construction: beta −0.00.
- **It is still not significant**, on development data that has been looked at many times.
  p=0.16 after a dozen prior tests is weak.
- **The decile pattern is not monotone.** Decile 9 beats decile 10 (+0.60 vs +0.23), and
  decile 9 minus 1 is the strongest pair. A signal whose second-best bucket beats its best
  is a warning: either the extreme tail is different in kind, or this is noise.
- **Low volatility flatters the Sharpe.** 4.2% volatility on a 1.5% return is a thin edge,
  and costs were charged at 10bps a side plus 1%/yr borrow; both could be worse in practice.

### What would have to happen before it could trade

1. **A pre-registered test on the S&P 400/600 exam set** — the only unspent data. Write the
   rule (which deciles, holding period, costs, borrow) and the verdict condition first, and
   require **T1 and T2**, as the gate demands. The exam set's releases are already
   downloaded; it would need layer 0, labels and reader scoring — roughly 5-7 hours of
   compute on this machine — and none of that touches its returns until the
   pre-registration is committed.
2. **Shorting in the engine.** `trader/engine.py`, `portfolio.py` and `exits.py` are
   long-only throughout: sizing, the volatility trailing stop and the benchmark-cash rule
   all assume long positions. This is real work, not a flag.
3. **A broker that can short.** Alpaca paper supports it; the simulated ledger does not
   model borrow, locate, or margin at all.

### The constraint that matters most

**The intended real stake is about $100, and you cannot short with $100.** A US margin
account needs $2,000 minimum under Reg T before any short position is possible, and borrow
plus margin interest eat a 1.5%/yr edge at small size. So long-short is a research
direction and a paper-trading question for now, not a path to real money at this size. That
is worth deciding deliberately rather than discovering after building shorting into the
engine.

### If you want the cheapest next step

Score the exam set's 27,164 releases and pre-register the decile-9-or-10 minus decile-1
test. That answers "is this real?" for about 6 hours of background compute and no money,
and it either kills the idea or makes the engine work worth doing.

---

## 7. Where everything lives

| Path | What |
|---|---|
| `trader/` | the trading system; `run_daily.py` is the entry point |
| `trader/text/` | the production text pipeline (fetch, parse, novelty, readers, features, model, pipeline, store) |
| `trader/shadow.py` | the shadow account |
| `research/` | every experiment, each with its result in its header or a design doc |
| `research/design_sentiment_v3.md` | the v3 design, every decision and every measured result |
| `research/prereg_*.md` | pre-registrations, committed before their tests ran |
| `config/` | `strategy.json` (live parameters + evidence), `text_model.{json,npz}`, `tripwire_band.csv` |
| `data_archive/` | committed: the scored sample, reader features, reports, the v1 news articles |
| `reports/` | daily reports (gitignored) |

**Databases** (all gitignored, in `data/`):

| File | Size | Contents | Reproducible? |
|---|---|---|---|
| `research.db` | 244 MB | prices, VIX, earnings, 41,569 scored news articles | prices/earnings yes; **the news articles no** (archived) |
| `edgar.db` | 692 MB | 26,357 S&P 500 releases, raw HTML with structure | yes, hours |
| `edgar_midsmall.db` | 363 MB | **the exam set**: 27,164 S&P 400/600 releases, text only | yes |
| `v3.db` | 920 MB | layer 0 output, reaction labels, reader outputs, features | yes, ~9 h of compute |
| `trader.db` | 3 MB | the live account, and `text_scores` | **no — this is the live record** |
| `shadow.db` | 3 MB | the shadow account | **no** |

---

## 8. Environment notes that cost time to learn

- **The laptop sleeps.** A 20-hour stretch of "slow" scoring turned out to be 19.8 hours of
  standby. Long jobs need AC power, an open lid, and ideally sleep disabled while they run.
- **On battery, Windows throttles the CPU** and kills scheduled tasks on power-source
  change unless the task says otherwise (it now does).
- **All long jobs are resumable.** Re-running the same command picks up where it stopped.
- **SQLite is in WAL mode** where readers and writers overlap. `filings` has no index on
  `cik`; filtering it per company re-reads the whole blob-heavy table, which once turned a
  10-minute job into hours.
- **Everything is local and free.** The SEC contact is read from `git config user.email` at
  run time and never written into the repo.

---

## 8b. The universe repair (2026-09-22) and what it cost to get right

`research/universe.py` skipped any quarter that raised, silently. Two faults were hitting
it: no retry against Wikipedia's throttling (which returns an HTML error page where JSON is
expected), and a trailing footnote row that parses as NaN and survives `.astype(str)` as a
float under pandas' `str` dtype. Between them, **2012-01 through 2014-07 were never
collected** and every pre-2010 revision failed. `point_in_time` falls back to the last
snapshot before a gap, so events from 2012 to mid-2014 were matched against the 2011-10
index — up to three years stale.

Both fixed; snapshots 56 → 79, coverage now 2007-04 → 2026-09.

**The trap, recorded because it nearly became a finding.** Filling `universe_history`
without rerunning `research/universe_ids.py` makes things *worse*, not better:
`point_in_time` matches by firm CIK, and a snapshot with no `universe_ids` row falls back
to `"SYM:<ticker>"` and matches almost nothing. Recognised 2012 events went from ~1,360 to
**16**. The resulting weaker numbers looked like "correct data deflates the signal" and were
written up as such for one commit. They were a bug. **Run `universe_ids.py` every time.**

**Why the analysis window starts 2010-05 even though data now reaches 2007-04.** Measured,
not assumed:

| Window | Trades | T1 | T2 alpha | 95% CI |
|---|---|---|---|---|
| 2010-05 → 2026 | 1,289 | +1.89pp (p=0.077) | −0.95%/yr | [−9.23, +7.34] |
| 2008-01 → 2026 | 1,473 | +2.15pp (p=0.027) | +2.94%/yr | [−5.22, +11.10] |
| **2008-2010 only** | **184** | +3.88pp (p=0.092) | **+24.49%/yr** | **[+1.59, +47.39]** |

Only 54% of 2007-era index members have prices, and the missing half is disproportionately
the firms that failed — 48 left the index in 2008 alone. Coverage climbs about a point a
year to 100% today, so the further back you go the more the sample is reconstructed from
winners. That window measures survivorship, not drift, and including it moves the headline
from −0.95%/yr to +2.94%/yr.

**Consequence to know about:** `research/pead_trailing.py` has no window guard, so after the
repair it now prints **PASS, +1.019pp, p=0.0093**, where it recorded FAIL (+0.767pp,
p=0.063). That flip is the contaminated window (its 2008-2013 era reads +1.999pp, p=0.003).
Its pre-registered definition was left untouched deliberately — do not read its verdict
without this paragraph. `research/sp1500_gate.py` carries the guard and the honest numbers.

---

## 8c. Backtest mechanics are now shared and tested (2026-09-23)

Five errors between 2026-09-21 and 2026-09-23, every one of which returned a number that
looked like a finding, and every one of which was a line of arithmetic written slightly
wrong in a script with no tests:

| | error | cost of it |
|---|---|---|
| 1 | in-sample ranking (`pd.qcut` over the whole sample) | +8.85%/yr → +1.18%/yr |
| 2 | full-sample `mean + k·sd` threshold | +9.54%/yr → +1.65%/yr (−2.65% on 1y) |
| 3 | `out_of_fold` trains on later folds | ~half the remaining effect |
| 4 | `side` applied AFTER the cost subtraction | +1.51%/yr → +0.58%/yr |
| 5 | post-hoc vol matching with no margin charged | showed 165× growth |

Four are timing errors, one is accounting. `research/mechanics.py` now holds the correct
version of each — `hold_from_month_end`, `trailing_stats`, `position_book`,
`levered_return`, `performance`, `years_to_significance` — with each docstring naming the
error it prevents. `tests/test_mechanics.py` encodes all five as regressions; the suite is
94 tests, up from 75. Reintroducing error 4 makes a short position's cost a **gain** of
+0.0020 where the test asserts a loss, so it fails.

**Use these rather than rewriting them.** The existing scripts still carry their own
copies; new work should not.

`performance()` reports excess Sharpe (over T-bills) **and** raw, because raw flatters
whichever strategy holds more cash and quoting only one lets it be chosen quietly. Every
Sharpe recorded before 2026-09-23 in this file was raw.

`years_to_significance()` exists to be quoted before anyone proposes a shadow book as a
test: at Sharpe 0.36 it is ~30 years, at 0.63 about 10. That is why Phase 4 was not built.

---

## 8d. What seven pre-registered tests say (2026-09-23)

| Test | Sample | ΔexSharpe | ΔDrawdown | Verdict |
|---|---|---|---|---|
| SPY holdout | 1993-2001 | +0.06 | +24% | **FAIL** |
| 17 country ETFs | 1996-2026 | +0.067 excess (+0.144 raw) | +42% | **PASS on raw, FAIL on excess** |
| S&P 500 price index | 1928-1992 | +0.19 | +34% | **PASS** |
| 9 SPDR sectors | 1999-2026 | +0.098 | +48% | **FAIL** by 0.0022 |
| 5 national indices | pre-1996 | −0.00 | 4 of 5 up | **FAIL** |
| 14 EM countries | full history | +0.038 | +40% | **FAIL** |
| **14 EM countries** | **2015-2026** | **−0.028** | **+35%** | **FAIL** |

**Two of seven as executed; one of seven on a consistent excess-Sharpe basis** (§8e). Roughly 100 years, five continents, sectors, asset classes and both
developed and emerging markets.

### The conclusion, now firmly evidenced

**Volatility targeting is a drawdown-reduction tool. It is not a return tool.**

- **Drawdown fell in every single sample**, by 24% to 48%, and in all 14 EM markets in both
  windows. Nothing tested has failed this.
- **Sharpe moved +0.19, +0.14, +0.098, +0.038, −0.00, −0.028, +0.06** — inconsistent in size
  and sign, and *negative* in the most recent window.

### The recent regime: the answer is no

Test E isolated 2015-2026 on data never previously examined. Median excess-Sharpe change
**−0.028**, and only **4 of 14** markets improved, against 10 of 14 over full history. This
was predicted in advance and it confirms what the diversified book already showed (+6.6%/yr
against SPY's +15.1% since 2021). The weakness is real, it is recent, and it is not an
artifact of one market or one construction.

### Three predictions, made in advance, all correct

Before running: drawdown would pass in both tests; Sharpe would fail in at least one; and E
would be worse than D on Sharpe. **3/3.** That matters more than any single verdict — the
model of what this mechanism does is now accurate enough to predict outcomes on unseen data,
which is the first time anything in this project has cleared that bar.

### What is left

Every sample is spent: SPY 1993-2001, 17 developed countries, ^GSPC 1928-1992, 9 sectors,
5 pre-1996 indices, 14 EM funds (twice). Further testing needs data that does not exist
here. The defensible claim is narrow and well-supported: **a book that cuts drawdown by
roughly a third with no reliable effect on risk-adjusted return, and a negative effect over
the last decade.**

---

## 8e. What separates passes from failures (2026-09-24, post-hoc)

`research/voltarget_synthesis.py`. **Post-hoc on spent data** — every sample below already
carries a verdict, so this explains those verdicts and generates hypotheses; it proves
nothing. 61 series, not independent.

### A correction first

The 17-country test measured **raw** Sharpe (`voltarget_intl.py`'s `stats()` returns
`ann / vol`). On excess Sharpe its median improvement is **+0.067**, not +0.144, which misses
the +0.10 bar. The bias runs in the strategy's favour by construction — it holds cash, and
raw Sharpe puts the risk-free rate in the numerator — and it was identified on 2026-09-23
without going back to check which verdicts depended on it. This one did. On a consistent
basis **one of seven tests passes**: the S&P 500 1928-1992.

### The mechanism explains the verdicts; the intuitive stories do not

Correlation of each series' Sharpe improvement with:

| candidate | r | rank r |
|---|---|---|
| **month-end volatility vs the NEXT month's return** | **−0.718** | **−0.699** |
| **volatility persistence** | **+0.457** | **+0.537** |
| buy-and-hold Sharpe (the story pre-registered and failed) | +0.186 | +0.176 |
| how long the worst drawdown took to bottom ("slow bears") | −0.109 | −0.029 |

Targeting cuts exposure when volatility is high. It works when two things hold:
**volatility is persistent**, so today's estimate is still true next month; and **high
volatility precedes weak returns**, so stepping aside avoids losses rather than missing a
rebound. Together these account for about half the cross-series variance (r² ≈ 0.52).

The two failures that looked mysterious are explained by one leg each:

- **Pre-1996 national indices** had *favourable* vol→return (−0.027) but the lowest
  persistence of any sample (0.36). The estimate was stale by the time it was used.
- **EM 2015-2026** had the **most positive vol→return of any sample, +0.156** — high
  volatility has been followed by rebounds. That is the recent regime in one number, and it
  is why targeting has stopped working: de-risking into a spike that reverses is a loss.

### The passes are episodes

Removing each series' worst drawdown (peak to trough plus six months):

| | with | without |
|---|---|---|
| S&P 500 1928-1992 | +0.186 | **+0.054** — the Depression is ~70% of the result |
| 17 countries | +0.067 | +0.031 |
| 9 sectors | +0.098 | +0.050 |
| EM 2015-2026 | −0.028 | **+0.025** — COVID's V-shape was the damage |

### The one thing that behaves like a law

| across 61 series | share |
|---|---|
| **drawdown smaller** | **98%** |
| return per unit of drawdown higher | 80% |
| excess Sharpe higher | 72% — but small, and episode-driven |
| **raw return lower** | **67% — the premium paid for the insurance** |

### It explains the past and cannot predict the future

The obvious next step was a regime-aware version: estimate persistence and the vol→return
relation on trailing data, and target only when both look favourable. Checked before
building it, by splitting all 61 series in half by time:

| | r |
|---|---|
| vol→return, first half vs second half | **−0.283** — it does not persist, it tends to *reverse* |
| Sharpe gain, first half vs second half | **−0.032** — whether it helped before says nothing |
| first-half vol→return predicting second-half gain | **+0.151** — wrong sign, and weak |
| *same-period* vol→return against the same period's gain | *−0.832* |

The variable that decides whether targeting works is itself not forecastable from its own
history. **The mechanism explains outcomes almost perfectly after the fact (r = −0.83) and
has no power in advance.** A regime-conditional version was therefore not built: it would
need to forecast the one quantity the data says cannot be forecast from its past.

This is the pattern of the whole project in one table. Every in-sample explanation here
has been strong; every attempt to use one out of sample has been weak.

Volatility targeting is insurance. It pays out in slow, persistent crashes, costs a premium
in most other years, and costs more than usual in a regime where volatility spikes get
bought — which describes the last decade.

---

## 8f. Bear markets, and what a broad search would find (2026-09-24, post-hoc)

### It is a bear-market tool, and you cannot switch it on only in bear markets

`research/bear_regimes.py`, 61 series. Targeted minus buy-and-hold, annualised, medians:

| when | edge |
|---|---|
| **hindsight** bear markets, peak to trough | **+17.7%/yr** |
| price below its 200-day average (knowable at the time) | **−6.4%/yr** |
| already down 20% from the peak (knowable at the time) | **−7.5%/yr** |
| outside hindsight bear markets | −10.3%/yr |

In a bear market recognised afterwards, targeting is excellent. In every bear market as a
real-time signal would have defined it, it loses. Across 272 bear markets, by the time price
fell below its 200-day average **31% of the eventual fall was already gone**; by the time it
was 20% off the peak, **59%**. And of the days each signal called "bear", only 45% and 32%
were actually falling — the rest were the bottom and the rebound, which is exactly where
de-risking costs money.

Hence the switching rule loses to leaving it on: ΔexSharpe −0.020 (below 200-day) and
−0.034 (down 20%) against **+0.046** always on, with worse drawdowns (−49%, −47% against
−36%). **Volatility is already the fastest real-time bear signal available** — it rises at the
start of a decline, before price crosses a moving average — so gating it behind a slower
signal only makes it later. The way to own the bear-market payoff is to hold the insurance
all the time and pay the premium in normal years.

And the timing it does have is real. `research/noise_search.py` keeps the S&P 1928-1992
weights and shuffles them across months — identical exposures, timing destroyed. **None of
1,000 shuffled books matched the actual +0.186** (mean −0.064, 95th percentile +0.025, best
+0.120). Whatever drives the one surviving pass, it is not having the right exposure at a
lucky moment.

### "Try as many things as possible until something clicks"

The same script feeds the gate's T2 test strategies that contain **no information**:

| | tested | passed | best result |
|---|---|---|---|
| coin-flip timing rules | 1,000 | **20** | alpha +3.13%/yr, **p = 0.0009** |
| pairwise "interactions" of 40 coin flips | 780 | **64** (8.2%) | alpha +2.51%/yr, p = 0.0004 |

Best of *n* noise rules: 1 tried, p = 0.35; 10, p = 0.13; **100, p = 0.0014**; 1,000,
p = 0.0009. By a hundred tries, pure noise produces a +3%/yr alpha that would clear this
project's T2. The best of a thousand reaches p = 0.0009 — the same p-value as the S&P 400/600
result this project was once built on.

**Interactions make it worse, not better.** Pairs share components, so one lucky base
signal manufactures dozens of correlated "discoveries": 8.2% of noise interactions passed
against 2.0% of single rules.

A checker is good because it is pointed at few hypotheses, each once, on fresh data. Pointed
at a firehose it becomes a false-positive generator. The legitimate version of a broad search
needs three things this project does not yet have: **a ledger that counts every trial**, a
**pass bar that scales with that count** (Bonferroni at 1,000 trials is p < 0.00005, which
none of the noise above reaches), and a **discovery / validation / holdout split fixed before
the search starts**. The shuffle null above is the other half: a strategy must beat its own
exposures reordered at random, not just zero.

---

## 9. Open questions

1. **Keep the shadow account running?** Its rule (top 5% of the text score, long-only) has
   no development edge. Live evidence is cheap, but it is evidence about a rule we expect
   to be flat.
2. **Long-short**, as above: test on the exam set first, decide about engine work after,
   and settle whether a strategy that needs $2,000+ fits a project aiming at $100.
3. **Real money** on the earnings strategy is not supported by the current evidence, and
   the strategy does not pass the upgraded gate. The paper account is the only clean test
   left running.
4. **Expanding the universe does not buy the power we need — measured, 2026-09-22.**
   `research/sp1500_gate.py` pooled the S&P 500 and S&P 400/600 PEAD samples: 2,758 events
   against 1,278, i.e. 2.7x. The gate's answer got *clearer*, not better: T1 +2.461pp
   (p=0.013) but **T2 alpha +0.56%/yr, p=0.849, beta 1.40**. After market exposure there is
   nothing there, and the pooled estimate is now tight enough to say so.

   The structural lesson matters more than the number. **Pooling added 2.7x the events and
   zero quarters** (65 → 65) and one month (196 → 197), because both halves span the same
   calendar. T1 clusters by quarter and T2 regresses monthly returns, so both are limited by
   *calendar span*, not by how many stocks are in it. The T2 standard error fell only
   4.19% → 2.93%/yr (1.43x) and the T1 standard error barely moved (1.048 → 0.965pp, 1.09x).
   More names diversify each month; they do not add independent months, and that saturates.

   So the earlier "~10x the universe would cut T2's SE to ~1.7%/yr" was **wrong**. On the
   measured scaling it would buy perhaps 2x, with diminishing returns, and the residual is
   market-wide noise that only more years can reduce. There are only ~16 years of
   point-in-time membership and they cannot be bought. Build a 5,000-stock pipeline for
   better *coverage* if that is wanted, but not expecting it to make these tests decisive.
5. **The 8-K text file `edgar.db` keeps growing** as the daily run caches new releases,
   which is deliberate: today's filing is next quarter's history.

---

## 10. Verifying any of this

```
.venv\Scripts\python.exe -m unittest discover -s tests -t .     # 75 tests
.venv\Scripts\python.exe -m unittest discover -s tests -t .     # 94 tests (19 are mechanics regressions)
.venv\Scripts\python.exe -m trader.run_daily --dry-run          # full pipeline, saves nothing
.venv\Scripts\python.exe research\inference_check.py            # the PEAD re-check (T0/T1/T2/T3)
.venv\Scripts\python.exe research\v3_readers_eval.py            # reader comparison
.venv\Scripts\python.exe research\v3_gate_check.py              # text signal vs the gate
.venv\Scripts\python.exe research\v3_longshort_probe.py         # the long-short numbers above
.venv\Scripts\python.exe research\v3_drift_rule.py              # ranking the picks + the hindsight ladder
.venv\Scripts\python.exe research\sp1500_gate.py              # the PEAD gate on the pooled S&P 1500
.venv\Scripts\python.exe research\longshort_sim.py            # the shorting grid: signal x cutoff x horizon
.venv\Scripts\python.exe research\voltarget_test.py           # the pre-registered vol-targeting test (FAIL)
.venv\Scripts\python.exe research\voltarget_intl.py           # the international replication (PASS)
.venv\Scripts\python.exe research\voltarget_multi.py          # the diversified book (construction)
.venv\Scripts\python.exe research\voltarget_erc.py            # correlation-aware weighting (rejected)
.venv\Scripts\python.exe research\voltarget_financing.py      # the financing curve (+1.7%/yr)
.venv\Scripts\python.exe research\voltarget_deep.py           # three fresh pre-registered tests (1 of 3 pass)
.venv\Scripts\python.exe research\voltarget_em.py             # EM cohort + the recent-regime test (both FAIL)
.venv\Scripts\python.exe research\voltarget_synthesis.py      # what separates passes from failures (post-hoc)
.venv\Scripts\python.exe research\bear_regimes.py             # can bear markets be seen in time? (no)
.venv\Scripts\python.exe research\noise_search.py             # what the gate passes when fed pure noise
```
