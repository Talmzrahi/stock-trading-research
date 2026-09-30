# Commit map

On 2026-09-30 the history was rewritten to remove third-party data before publication:

- `data_archive/event_articles.csv.gz` and `data_archive/event_fetch_progress.csv.gz`:
  41,569 news headlines and summaries with their sentiment scores, and the per-event fetch
  log, licensed from Finnhub, whose terms forbid redistributing its data or results
  derived from it.
- `data_archive/earnings.csv.gz`: EPS estimates and actuals downloaded from Yahoo Finance
  through yfinance, whose terms restrict redistribution too. `research/ingest.py`
  downloads them again.

Nothing else changed: same files, same dates, same messages.

Removing a file from history gives every later commit a new ID, and 82 of the
86 commits changed. Commit messages and documents written
before the rewrite still cite the original IDs (for example "committed first, fbddc02");
this table translates them. Commits made before the data was added kept their IDs.

| Date | Original ID | Current ID | Commit |
|---|---|---|---|
| 2026-09-14 | `e11a2ea` | `e11a2ea` | Initial commit: local-runnable sentiment pipeline + hypothesis test harness |
| 2026-09-14 | `cce18b5` | `cce18b5` | Fix backtest to test the interaction, not the conflated contrast |
| 2026-09-14 | `b45a6b2` | `b45a6b2` | Add pre-earnings news sentiment collection for the fusion test |
| 2026-09-14 | `e28e2a0` | `e28e2a0` | Add fusion hypothesis test with pre-registered direction |
| 2026-09-14 | `183d1c2` | `1c89af7` | Record Phase 1 results and archive the perishable news corpus |
| 2026-09-14 | `4cdb279` | `b7bd874` | Re-specify PEAD with SUE deciles and long horizons |
| 2026-09-15 | `00393cc` | `c43b3c0` | Add point-in-time universe and re-run PEAD against it |
| 2026-09-15 | `a7fb914` | `a0f2d43` | Re-test PEAD with point-in-time decile cutoffs: PASS |
| 2026-09-15 | `b38916e` | `16e357e` | Build layers 2-7 core and select the strategy on the portfolio gate |
| 2026-09-15 | `91c5a0e` | `b7258fa` | Add live daily runner, simulated-account persistence and reports (WIP) |
| 2026-09-15 | `9e8b772` | `3b26046` | Update CLAUDE.md for Phase 2 and record verified Alpaca constraints |
| 2026-09-15 | `e3434a3` | `d12f629` | Record paper account start and scheduled daily run |
| 2026-09-15 | `77a2a75` | `0eff7df` | Stress-test the strategy and repair the survivors-only universe: gate FAILS |
| 2026-09-15 | `bc09e15` | `573b989` | Pre-register out-of-sample test of top-5% surprise drift in S&P 400/600 |
| 2026-09-15 | `ac9e508` | `17fbfdb` | Wire the Alpaca paper broker into the daily run; add S&P 400/600 test scripts |
| 2026-09-15 | `da6407b` | `6336fb1` | Pre-register the conditional top-5% re-specification; add price retry pass |
| 2026-09-15 | `174fc1f` | `923996a` | Out-of-sample test PASSES; re-specify the strategy at top 5% + 12sd stop |
| 2026-09-16 | `dc345e6` | `b538b1f` | Add README and archive index membership for reproducibility |
| 2026-09-16 | `1b5435e` | `c2759a8` | Quantify the survivorship hole; find the edge lives only in volatile stocks |
| 2026-09-17 | `b9ea452` | `92e309d` | Design and pre-register the v2 sentiment signal on SEC filings |
| 2026-09-17 | `a09e3f3` | `2b00c9c` | Add the EDGAR 8-K downloader for the v2 sentiment signal |
| 2026-09-17 | `739c4ba` | `fd6b0fd` | Add the 8-K featuriser; strip safe-harbor boilerplate from tone |
| 2026-09-17 | `60c2da2` | `7984a81` | Add the pre-registered sentiment gate, with the holdout protected in code |
| 2026-09-17 | `fedcde1` | `6bb9452` | Add the N-year replay against SPY |
| 2026-09-17 | `33273cf` | `dd8fe8b` | Fetch press releases in parallel; the download was latency-bound |
| 2026-09-17 | `24b3f7c` | `c7b4af5` | Halve requests per filing: fetch the combined submission, not index+exhibit |
| 2026-09-18 | `491e290` | `8df520e` | Add language-change and filing-latency features; fix gate ridge and binning |
| 2026-09-18 | `fea6279` | `fa162a3` | Register the final eight text features before the holdout look |
| 2026-09-18 | `02b0d0b` | `9a71a0e` | Record the v2 sentiment verdict: FAIL on the 2020-2026 holdout |
| 2026-09-18 | `fbddc02` | `fabfc0a` | Pre-register an overlap-robust re-check of the PEAD evidence |
| 2026-09-18 | `7fd64bf` | `0c26c81` | Add the overlap-robust inference check, before running it |
| 2026-09-18 | `f734e42` | `95bb6b5` | Record the overlap-robust re-check: the PEAD signal does not survive |
| 2026-09-18 | `53c9189` | `e426d4e` | Start v3 sentiment: structured releases and layer 0 (what is new) |
| 2026-09-18 | `023504c` | `3a2cf45` | Add v3 reaction labels; fix the gap trade's timing |
| 2026-09-18 | `14690e4` | `fdae8b1` | Prepare the S&P 400/600 exam set: --set midsmall in downloader and layer 0 |
| 2026-09-19 | `a864eff` | `122133e` | Keep the daily run alive on battery; log each run's start and exit |
| 2026-09-19 | `75b6435` | `04d1d70` | Add the v3 end-to-end prototype and its first results |
| 2026-09-19 | `e601cd5` | `16cd296` | Record the full layer 0 run: 1.3M new or edited sentences to read |
| 2026-09-19 | `a6afcb4` | `2683ac0` | Record the owner's rule: the project stays free, so no paid LLM teacher |
| 2026-09-19 | `82a2615` | `dda7280` | Record the owner's idea: model disagreement and model bias as signals |
| 2026-09-19 | `09e85f8` | `f6bfbaa` | Make overlap-robust, market-adjusted inference part of the mandatory gate |
| 2026-09-19 | `df5512b` | `856249d` | Add the layer 1 reader comparison: five free readers on 2,000 releases |
| 2026-09-19 | `e9382d6` | `209e5c1` | Record full-data prototype and first reader comparison results |
| 2026-09-19 | `43b2bfe` | `05fe33f` | Record measured power: 2,000 releases detect a real text effect 8% of the time |
| 2026-09-19 | `d1f6b97` | `f0c9fe3` | Grow the reader comparison to 8,000 releases, three readers |
| 2026-09-19 | `6dbfe92` | `2b062e1` | Measure parallel readers: no gain on this laptop, keep them sequential |
| 2026-09-19 | `84a7c00` | `d9e8b32` | Record the S&P 400/600 exam-set download: 27,164 releases, text only |
| 2026-09-20 | `d39a897` | `c66aa43` | Cap reading at the first 20 changed sentences; MiniLM beats the word list |
| 2026-09-20 | `716b9e2` | `84c6343` | Reader comparison with capped reading: the finance mood model leads |
| 2026-09-20 | `762c7ef` | `92f9017` | Reader comparison complete: DistilRoBERTa wins, disagreement pays, idea 2 dead |
| 2026-09-20 | `94fb583` | `c872636` | Probe for a tradable signal: the gap is dead, continuation is borderline |
| 2026-09-20 | `e71e368` | `285c711` | Move the release parser and layer 0 into trader/text |
| 2026-09-20 | `78888ae` | `6c28994` | Add the production reader and the fitted text model |
| 2026-09-20 | `f7d7e1e` | `69eb070` | Build the text signal into the daily run, on shadow books |
| 2026-09-20 | `277e30f` | `2fe1b85` | Fix the shadow account settling into the future |
| 2026-09-20 | `eeeacfe` | `ba01b26` | Document the text signal and its shadow account in CLAUDE.md |
| 2026-09-20 | `756b1ae` | `1b5c540` | Fix two mangled command lines in CLAUDE.md |
| 2026-09-20 | `1c1f77d` | `2915123` | Record the v3 shadow build in ROADMAP and the design doc |
| 2026-09-20 | `19ddef2` | `a260531` | Point the design doc at the parser's new home |
| 2026-09-21 | `78bb78b` | `3046d8e` | Full development data: the reader improves, the long-only trade does not |
| 2026-09-21 | `d70f95e` | `b03aabf` | Refit on all 23,997 releases; stamp each score with its model |
| 2026-09-21 | `6667896` | `5bba425` | Archive the full scored sample: all 23,997 development releases |
| 2026-09-21 | `c9d0a74` | `b68aace` | Add PROJECT_STATE.md and measure the long-short version |
| 2026-09-22 | `5583b50` | `fc9a148` | Rank the picks we already take: three kinds of hindsight, each worth half |
| 2026-09-22 | `92b03bc` | `88ac3f2` | Pool the S&P 1500: 2.7x the events, zero extra quarters |
| 2026-09-22 | `1e57b45` | `b03496d` | Repair the point-in-time universe: 2012-2014 was silently missing |
| 2026-09-22 | `02efc8b` | `1c4733e` | Correct the record: the weaker numbers were my bug, not correct data |
| 2026-09-22 | `33dcdee` | `554ed0a` | Simulate shorting across 27 configs, and fix a cost sign bug |
| 2026-09-22 | `1094a34` | `463da31` | Phase 0: prices and volatility indices for the vol-targeting work |
| 2026-09-22 | `2eb5b85` | `1a27427` | Pre-register the volatility-targeting test before touching the holdout |
| 2026-09-22 | `b9cf005` | `66681eb` | Volatility targeting: pre-registered test FAILS on the 1993-2001 holdout |
| 2026-09-22 | `e584e7f` | `6b322c3` | Pre-register the international replication before scoring any return |
| 2026-09-22 | `a16e24a` | `d1c538d` | International replication PASSES all five — and half of it is not timing |
| 2026-09-23 | `6eef8bf` | `c200363` | Build the diversified book: SPY's return at a third less drawdown |
| 2026-09-23 | `aee31ce` | `ffdf910` | Three improvement findings: one rejected, one real, one correction |
| 2026-09-23 | `47830d9` | `fbcc2e3` | Correlation-aware weighting: rejected, and the correlations are the harm |
| 2026-09-23 | `788285e` | `46db9d5` | Share the backtest mechanics, with tests; correct the understated record |
| 2026-09-23 | `6d9e57d` | `df17222` | Fresh data for three tests, pre-registered before any of it is scored |
| 2026-09-23 | `393cd34` | `50874fa` | Three fresh tests: 1 of 3 pass, and a pre-registered prediction fails |
| 2026-09-23 | `ddc72cd` | `2234ea5` | Pre-register the EM cohort and the recent-regime test |
| 2026-09-23 | `5950c72` | `697e274` | Recent data answers it: the recent regime does not work, as predicted |
| 2026-09-24 | `87c6479` | `0ed96e0` | Synthesis: the mechanism explains the verdicts, and one pass was raw Sharpe |
| 2026-09-24 | `169a438` | `0c1a756` | The mechanism explains the past and cannot predict the future |
| 2026-09-24 | `ea01ba9` | `936aacc` | Bear markets can't be seen in time; noise passes the gate when searched |
| 2026-09-30 | `0542d82` | `62482ab` | Clean up the snapshot's documents and leftovers for publication |
| 2026-09-30 | `2f8af63` | `bd2c662` | Add the charts and a findings write-up; rename to stock-trading-research |
