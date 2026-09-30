# Commit map

On 2026-09-30 the history was rewritten to remove two files that held data licensed from
Finnhub, whose terms forbid redistributing it or results derived from it:
`data_archive/event_articles.csv.gz` (41,569 news headlines and summaries, with their
sentiment scores) and `data_archive/event_fetch_progress.csv.gz` (article counts per
earnings event). Nothing else changed: same files, same dates, same messages.

Removing a file from history gives every later commit a new ID, and 82 of the
86 commits changed. Commit messages and documents written before the rewrite
still cite the old IDs (for example "committed first, fbddc02"); this table translates them.
Commits made before the news files were added kept their IDs.

| Date | Old ID | New ID | Commit |
|---|---|---|---|
| 2026-09-14 | `e11a2ea` | `e11a2ea` | Initial commit: local-runnable sentiment pipeline + hypothesis test harness |
| 2026-09-14 | `cce18b5` | `cce18b5` | Fix backtest to test the interaction, not the conflated contrast |
| 2026-09-14 | `b45a6b2` | `b45a6b2` | Add pre-earnings news sentiment collection for the fusion test |
| 2026-09-14 | `e28e2a0` | `e28e2a0` | Add fusion hypothesis test with pre-registered direction |
| 2026-09-14 | `183d1c2` | `b2a2a77` | Record Phase 1 results and archive the perishable news corpus |
| 2026-09-14 | `4cdb279` | `4a5169e` | Re-specify PEAD with SUE deciles and long horizons |
| 2026-09-15 | `00393cc` | `a335029` | Add point-in-time universe and re-run PEAD against it |
| 2026-09-15 | `a7fb914` | `97fdf43` | Re-test PEAD with point-in-time decile cutoffs: PASS |
| 2026-09-15 | `b38916e` | `d3de71f` | Build layers 2-7 core and select the strategy on the portfolio gate |
| 2026-09-15 | `91c5a0e` | `2bb0595` | Add live daily runner, simulated-account persistence and reports (WIP) |
| 2026-09-15 | `9e8b772` | `af25fec` | Update CLAUDE.md for Phase 2 and record verified Alpaca constraints |
| 2026-09-15 | `e3434a3` | `1d54503` | Record paper account start and scheduled daily run |
| 2026-09-15 | `77a2a75` | `d8d7ff8` | Stress-test the strategy and repair the survivors-only universe: gate FAILS |
| 2026-09-15 | `bc09e15` | `6d55152` | Pre-register out-of-sample test of top-5% surprise drift in S&P 400/600 |
| 2026-09-15 | `ac9e508` | `7c2c0ea` | Wire the Alpaca paper broker into the daily run; add S&P 400/600 test scripts |
| 2026-09-15 | `da6407b` | `cbd940f` | Pre-register the conditional top-5% re-specification; add price retry pass |
| 2026-09-15 | `174fc1f` | `b7bcc11` | Out-of-sample test PASSES; re-specify the strategy at top 5% + 12sd stop |
| 2026-09-16 | `dc345e6` | `91a4b91` | Add README and archive index membership for reproducibility |
| 2026-09-16 | `1b5435e` | `583433a` | Quantify the survivorship hole; find the edge lives only in volatile stocks |
| 2026-09-17 | `b9ea452` | `5d0d5a0` | Design and pre-register the v2 sentiment signal on SEC filings |
| 2026-09-17 | `a09e3f3` | `72cd7d7` | Add the EDGAR 8-K downloader for the v2 sentiment signal |
| 2026-09-17 | `739c4ba` | `9e25d01` | Add the 8-K featuriser; strip safe-harbor boilerplate from tone |
| 2026-09-17 | `60c2da2` | `368a182` | Add the pre-registered sentiment gate, with the holdout protected in code |
| 2026-09-17 | `fedcde1` | `0dc085d` | Add the N-year replay against SPY |
| 2026-09-17 | `33273cf` | `bc263f5` | Fetch press releases in parallel; the download was latency-bound |
| 2026-09-17 | `24b3f7c` | `eeae400` | Halve requests per filing: fetch the combined submission, not index+exhibit |
| 2026-09-18 | `491e290` | `014557d` | Add language-change and filing-latency features; fix gate ridge and binning |
| 2026-09-18 | `fea6279` | `86e0547` | Register the final eight text features before the holdout look |
| 2026-09-18 | `02b0d0b` | `1b40120` | Record the v2 sentiment verdict: FAIL on the 2020-2026 holdout |
| 2026-09-18 | `fbddc02` | `923ec85` | Pre-register an overlap-robust re-check of the PEAD evidence |
| 2026-09-18 | `7fd64bf` | `150b468` | Add the overlap-robust inference check, before running it |
| 2026-09-18 | `f734e42` | `795245d` | Record the overlap-robust re-check: the PEAD signal does not survive |
| 2026-09-18 | `53c9189` | `9f89dc9` | Start v3 sentiment: structured releases and layer 0 (what is new) |
| 2026-09-18 | `023504c` | `10b4bcc` | Add v3 reaction labels; fix the gap trade's timing |
| 2026-09-18 | `14690e4` | `35c749a` | Prepare the S&P 400/600 exam set: --set midsmall in downloader and layer 0 |
| 2026-09-19 | `a864eff` | `8ef1383` | Keep the daily run alive on battery; log each run's start and exit |
| 2026-09-19 | `75b6435` | `6ee5a15` | Add the v3 end-to-end prototype and its first results |
| 2026-09-19 | `e601cd5` | `dccc04b` | Record the full layer 0 run: 1.3M new or edited sentences to read |
| 2026-09-19 | `a6afcb4` | `62a6f7a` | Record the owner's rule: the project stays free, so no paid LLM teacher |
| 2026-09-19 | `82a2615` | `1409dfa` | Record the owner's idea: model disagreement and model bias as signals |
| 2026-09-19 | `09e85f8` | `f766c54` | Make overlap-robust, market-adjusted inference part of the mandatory gate |
| 2026-09-19 | `df5512b` | `40fcc04` | Add the layer 1 reader comparison: five free readers on 2,000 releases |
| 2026-09-19 | `e9382d6` | `5d59989` | Record full-data prototype and first reader comparison results |
| 2026-09-19 | `43b2bfe` | `c064b83` | Record measured power: 2,000 releases detect a real text effect 8% of the time |
| 2026-09-19 | `d1f6b97` | `8665c36` | Grow the reader comparison to 8,000 releases, three readers |
| 2026-09-19 | `6dbfe92` | `79267fa` | Measure parallel readers: no gain on this laptop, keep them sequential |
| 2026-09-19 | `84a7c00` | `b2c0b2c` | Record the S&P 400/600 exam-set download: 27,164 releases, text only |
| 2026-09-20 | `d39a897` | `dbcd224` | Cap reading at the first 20 changed sentences; MiniLM beats the word list |
| 2026-09-20 | `716b9e2` | `e95f6ce` | Reader comparison with capped reading: the finance mood model leads |
| 2026-09-20 | `762c7ef` | `07b8789` | Reader comparison complete: DistilRoBERTa wins, disagreement pays, idea 2 dead |
| 2026-09-20 | `94fb583` | `cfd0408` | Probe for a tradable signal: the gap is dead, continuation is borderline |
| 2026-09-20 | `e71e368` | `3ccd45a` | Move the release parser and layer 0 into trader/text |
| 2026-09-20 | `78888ae` | `53ab2c2` | Add the production reader and the fitted text model |
| 2026-09-20 | `f7d7e1e` | `eb8137e` | Build the text signal into the daily run, on shadow books |
| 2026-09-20 | `277e30f` | `fc98897` | Fix the shadow account settling into the future |
| 2026-09-20 | `eeeacfe` | `a511c9f` | Document the text signal and its shadow account in CLAUDE.md |
| 2026-09-20 | `756b1ae` | `669be38` | Fix two mangled command lines in CLAUDE.md |
| 2026-09-20 | `1c1f77d` | `8b9cfa1` | Record the v3 shadow build in ROADMAP and the design doc |
| 2026-09-20 | `19ddef2` | `2fa7a6c` | Point the design doc at the parser's new home |
| 2026-09-21 | `78bb78b` | `620dd1a` | Full development data: the reader improves, the long-only trade does not |
| 2026-09-21 | `d70f95e` | `33e3398` | Refit on all 23,997 releases; stamp each score with its model |
| 2026-09-21 | `6667896` | `28afc13` | Archive the full scored sample: all 23,997 development releases |
| 2026-09-21 | `c9d0a74` | `520a7d9` | Add PROJECT_STATE.md and measure the long-short version |
| 2026-09-22 | `5583b50` | `89c675c` | Rank the picks we already take: three kinds of hindsight, each worth half |
| 2026-09-22 | `92b03bc` | `dc5f4fa` | Pool the S&P 1500: 2.7x the events, zero extra quarters |
| 2026-09-22 | `1e57b45` | `1d98ef6` | Repair the point-in-time universe: 2012-2014 was silently missing |
| 2026-09-22 | `02efc8b` | `51ac906` | Correct the record: the weaker numbers were my bug, not correct data |
| 2026-09-22 | `33dcdee` | `e21fa0c` | Simulate shorting across 27 configs, and fix a cost sign bug |
| 2026-09-22 | `1094a34` | `06d3ad2` | Phase 0: prices and volatility indices for the vol-targeting work |
| 2026-09-22 | `2eb5b85` | `4389dd4` | Pre-register the volatility-targeting test before touching the holdout |
| 2026-09-22 | `b9cf005` | `bb57edc` | Volatility targeting: pre-registered test FAILS on the 1993-2001 holdout |
| 2026-09-22 | `e584e7f` | `cdc1fa7` | Pre-register the international replication before scoring any return |
| 2026-09-22 | `a16e24a` | `86832a7` | International replication PASSES all five — and half of it is not timing |
| 2026-09-23 | `6eef8bf` | `5b37264` | Build the diversified book: SPY's return at a third less drawdown |
| 2026-09-23 | `aee31ce` | `70e4703` | Three improvement findings: one rejected, one real, one correction |
| 2026-09-23 | `47830d9` | `719d365` | Correlation-aware weighting: rejected, and the correlations are the harm |
| 2026-09-23 | `788285e` | `fda8c06` | Share the backtest mechanics, with tests; correct the understated record |
| 2026-09-23 | `6d9e57d` | `afa80ef` | Fresh data for three tests, pre-registered before any of it is scored |
| 2026-09-23 | `393cd34` | `ee43f6b` | Three fresh tests: 1 of 3 pass, and a pre-registered prediction fails |
| 2026-09-23 | `ddc72cd` | `bf94382` | Pre-register the EM cohort and the recent-regime test |
| 2026-09-23 | `5950c72` | `ecd51d4` | Recent data answers it: the recent regime does not work, as predicted |
| 2026-09-24 | `87c6479` | `2422c4a` | Synthesis: the mechanism explains the verdicts, and one pass was raw Sharpe |
| 2026-09-24 | `169a438` | `9b3e2e6` | The mechanism explains the past and cannot predict the future |
| 2026-09-24 | `ea01ba9` | `399dd7d` | Bear markets can't be seen in time; noise passes the gate when searched |
| 2026-09-30 | `0542d82` | `d27d29e` | Clean up the snapshot's documents and leftovers for publication |
| 2026-09-30 | `2f8af63` | `8cfa7ad` | Add the charts and a findings write-up; rename to stock-trading-research |
