# data_archive

Data committed to git because it cannot be downloaded again, or took hours to compute.
`research/archive.py restore` loads the tables back into `data/research.db` and
`data/oos_midsmall.db`; a missing file is skipped.

| File | What it holds | Source |
|---|---|---|
| `universe_history`, `universe_ids` | S&P 500 membership by quarter, with SEC company IDs | Wikipedia revision history, SEC |
| `oos_*` | the same for the S&P 400/600, plus price coverage | Wikipedia revision history, SEC, Yahoo |
| `departed_coverage` | price coverage of firms that left the S&P 500 | Yahoo |
| `loughran_mcdonald` | the Loughran-McDonald finance word lists used by the v2 test | Loughran & McDonald |
| `v3_reader_sample`, `v3_reader_features` | which releases the text readers scored, and their outputs | SEC EDGAR 8-K filings |
| `v3_*.txt` | archived outputs of the text-model runs | this project |

**Removed on 2026-09-30,** from the whole history (see `docs/COMMIT_MAP.md`):

- the Finnhub news behind the Phase 1 test (41,569 headlines and summaries with their
  sentiment scores, and the per-event fetch log). Finnhub's terms forbid redistributing its
  data or results derived from it.
- `earnings`, the EPS estimates and actuals from Yahoo Finance, whose terms restrict
  redistribution too. `research/ingest.py` downloads them again.

The Phase 1 result stays documented in
`ROADMAP.md`; rerunning it needs your own Finnhub key and `research/sentiment_events.py`,
and Finnhub's free tier only reaches back about a year.
