# Sentiment signal, v2 — design and pre-registration

Written 2026-09-17, before any filing has been downloaded. Replaces the v1 news-sentiment
pipeline (`Main.py`, `research/sentiment_events.py`) for the purpose of the fusion test.
v1 is not deleted — it stays as the news-scoring tool it is.

## Why v1 didn't work

Ranked by how much damage each does:

1. **It predicts tone, not returns.** FinBERT, finbert-tone and Twitter-RoBERTa were all
   trained on human sentiment labels. Nothing in the pipeline was ever fitted to predict a
   price move. Sounding positive and going up are different targets.
2. **It measures level, not surprise.** A stock with permanently upbeat coverage scores
   permanently bullish. Only deviation from a company's *own* norm carries information —
   the same transformation that made the earnings signal work.
3. **The ensemble is one model in three coats.** Two of the three share a base model; the
   third comes from a different domain entirely. Averaging correlated mediocre scorers
   costs compute and buys little. The temperature and neutral-bias constants tune the
   score distribution, not its predictive power.
4. **Headline-only, 192 tokens.** Guidance magnitude and specifics live in the body.
5. **No deduplication.** One syndicated story counted twenty times dominates the mean.
6. **No source discrimination.** Company press releases are positive by construction and
   Finnhub's free tier is mostly PR and aggregators.
7. **No relevance filter.** A market-wide story mentioning Apple scores as Apple sentiment.
8. **Three classes throw away magnitude.** "Raised guidance 12%" and "CEO feels good"
   both score +1.

Plus the fatal practical limit: Finnhub's free news is a rolling 12 months, which is what
made the original fusion test unanswerable.

## v2 design

### Source: SEC EDGAR 8-K filings

Every US company files an 8-K with Item 2.02 ("Results of Operations") when it reports
earnings, with the press release attached. Compared with news this gives:

- **complete coverage** — it is a legal requirement, not an editorial choice
- **exact timestamps** — filed to the minute, and the drift starts here
- **one company per document** — no relevance problem, no entity resolution
- **no duplicates, no PR-versus-journalism confusion** — it *is* the company's framing,
  which is a known, constant bias rather than an unknown, varying one
- **free and complete back to 2001** — we start at 2010 to match the membership data

Company identity comes from the SEC CIKs already stored in `universe_ids` for the ticker
rename fix, so no new identity work is needed. The SEC asks automated clients to declare a
contact; it is read from `git config user.email` at runtime and never written to the repo.

Verified end to end on 2026-09-17 before committing to the download:

- CIKs are on hand for all 503 current members; the submissions API answers 200.
- Apple's recent index alone holds 45 earnings 8-Ks; the EX-99.1 press release extracts
  cleanly (12,130 characters of "Apple reports third quarter results ...").
- The Loughran-McDonald word lists download without an account.

Two wrinkles found and handled:

- The API's `recent` block caps at 1,000 filings and only reaches 2015; older filings sit
  in shard files listed under `filings.files`, one extra request per company.
- The filing `index.json` `type` field returns icon names (`text.gif`), not document
  types, so the press release is located from the filing's index *page*, which carries a
  real type column, rather than guessed from the filename.

Volume: filings are fetched only where they match one of the 26,464 point-in-time earnings
events already built — roughly 53,000 requests at the SEC's 10/second limit, about 2-3
hours, and a few hundred MB of extracted text.

### Architecture: download once, featurize many times

Raw filing text is cached in its own database, so adding a feature later is a re-parse
rather than a re-download. This is the whole reason the feature set can stay open.

### Features (v2.0)

Chosen now:

- **Tone against the company's own norm** — Loughran-McDonald finance word lists (free),
  z-scored against that company's previous filings. The SUE transformation, applied to text.
- **Guidance and specificity** — whether guidance is raised, cut or dodged; how numeric
  and concrete the language is. Vagueness is informative.

Deferred, cheap to add later because the text is cached: language change versus the
company's own previous release (the *Lazy Prices* effect), filing latency versus its own
habit, non-GAAP emphasis, hedging and uncertainty density.

Each feature is layered rather than used raw: raw value → normalised against the company's
own history → normalised against the same quarter's cross-section → interacted with the
earnings surprise. The depth goes in the features, not in the model.

### Target: what the numbers don't already explain

The model predicts the 60-session excess return over SPY **with the earnings surprise
included as an input**, so text is only rewarded for information the numbers do not already
carry. A filter that merely rediscovers SUE is worthless.

### Model

Ridge or logistic regression over a small feature set, every coefficient inspectable. With
~26,000 events and a weak signal, a deep net fits noise faster than data. Gradient-boosted
trees may be tried only if the linear version passes first, and only as a reported
comparison.

### Validation (pre-registered)

- **Fit on 2010-2019. The 2020-2026 holdout is not touched until the end.** One look.
- Inference is date-clustered, as in every other gate here.
- **PASS** iff, in the untouched holdout, the text score predicts residual return with
  p < 0.05 and the correct sign, *and* sorting the top-5% SUE events by that score
  separates their outcomes.
- **FAIL** otherwise, and the fusion thesis is reported as unsupported rather than retuned.
- If a bug is found, it is fixed and disclosed; the rule does not change.

### What this does not authorise

- No change to `config/strategy.json` and no real money. A pass earns a portfolio-level
  test, nothing more.
- Loosening the top-5% entry cutoff to give the filter more events is a **separate**
  pre-registered question, to be asked only after the filter exists.
- Mid/small-cap filings are a later, second holdout — not part of this test.
