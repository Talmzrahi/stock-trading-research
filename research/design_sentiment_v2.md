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

## Final feature set (v2.1): registered 2026-09-18, before the holdout is read

This section fixes the test. It is committed before any run of the gate on these eight
features, in dev or final mode. The only gate output seen so far was the dev run on the
original four (train only, top-minus-bottom quintile +0.735pp, unclustered p=0.032, tone
coefficient negative). After that run the owner chose to add three of the four deferred
features before spending the holdout. That is allowed because dev never reads 2020-2026.

| Feature | What it measures | Scale |
|---|---|---|
| `tone_z` | Loughran-McDonald (pos−neg)/(pos+neg+1), narrative only | robust z vs own history |
| `guide_dir` | (raises − cuts)/(raises + cuts + 1) within guidance sentences | raw |
| `guide_share` | share of sentences that are forward-looking | raw |
| `guide_numeric` | guidance contains a number (0/1) | raw |
| `sim_prev` | cosine similarity of the narrative to the company's previous release | raw |
| `sim_year` | the same, against the release four back (same quarter last year) | raw |
| `days_since_prev_z` | days since the company's previous earnings release (latency) | robust z vs own history |
| `nongaap_density_z` | non-GAAP / "adjusted" mentions per narrative word | robust z vs own history |

"Robust z vs own history" means median and MAD over the company's 12 most recent *prior*
releases, with at least 6 required, capped at ±5. Hedging/uncertainty density stays deferred.

**Why the similarity features are raw.** Similarity is already a comparison with the
company's own previous text, so it is a change measure by construction. Z-scoring the
level features exists to turn them into change measures. Z-scoring similarity again would
measure how unusual this quarter's *rate* of rewriting is, which is not what *Lazy
Prices* tested, and it would require 7+ prior releases. This was decided on principle,
before any results existed for either version.

**Model, sample, verdict.** These are unchanged from the committed gate. A baseline ridge
predicts the 60-session excess return over SPY from `sue` and `conviction`. A text ridge
(α=1, standardised features) predicts that baseline's residual from the eight features.
Both are fitted on training events only. The sample is every point-in-time S&P 500
event that has all eight features, which in practice means firms with 7+ prior releases.
The operational PASS rule:

1. In the holdout, the highest text-score quintile minus the lowest, in mean residual
   return, is **> 0 with p < 0.05**, and
2. among holdout events the strategy actually trades (conviction ≥ the config cutoff, 0.95;
   at least 100 events), the better-scored half minus the worse half is **> 0 with p < 0.05**.

Both tests are date-clustered. Anything else is FAIL. No retuning follows.

### Corrections made before the look

These were found while reviewing the gate before registration. They are disclosed under the
rule "if a bug is found, it is fixed and disclosed; the rule does not change". Each one
removes leakage, makes the test stricter, or restores data the code was dropping by
mistake. None came from looking at returns.

1. **Clustering was missing from the verdict.** The design above says inference is
   date-clustered. The code's two verdict tests were plain Welch two-sample t-tests, which
   treat events entered on the same day as independent. Each test is now a
   difference-in-means regression with standard errors clustered by entry date (CR1, t on
   G−1 df). The estimate matches statsmodels exactly; the p-value is slightly more
   conservative. The dev figure p=0.032 quoted above predates this fix.
2. **Look-ahead in filing timing.** Filings are matched to events within ±2 days, and 370
   of 26,449 matches (1.4%; 3% in 2010, under 1% after 2021) were accepted by the SEC
   after the 14:30 ET decision on the entry day. The live system could not have read them,
   so they are now dropped.
3. **No embargo between train and holdout.** Training events entered in late 2019 hold into
   2020, so their targets shared prices with the holdout, including the COVID crash. Events
   entered before 2020 whose 60-session window ends after 2019-12-31 are now used by neither
   side.
4. **The own-history norm needed 12 prior releases, not 6.** `shift()` leaves a blank in
   the first slot of every rolling window, and `np.median` of a window containing a blank
   is blank. So no company got a z-score until the window slid past that slot, at its
   13th release, whatever `MIN_PRIOR` said. The dev run on the original four was therefore
   limited to firms with 12+ prior releases. The window functions now skip blanks, so
   scoring starts at the 7th release as documented. This adds 2011-2013 events to
   training, plus the early releases of firms that joined later.
5. **A zero spread was treated as missing.** Some firms file the same template every
   quarter: BXP's tone is identical in all 66 releases, and filing gaps of exactly 91 days
   repeat. With MAD = 0 the z-score came out blank. After fix 4 this still dropped 1%
   (tone), 15% (non-GAAP) and 17% (latency) of firms with 7+ prior releases. Now a value
   equal to the norm scores 0. A departure is scaled by the mean absolute deviation
   (×1.2533) instead. A departure from a norm that has never varied scores the ±5 cap.
   After fixes 4 and 5, none of the three features is missing for any firm with 7+ prior
   releases. Both fixes are covered by `tests/test_edgar_features.py`.

## Result (2026-09-18): FAIL

The gate was run exactly as committed in `fea6279`: `--dev` once, then `--final` once.
There are 21,686 events with all eight features: 10,012 train (2012-2019), 390 embargoed,
11,284 holdout (2020-01 to 2026-06). Another 317 were dropped as unreadable at the decision time.

Text coefficients, fitted on train (pp of unexplained 60-session return per sd):
latency `days_since_prev_z` +0.258, `guide_share` +0.131, `guide_numeric` −0.120,
`tone_z` −0.111, `sim_year` +0.076, `sim_prev` +0.058, `guide_dir` +0.017,
`nongaap_density_z` −0.005.

| | Train (in-sample) | Holdout (never fitted) |
|---|---|---|
| correlation, score vs unexplained return | +0.036 | **+0.003** |
| lowest quintile | −0.333pp | −0.394pp |
| 2 | −0.208pp | −0.596pp |
| 3 | −0.031pp | −0.895pp |
| 4 | +0.240pp | −1.039pp |
| highest quintile | +0.428pp | −0.206pp |
| **highest − lowest** | +0.855pp (p=0.009) | **+0.431pp (p=0.384)** |

Among the 576 holdout events the strategy actually trades (top 5% SUE), the better-scored
half returned +1.946pp and the worse half +2.782pp (p=0.618). That is the wrong direction.

**Verdict: FAIL on both conditions.** The in-sample quintiles were monotone. The holdout
quintiles are not, and the correlation is effectively zero. This is what an in-sample fit
to noise looks like. The fusion thesis, as tested on company-authored earnings text, is
**unsupported**. Per the rules above it is not retuned: no new features, no other model,
no second look at 2020-2026.

What this does **not** settle:

- Text written by others about the company (news, analysts, social media) is a different
  hypothesis. v1 could not test it (a 12-month window), and v2 did not.
- Mid/small-cap filings were reserved as a second holdout. With the S&P 500 result a
  clean zero, spending them on the same model has little prior support.
- Anything built from these features would need a new pre-registration on new data.
