# Sentiment signal, v3: design

Started 2026-09-18. v2 (word lists and regexes on 8-K releases) failed its one-shot
holdout, and v1 (the news ensemble in `Main.py`) was never testable. v3 is a new reader,
built in layers, with every layer's job checkable. This file records decisions as they
are made. The final test will get its own pre-registration before any of its data is used.

## What the review of v1 measured (2026-09-18)

These are measured on the 41,569 articles v1 scored, not argued. They set the priorities.

| Check | Result | Meaning |
|---|---|---|
| Temperature/bias tuning vs untuned | rank correlation 0.993-0.999; 7% of labels flip | cosmetic: the numbers look different, but the article ordering barely changes |
| Stated weights 34/33/33 | actual influence 37/42/22 | the temperatures silently cut Twitter-RoBERTa by a third |
| The three models | correlation 0.70; labels agree 53% | related but not copies. This corrects v2's "one model in three coats" |
| Relevance | 24% of articles filed under 2+ tickers | a quarter is market-wide chatter |
| Sentiment vs the earnings surprise | −0.002 | nothing |
| Sentiment vs the announcement-day move | −0.020 | nothing |
| Sentiment vs the stock's own prior 10-day return | **+0.14** (p < 1e-7) | news narrates the price. It is an echo. |
| Speed on this PC (CPU) | about 2 scorings/second per BERT-size model | a million paragraphs would take about 6 days per model |

The root problem is not the output tampering. It is that nothing in v1 was ever fitted to
predict a price, and that what it read mostly repeats the price.

## Owner's decisions (2026-09-18)

1. **Two ideas carry v3:**
   - **The market labels the text.** The top layer learns from what the stock did right
     after the release, not from human opinion labels.
   - **Trade the gap.** The signal is the text-implied reaction minus the actual reaction
     at trade time. The market under-reacting is the drift the project is built on.
2. **The source is earnings press releases** (8-K EX-99), not news. They are timestamped
   before the reaction, have 15 years of history, and cover one company per document.
   With news, "the market labels the text" would teach the echo, because much news is
   written after the move it describes. News may return later, for live use only.
3. **The releases are re-downloaded with their structure** (`release_html`, below).
4. **The architecture is layered:** big pretrained models frozen at the bottom, small
   learned layers at the top, and no hand-set constants between layers. Start with layer 0.

## Architecture

| Layer | Job | Learned? | Checkable by |
|---|---|---|---|
| 0 | cut each release into units; label each vs the company's own previous releases: boilerplate / template / edited / new | no | reading the output: no returns involved |
| 1 | read the units that matter: a fast embedding per unit, the three v1 models' raw logits (no temperatures), numbers from tables and guidance | frozen models | extraction accuracy on samples |
| 2 | per release: pool unit readings, weighted by novelty; compare with the company's history | no / simple | stability, coverage |
| 3 | predict the announcement reaction from text + surprise | **yes**, small, training years only | out-of-fold fit |
| 4 | gap = predicted − actual reaction at entry | no | the pre-registered exam |

### Timing (fixed 2026-09-18, `research/v3_labels.py`)

**Reaction** (the label for idea 1): the stock's return minus SPY from **the last close
before the announcement** (the same day's close for after-close releases, the previous
day's otherwise) to the entry close in `trader/events.py`. Only post-announcement moves
fall inside the window: 1 session for 99.5% of events, 2 for intraday releases. This is
fixed now so later layers cannot drift toward whatever fits.

**Gap** (idea 2): predicted reaction − actual reaction. **Correction to the first draft,
which said "known at the entry close":** the daily run decides at 14:30 ET, and there is
no free intraday price history. The reaction is therefore only complete after the entry
close, so **a gap trade enters at the next session's close** (`gap_entry_idx`). That
costs one day of drift but has no look-ahead. The release must be public by 14:30 ET on
that day (`readable`; true for 99.9%). Layer 3 must be cross-fitted, meaning each event's
prediction comes from a model that never saw that event, or the gap is zero by construction.

**Positive control (passed):** 26,131 S&P 500 events. The reaction's rank correlation
with the earnings surprise is +0.28 overall, and +0.30 / +0.24 / +0.18 for pre-open /
after-close / intraday releases. Only 7 announcements carry a date-only timestamp. The
median absolute reaction is 2.85pp.

## Data

- `data/edgar.db`, table `release_html`: each release's EX-99 exhibit as compact HTML.
  Styling is stripped; paragraphs, tables, bold and superscripts are kept. It is the same
  document v2 read (100% word overlap checked on samples), and about 350 MB for all
  26,358. Parser: `research/release_text.py`, with tests in `tests/test_release_text.py`.
  - Why: v2's stored text was one flattened line per release. Paragraphs were merged,
    tables were smeared into number strings, and old Windows-1252 quotes and dashes were
    deleted. Layer 0 needs real pieces.
  - Download route: index page + exhibit. The combined submission file is built on
    demand for older filings and took up to 18 s each.
- `data/v3.db`, table `layer0`: per release, the counts per class and the units themselves.
- `data/v3.db`, table `labels`: per S&P 500 event, the reaction and timing above. The
  S&P 400/600 exam set is not labelled before its pre-registration.

## Compute budget for layer 1 (measured 2026-09-18)

Real new and edited release sentences have a median of 34 tokens (90th percentile 63).
On this PC's CPU, batched and length-sorted:

- FinBERT at full precision: **18 sentences/s**
- FinBERT compressed to 8-bit: **9 sentences/s**, and it disagrees with full precision
  (rank correlation 0.91), so it is rejected

At roughly 1.3 million new and edited sentences, that is **about 20 hours per BERT-size
model**, and about 60 for all three v1 models. Layer 1 therefore needs a small, fast
reader or the big-teaches-small route. The exact sentence count comes from the full
layer 0 run.

## Layer 0 (`research/v3_layer0.py`)

**Units:** each paragraph sentence (not split after "Inc." / "U.S." / initials), short
heading-like fragments, and each table row.

**Classes**, against the company's previous `PRIOR_K` releases, strictly earlier:

- **boilerplate:** the identical text appeared before
- **template:** the same words, with only numbers, months or quarter names changed. The
  news in these units is in the numbers, which layer 1 extracts.
- **edited:** word-pair Jaccard ≥ `EDITED` with some earlier unit
- **new:** nothing close

Within-release repeats (a headline bullet restated in the body) are flagged, not dropped.

### Decisions made without the owner (technical, reversible)

- `PRIOR_K = 4`: one year of history, which includes the same quarter last year.
- `EDITED = 0.45`. It is the low point of the best-match similarity histogram for
  sentences that are neither boilerplate nor template: 163,880 sentences from 3,081
  mature releases of 98 companies, in 0.1 bins. The count falls from 26,908 (0.1-0.2) to
  its minimum of 12,226 (0.4-0.5), then rises to a second cluster over 0.5-0.9. This uses
  text only, no returns. The fine 0.05 bins are jagged, because short sentences give
  Jaccard ratios like 1/2 and 2/3, so the 0.1 bins were used.
- **Edited sentences are read by later layers, not only new ones.** Examples above 0.5
  include "higher mobile and server microprocessor revenues were partially offset by lower
  desktop revenue": last quarter's sentence with its meaning changed. So the cutoff only
  separates "rewritten" from "brand new", and nothing is discarded at the boundary. Only
  boilerplate is skipped. Template units pass their numbers on, not their words.
- Word pairs made only of placeholders ("# #") are ignored. After number masking, every
  table row shares them, which made each comparison all-rows-against-all-rows (2 s per
  release instead of 0.08 s) and said nothing about identity.

An earlier trial on 32 companies (1,112 mature releases), before the placeholder fix:

| Unit | boilerplate | template | edited | new |
|---|---|---|---|---|
| sentences | 43% | 14% | 16% | 26% |
| table rows | 21% | 66% | 10% | 3% |
| headings | 83% | 10% | 1% | 5% |

It ran at `EDITED = 0.5`, before the threshold was set. The full-run figures replace it.

**Full run (2026-09-19):** 26,357 S&P 500 releases, 11.1M units, 0 failures. The
24,061 mature releases (4+ prior) break down as:

| Unit | boilerplate | template | edited | new | count |
|---|---|---|---|---|---|
| sentences | 44% | 15% | 19% | 23% | 3,156,807 |
| table rows | 25% | 65% | 8% | 3% | 5,580,594 |

**New plus edited sentences come to 1,301,287.** That is the layer 1 reading load for
full coverage, and about 54 per release.
- A company's first releases have little or no history, so every unit looks new.
  `n_prior` is stored, and later layers should require `n_prior ≥ PRIOR_K`.

## First end-to-end prototype (2026-09-19, exploratory, development data only)

`research/v3_prototype.py`. The reader is a cheap stand-in: Loughran-McDonald tone plus
guidance words. The layer-0 output covered about a third of companies (150), which gives
9,155 out-of-sample S&P 500 events, predicted walk-forward from 2013. This is not
evidence; it is a check for signs of life.

- **Idea 1 works at a basic level.** Out-of-sample R² for the reaction: surprise alone
  0.084; plus tone on all sentences 0.092 (gain p < 0.001, by year); plus tone on
  changed sentences 0.091 (p = 0.002); plus tone on **boilerplate only, the placebo,
  0.086 (p = 0.22)**. The placebo behaves, so text repeated verbatim carries nothing.
  With word counts, "changed only" is no better than "all", because repeated sentences
  only add a per-company constant. Layer 0 should matter for a real reader.
- **Idea 2 shows no sign of life yet.** The gap's top-minus-bottom quintile drift is
  +0.56pp (p = 0.34, quarter-clustered). The text's own part of the gap gives −0.15pp
  (p = 0.78). So far the market appears to price the release text on the day.

**Re-run on the full layer 0 (2026-09-19):** 21,493 out-of-sample events from 2013.

| Reaction model | R² | gain vs surprise, by year |
|---|---|---|
| surprise only | 0.0711 | — |
| + tone/guidance, all sentences | 0.0775 | p < 0.001 |
| + tone/guidance, **changed sentences** | **0.0782** | p < 0.001 |
| + placebo: boilerplate only | 0.0714 | p = 0.71 |

- **Idea 1 holds**, and with the full data "changed only" edges out "all": layer 0
  helps even a word-list reader.
- **Idea 2 still shows nothing.** The gap's quintile spread is +0.57pp (p = 0.19). The
  text's own part of the gap is **−0.72pp (p = 0.14)**, which is the wrong sign, if
  anything hinting that the market over-reacts to release text on the day.
- **Sample size for comparing readers:** the word-list gain that is clear at 21,493
  events is invisible at 2,000 (reader comparison, first run: R² 0.0629 surprise only,
  0.0627 with word-list tone, 0.0619 with the MiniLM map). Scaling the t-statistic by
  √n, about 8,000 releases are needed to detect an effect this size.

**Owner's steer (2026-09-19): develop on samples** (about 2,000 releases), not the full
data. That makes layer 1 affordable: FinBERT on 2,000 releases' changed sentences takes
about 1.5 hours instead of about 20. The final exam still uses the full exam set: at
about ±15% per-event noise, 2,000 events give roughly ±1pp uncertainty on a quintile
spread, which is too coarse for effects around 0.5pp.

## Testing plan (to be pre-registered before layer 3 is fitted)

- **Development:** S&P 500 releases. v2 spent the 2020-2026 S&P 500 holdout on its own
  question (does v2's score predict drift). Using those years' *announcement reactions*
  to develop a different model is a new question, not a second look at that one.
- **Final exam, once:** S&P 400/600 releases. No v3 layer that touches returns may run on
  them before the pre-registration is committed. Layer 0 can, since it reads no returns.

## Owner's idea: model disagreement and model bias as signals (2026-09-19)

Each reader has its own habits. If FinBERT is very positive where the others are not,
and FinBERT is known to spike on certain kinds of sentence, that pattern may carry
information. v1 averaged the models, which destroys exactly this. On v1's news, the
three models agreed on a label only 53% of the time (correlation 0.70), so about a third
of each model's variation is its own.

Planned inputs for layer 3. None of these are hand-written rules; the weights are learned
from the market reaction:

1. **Disagreement:** each model's score relative to the others (signed difference and
   spread) for the release's changed sentences.
2. **Bias-corrected score:** each model's score minus what that model usually says about
   this kind of sentence, learned on training years. This is "own norm" applied to the
   reader instead of the company.
3. **Ambiguity hypothesis (idea 2):** releases where the readers disagree are harder to
   read, may be digested more slowly, and so may show larger gaps and drift. This is to be
   tested, not assumed.

**Guard:** there are many possible disagreement and bias features, and some will fit
development data by chance. The set is kept small and is fixed in the pre-registration
before the exam.

## Open

- ~~An LLM "teacher" for layer 1~~ **Ruled out by the owner (2026-09-19): the project
  is free and stays free unless the owner says otherwise.** Layer 1 uses free models run
  locally (FinBERT, which is already cached, or a small free Hugging Face model), trained
  on samples.
- The S&P 400/600 releases still need downloading (the machinery exists).
