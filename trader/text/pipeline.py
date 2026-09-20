# ═══════════════════════════════════════════════════════════════════════
#  Reading today's releases, end to end.
#
#  For each earnings event the daily run is about to consider:
#
#    find the 8-K       the company's recent filings, Item 2.02, within
#                       two days of the announcement
#    check it is public the filing's acceptance time must be in the past;
#                       a release we could not have read is not scored
#    fetch and cache    into data/edgar.db, where it becomes this
#                       company's history next quarter
#    what is new        against its previous four releases (layer 0)
#    read               the first 20 new or edited sentences
#    score              features -> the fitted model -> predicted reaction
#
#  Every event gets a row, including the ones that could not be scored and
#  why, so gaps are visible instead of silently missing.
# ═══════════════════════════════════════════════════════════════════════

import math

import pandas as pd

from .fetch import find_release, release_html
from .novelty import PRIOR_K, changed_sentences, classify
from .parse import blocks
from .readers import CAP
from .store import cache_release, history, html_of, record

SKIPPED = ("no_cik", "no_filing", "not_public", "fetch_failed", "short_history",
           "no_new_text", "no_surprise")


def score_event(event, *, edgar, model, readers, cik, now):
    """(status, row-without-keys) for one event. Network and model work."""
    ann = pd.Timestamp(event["ann_date"]).strftime("%Y-%m-%d")
    meta = find_release(cik, ann)
    if meta is None:
        return "no_filing", {}
    accepted = pd.to_datetime(meta["accepted_at"], utc=True, errors="coerce")
    if accepted is not pd.NaT and accepted > pd.Timestamp(now).tz_convert("UTC"):
        return "not_public", {"accession": meta["accession"]}

    html = html_of(edgar, meta["accession"])
    if html is None:
        got = release_html(cik, meta["accession"])
        if got is None:
            return "fetch_failed", {"accession": meta["accession"]}
        doc_type, filename, html = got
        cache_release(edgar, cik, event["symbol"], meta, doc_type, filename, html)

    past = history(edgar, cik, meta["filed_date"], PRIOR_K)
    if len(past) < PRIOR_K:
        return "short_history", {"accession": meta["accession"]}

    sentences = changed_sentences(classify(blocks(html), past), CAP)
    if not sentences:
        return "no_new_text", {"accession": meta["accession"]}

    conviction = event.get("conviction")
    sue = event.get("sue")
    if conviction is None or sue is None or math.isnan(conviction) or math.isnan(sue):
        return "no_surprise", {"accession": meta["accession"], "n_sentences": len(sentences)}

    features = readers.score(sentences)
    from .features import release_features            # local: keeps import cost off the fast path
    row = release_features(features, model.maps)
    row["sue_w"] = model.winsorise_sue(sue)
    row["conviction"] = float(conviction)
    prediction = float(model.predict(model.row(row))[0])
    if math.isnan(prediction):
        return "missing_features", {"accession": meta["accession"], "n_sentences": len(sentences),
                                    "features": row}
    return "scored", {"accession": meta["accession"], "n_sentences": len(sentences),
                      "prediction": prediction, "percentile": float(model.percentile(prediction)),
                      "features": row}


def score_pending(events, *, edgar, scores, model, readers, cik_of, now, log=print):
    """Score every event that has no score yet. One row written per event."""
    done = {k for (k,) in scores.execute("SELECT event_key FROM text_scores")}
    todo = [e for _, e in events.iterrows() if e["key"] not in done]
    if not todo:
        return []
    log(f"Text signal: reading {len(todo)} release(s) …")
    written = []
    for event in todo:
        cik = cik_of.get(event["symbol"])
        if not cik:
            status, extra = "no_cik", {}
        else:
            try:
                status, extra = score_event(event, edgar=edgar, model=model, readers=readers,
                                            cik=cik, now=now)
            except Exception as e:                      # one bad filing must not stop the run
                status, extra = f"error:{type(e).__name__}", {}
                log(f"   {event['symbol']}: {type(e).__name__} {str(e)[:80]}")
        row = {"event_key": event["key"], "scored_on": pd.Timestamp(now).strftime("%Y-%m-%d"),
               "symbol": event["symbol"], "cik": cik,
               "accession": extra.get("accession"),
               "ann_date": pd.Timestamp(event["ann_date"]).strftime("%Y-%m-%d"),
               "entry_date": pd.Timestamp(event["entry_date"]).strftime("%Y-%m-%d"),
               "status": status, "n_sentences": extra.get("n_sentences"),
               "prediction": extra.get("prediction"), "percentile": extra.get("percentile"),
               "features": extra.get("features")}
        record(scores, row)
        written.append(row)
        if status == "scored":
            p = extra["percentile"]
            where = f"top {(1 - p) * 100:.0f}%" if p >= 0.5 else f"bottom {p * 100:.0f}%"
            log(f"   {event['symbol']}: {extra['n_sentences']} new sentences, "
                f"predicted reaction {extra['prediction'] * 100:+.2f}pp ({where} of training)")
        else:
            log(f"   {event['symbol']}: {status}")
    return written
