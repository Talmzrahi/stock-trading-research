# ═══════════════════════════════════════════════════════════════════════
#  Fusion hypothesis test — do hard numbers and narrative disagreeing
#  predict drift?
#
#  H1: among earnings beats, those preceded by NEGATIVE news sentiment
#      (numbers and narrative disagree -> the market has not caught on)
#      drift MORE than those preceded by positive sentiment (narrative
#      already agrees -> the good news is priced in).
#      => predicts a NEGATIVE beat x sentiment interaction.
#
#  Direction is pre-registered here deliberately: with two plausible
#  competing stories, letting the sign be decided after seeing the data
#  is how a null gets rationalised into a finding.
#
#  The sample is capped at ~12 months by Finnhub's free news history, so
#  this is powered to detect roughly 0.5pp and real effects in this domain
#  run 0.1-0.3pp. Read the confidence intervals, not the stars: the useful
#  output is "how large an effect can we rule out", not significant/not.
#
#    python research/fusion_test.py
# ═══════════════════════════════════════════════════════════════════════

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT     = Path(__file__).resolve().parent.parent
DB_FILE  = ROOT / "data" / "research.db"
MIN_ARTICLES = 5     # events with thinner coverage give too noisy a mean
HORIZONS = [1, 3, 5, 10]

_spec = importlib.util.spec_from_file_location("bt", Path(__file__).resolve().parent / "backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def load_sentiment(conn):
    df = pd.read_sql_query(
        f"""SELECT symbol, event_ts,
                   AVG(final_score) AS sent,
                   COUNT(*)         AS n_articles
            FROM event_articles
            WHERE scored = 1
            GROUP BY symbol, event_ts
            HAVING COUNT(*) >= {MIN_ARTICLES}""",
        conn,
    )
    df["announced_at"] = pd.to_datetime(df.event_ts, utc=True, format="mixed")
    return df.drop(columns=["event_ts"])


def fit(df, h, date_fe):
    import statsmodels.formula.api as smf

    d = df.assign(beat_i=df.beat.astype(int), date_id=df.entry_date.astype("category").cat.codes)
    formula = f"ret{h} ~ beat_i * sent_z" + (" + C(date_id)" if date_fe else "")
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.date_id})
    key = "beat_i:sent_z"
    ci = m.conf_int().loc[key]
    return {"h": h, "coef": m.params[key], "p": m.pvalues[key],
            "lo": ci[0], "hi": ci[1], "n": int(m.nobs)}


def main():
    conn = sqlite3.connect(DB_FILE)
    sent = load_sentiment(conn)
    if sent.empty:
        raise SystemExit("no scored sentiment yet — run research/sentiment_events.py first")
    wide, vix, earn = bt.load(conn)
    conn.close()

    ev = bt.build_events(wide, vix, earn)
    df = ev.merge(sent, on=["symbol", "announced_at"], how="inner")
    if df.empty:
        raise SystemExit("no overlap between scored sentiment and usable return events")

    df["sent_z"] = (df.sent - df.sent.mean()) / df.sent.std()

    print(f"Events with sentiment + returns : {len(df):,}")
    print(f"  distinct symbols              : {df.symbol.nunique()}")
    print(f"  distinct dates                : {df.entry_date.nunique()}")
    print(f"  beats (>{bt.BEAT_PCT}%)                  : {int(df.beat.sum()):,}")
    print(f"  articles per event            : {df.n_articles.mean():.1f} avg "
          f"({df.n_articles.min()}-{df.n_articles.max()})")
    print(f"  sentiment mean/sd             : {df.sent.mean():+.3f} / {df.sent.std():.3f}")
    print()

    print("── Mean 5-day market-adjusted return by beat x sentiment tercile ──")
    df["s_grp"] = pd.qcut(df.sent, 3, labels=["negative", "mid", "positive"])
    tab = df.groupby(["beat", "s_grp"], observed=True)["ret5"].agg(["mean", "size"])
    for (b, g), r in tab.iterrows():
        print(f"   beat={str(b):<5s} sentiment={str(g):<9s} "
              f"mean={r['mean']*100:+6.3f}%  n={int(r['size']):,}")
    beats = df[df.beat]
    if beats.s_grp.nunique() >= 2:
        neg = beats.loc[beats.s_grp == "negative", "ret5"].mean()
        pos = beats.loc[beats.s_grp == "positive", "ret5"].mean()
        print(f"\n   H1 predicts negative-sentiment beats drift MORE.")
        print(f"   beats: negative={neg*100:+.3f}%  positive={pos*100:+.3f}%  "
              f"gap={(neg-pos)*100:+.3f}pp  ({'consistent' if neg > pos else 'OPPOSITE'} with H1)")

    for date_fe in (False, True):
        tag = "WITH date fixed effects (within-day comparison)" if date_fe else "pooled"
        print(f"\n── beat x sentiment interaction — {tag} ──")
        print("   H1 predicts a NEGATIVE coefficient.")
        print(f"   {'h':>3} {'coef(pp)':>9} {'95% CI (pp)':>20} {'p':>7} {'n':>7}")
        for h in HORIZONS:
            try:
                r = fit(df, h, date_fe)
            except Exception as e:
                print(f"   {h:>3} fit failed: {type(e).__name__} {str(e)[:60]}")
                continue
            print(f"   {r['h']:>3} {r['coef']*100:>+9.3f} "
                  f"{f'[{r['lo']*100:+.3f}, {r['hi']*100:+.3f}]':>20} "
                  f"{r['p']:>7.3f} {r['n']:>7,}")

    print("\n" + "═" * 63)
    r5 = fit(df, 5, True)
    half = (r5["hi"] - r5["lo"]) / 2
    print(f"5-day interaction: {r5['coef']*100:+.3f}pp, 95% CI "
          f"[{r5['lo']*100:+.3f}, {r5['hi']*100:+.3f}]pp")
    if r5["p"] < 0.05 and r5["coef"] < 0:
        print("Consistent with H1 and significant — treat as provisional given the")
        print("12-month window, and re-test on a longer history before building on it.")
    elif r5["p"] < 0.05:
        print("Significant but the WRONG SIGN for H1: positive-sentiment beats drift")
        print("more, which fits the 'narrative carries real information' story instead.")
    else:
        print("Not distinguishable from zero. The CI is what matters here: effects")
        print(f"larger than about {half*100:.2f}pp in either direction are ruled out,")
        print("but the 0.1-0.3pp range this domain actually lives in is NOT ruled out.")
        print("This is an inconclusive result, not evidence of no effect.")


if __name__ == "__main__":
    main()
