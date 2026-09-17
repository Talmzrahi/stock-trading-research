# ═══════════════════════════════════════════════════════════════════════
#  The pre-registered test of the v2 sentiment signal.
#
#  Rules: research/design_sentiment_v2.md. Nothing here may be tuned after
#  seeing holdout output.
#
#  The question is NOT "does text predict returns" — the earnings surprise
#  already does that. It is "does the company's own framing of the quarter
#  predict what the surprise does not". So:
#
#    1. a baseline model of forward return from the surprise alone, fitted
#       on 2010-2019
#    2. its residual — the part the numbers do not explain
#    3. a text model of that residual, also fitted on 2010-2019
#    4. one look at 2020-2026: does the text score still predict?
#
#  THE HOLDOUT IS PROTECTED STRUCTURALLY. --dev cannot read it at all;
#  --final reads it once and says so loudly. Develop in --dev.
#
#    python research/sentiment_gate.py --dev      # train period only
#    python research/sentiment_gate.py --final    # the single look
# ═══════════════════════════════════════════════════════════════════════

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402

EDGAR_DB   = ROOT / "data" / "edgar.db"
TRAIN_END  = "2019-12-31"
HOLD_START = "2020-01-01"
HOLD       = 60          # sessions, matching the strategy's holding period

# Registered feature set. Adding to this needs the design updated first.
TEXT_FEATURES = ["tone_z", "guide_dir", "guide_share", "guide_numeric"]
SURPRISE_FEATURES = ["sue", "conviction"]


def build_panel(cfg):
    """Every point-in-time earnings event that has a filing, with its
    forward return, its surprise, and its text features."""
    market = load_market(cfg)
    ev = market.events
    ev = ev[ev.pit & ev.conviction.notna() & (ev.entry_idx + HOLD < len(market.cal))].copy()

    closes = market.closes.ffill()
    col = {s: i for i, s in enumerate(closes.columns)}
    px = closes.to_numpy()
    e = ev.entry_idx.to_numpy()
    x = e + HOLD
    c = ev.symbol.map(col).to_numpy()
    b = col[cfg.benchmark]
    ev["fwd"] = (px[x, c] / px[e, c] - 1) - (px[x, b] / px[e, b] - 1)
    ev = ev[np.isfinite(ev.fwd)]

    conn = sqlite3.connect(f"file:{EDGAR_DB}?mode=ro", uri=True, timeout=60)
    links = pd.read_sql_query(
        "SELECT event_key, accession FROM event_filings WHERE accession IS NOT NULL", conn)
    feats = pd.read_sql_query("SELECT * FROM features", conn)
    conn.close()

    panel = (ev.merge(links, left_on="key", right_on="event_key")
               .merge(feats.drop(columns=["symbol"], errors="ignore"), on="accession",
                      suffixes=("", "_f")))
    panel = panel.dropna(subset=TEXT_FEATURES + SURPRISE_FEATURES + ["fwd"])
    panel["period"] = np.where(panel.entry_date <= TRAIN_END, "train", "holdout")
    return panel


def fit_ridge(X, y, alpha=1.0):
    """Ridge by hand — one dependency fewer, and the maths is three lines.
    Features are standardised so the penalty treats them alike."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    A[-1, -1] = 0.0                      # never penalise the intercept
    beta = np.linalg.solve(A, Z.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta}


def predict(model, X):
    Z = np.column_stack([(X - model["mu"]) / model["sd"], np.ones(len(X))])
    return Z @ model["beta"]


def clustered(values, dates):
    """Mean and p-value with one observation per date, as every other gate."""
    m = pd.Series(values).groupby(pd.Series(dates).values).mean()
    if len(m) < 5:
        return float("nan"), float("nan"), len(m)
    return float(m.mean()), float(stats.ttest_1samp(m, 0).pvalue), len(m)


def report_split(panel, model_base, model_text, label):
    X_s = panel[SURPRISE_FEATURES].to_numpy(float)
    X_t = panel[TEXT_FEATURES].to_numpy(float)
    resid = panel.fwd.to_numpy() - predict(model_base, X_s)
    score = predict(model_text, X_t)

    r = np.corrcoef(score, resid)[0, 1]
    bands = pd.qcut(score, 5, labels=["lowest", "2", "3", "4", "highest"], duplicates="drop")
    print(f"\n── {label}: {len(panel):,} events "
          f"({panel.entry_date.min().date()} → {panel.entry_date.max().date()}) ──")
    print(f"   correlation between text score and unexplained return: {r:+.4f}")
    print(f"   {'text score':<10}{'n':>8}{'mean unexplained return':>26}{'p':>9}")
    for band, idx in panel.groupby(bands, observed=True).groups.items():
        rows = panel.index.get_indexer(idx)
        m, p, _ = clustered(resid[rows], panel.entry_date.iloc[rows])
        print(f"   {str(band):<10}{len(rows):>8,}{m * 100:>24.3f}pp{p:>9.3f}")

    top, bottom = bands == "highest", bands == "lowest"
    diff = resid[top].mean() - resid[bottom].mean()
    _, p_diff = stats.ttest_ind(resid[top], resid[bottom], equal_var=False)
    print(f"   highest minus lowest: {diff * 100:+.3f}pp (p={p_diff:.4f})")
    return r, diff, p_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="spend the single look at the 2020-2026 holdout")
    ap.add_argument("--dev", action="store_true", help="train period only (default)")
    args = ap.parse_args()
    final = args.final and not args.dev

    cfg = load_config()
    print("Building the event panel …", flush=True)
    panel = build_panel(cfg)
    train = panel[panel.period == "train"].reset_index(drop=True)
    n_hold = int((panel.period == "holdout").sum())
    print(f"{len(panel):,} events with filings — {len(train):,} train, {n_hold:,} holdout "
          f"({'holdout NOT read' if not final else 'holdout being read now'})")

    if len(train) < 500:
        raise SystemExit("Too few training events — is the download finished?")

    base = fit_ridge(train[SURPRISE_FEATURES].to_numpy(float), train.fwd.to_numpy())
    resid_train = train.fwd.to_numpy() - predict(base, train[SURPRISE_FEATURES].to_numpy(float))
    text = fit_ridge(train[TEXT_FEATURES].to_numpy(float), resid_train)
    print("\nText coefficients (standardised, on the unexplained return):")
    for name, b in zip(TEXT_FEATURES, text["beta"][:-1]):
        print(f"   {name:<16}{b * 100:+.3f}pp per sd")

    report_split(train, base, text, "TRAIN (fitted here — optimistic by construction)")

    if not final:
        print("\nHoldout untouched. Re-run with --final to spend the single look.")
        return

    hold = panel[panel.period == "holdout"].reset_index(drop=True)
    print("\n" + "=" * 62)
    print("SPENDING THE SINGLE LOOK AT THE HOLDOUT — this is not repeatable")
    print("=" * 62)
    r, diff, p_diff = report_split(hold, base, text, "HOLDOUT 2020-2026 (never fitted)")

    # Pre-registered secondary: does it sort the trades the strategy takes?
    top5 = hold[hold.conviction >= cfg.cutoff]
    if len(top5) >= 100:
        X_s = top5[SURPRISE_FEATURES].to_numpy(float)
        resid = top5.fwd.to_numpy() - predict(base, X_s)
        score = predict(text, top5[TEXT_FEATURES].to_numpy(float))
        half = score >= np.median(score)
        hi, _, _ = clustered(resid[half], top5.entry_date[half])
        lo, _, _ = clustered(resid[~half], top5.entry_date[~half])
        _, p_half = stats.ttest_ind(resid[half], resid[~half], equal_var=False)
        print(f"\n── Among the top-{round((1 - cfg.cutoff) * 100)}% events the strategy "
              f"actually trades ({len(top5):,}) ──")
        print(f"   better-framed half {hi * 100:+.3f}pp vs worse half {lo * 100:+.3f}pp "
              f"(p={p_half:.4f})")
        sorts = p_half < 0.05 and hi > lo
    else:
        print(f"\n   Only {len(top5)} top-cutoff events in the holdout — too few to sort.")
        sorts = False

    verdict = "PASS" if (diff > 0 and p_diff < 0.05 and sorts) else "FAIL"
    print(f"\n══ PRE-REGISTERED VERDICT: {verdict} ══")
    print(f"   text score spread {diff * 100:+.3f}pp (p={p_diff:.4f}), "
          f"sorts the traded events: {sorts}")
    if verdict == "FAIL":
        print("   The fusion thesis is reported as unsupported. It is not retuned.")


if __name__ == "__main__":
    main()
