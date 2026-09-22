# ═══════════════════════════════════════════════════════════════════════
#  Can a score rank the PEAD picks — and does it survive honest timing?
#
#  Everything in v3 predicts the announcement REACTION. This fits the
#  same features on the 60-session DRIFT instead, then asks whether the
#  resulting score can sort the trades the live system already takes.
#
#  The point of this file is the ladder of corrections. Each row removes
#  one source of hindsight, and each removal costs about half the effect:
#
#    1  in-sample quintiles + cross-fitted model   (both see the future)
#    2  trailing mean/sd threshold, same model     (threshold is honest)
#    3  expanding-window model + trailing threshold (nothing sees ahead)
#
#  Result (2026-09-22, development data): row 1 shows T1 +4.67pp p=0.047
#  and T2 +9.54%/yr; row 3 shows the best cut at T1 +3.37pp p=0.062 and
#  T2 +6.05%/yr p=0.276. The rule does NOT clear the gate. It is not
#  flat either — it is unresolved at ~25 trades/yr, which is the power
#  problem, not a verdict.
#
#  DEVELOPMENT DATA ONLY, and looked at many times. Nothing here is
#  evidence; the S&P 400/600 exam is the only test that counts, and on
#  these numbers it is not worth spending.
#
#    .venv\Scripts\python.exe research\v3_drift_rule.py
# ═══════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402
from inference_check import (calendar_time, daily_returns,  # noqa: E402
                             market_alpha, quarter_clustered)
from v3_readers_eval import N_FOLDS, out_of_fold, ridge  # noqa: E402
from v3_signal_probe import BASE, load  # noqa: E402

HOLD    = 60
COST    = 10 / 1e4
CUTOFF  = 0.95          # the live system's SUE percentile
WINDOW  = 1095          # 3 years, for trailing mean/sd
MIN_OBS = 30            # below this a trailing mean/sd is too noisy
KS      = [0.0, 0.5, 1.0, 1.5]


def trailing_stats(dates, values, window=WINDOW, min_obs=MIN_OBS):
    """Mean and sd of every value dated in [t - window, t), so same-day
    events never inform their own threshold and nothing leaks back."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    v = np.asarray(values, dtype=float)
    order = np.argsort(d, kind="stable")
    d_s, v_s = d[order], v[order]
    starts = np.searchsorted(d_s, d_s - np.timedelta64(window, "D"), side="left")
    _, first = np.unique(d_s, return_index=True)
    bounds = np.append(first, len(d_s))

    mean_s = np.full(len(v_s), np.nan)
    sd_s = np.full(len(v_s), np.nan)
    for f, last in zip(bounds[:-1], bounds[1:]):
        hist = v_s[starts[f]:f]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_obs:
            continue
        mean_s[f:last], sd_s[f:last] = hist.mean(), hist.std(ddof=1)

    mean, sd = np.empty_like(mean_s), np.empty_like(sd_s)
    mean[order], sd[order] = mean_s, sd_s
    return mean, sd


def build(cfg):
    """Events with both scores: cross-fitted (sees the future) and
    expanding-window (does not)."""
    d = load(cfg)
    text = [c for c in d.columns if any(c.startswith(p) for p in
            ("distilroberta_fin", "finbert", "minilm", "spread", "lm_tone"))]
    feats = BASE + text
    d = d[d.fwd60.notna()].copy()
    d["entry_date"] = pd.to_datetime(d.gap_entry_date)
    d["year"] = d.entry_date.dt.year
    d = d.sort_values("entry_date").reset_index(drop=True)

    blocks = np.array_split(sorted(d.year.unique()), N_FOLDS)
    d["fold"] = d.year.map({y: i for i, b in enumerate(blocks) for y in b})
    X, y = d[feats].to_numpy(float), d.fwd60.to_numpy()
    d["cross"] = out_of_fold(d, X, y)

    # Expanding window: year Y is scored by a fit on years < Y-1, because
    # last year's 60-session outcomes are not settled when Y starts.
    d["expand"] = np.nan
    for Y in sorted(d.year.unique()):
        tr, te = (d.year < Y - 1).to_numpy(), (d.year == Y).to_numpy()
        if tr.sum() < 2000:
            continue
        d.loc[te, "expand"] = ridge(X[tr], y[tr], 10.0)(X[te])
    return d, feats


def picks_with_returns(d, cfg, market):
    closes = market.closes.ffill()
    col = {s: i for i, s in enumerate(closes.columns)}
    px, n, b = closes.to_numpy(), len(market.cal), col[cfg.benchmark]
    p = d[(d.conviction >= CUTOFF) & (d.gap_entry_idx + HOLD < n)].copy()
    g, c = p.gap_entry_idx.to_numpy(), p.symbol.map(col).to_numpy()
    p["excess"] = (px[g + HOLD, c] / px[g, c] - 1) - (px[g + HOLD, b] / px[g, b] - 1)
    return p[np.isfinite(p.excess)].sort_values("entry_date").reset_index(drop=True), col, b


def gate(s, label, years, closes, col, b, rets):
    if len(s) < 40:
        print(f"   {label:<34}{len(s):>6}   too few")
        return
    means = (s.excess - COST).groupby(s.entry_date).mean()
    m1, p1, _ = quarter_clustered(means)
    monthly = calendar_time(s.gap_entry_idx.to_numpy() - 1, s.symbol.map(col).to_numpy(),
                            np.full(len(s), b), rets, closes.index, COST)
    a, p2, beta, _ = market_alpha(monthly)
    flag = " CLEARS" if (m1 > 0 and p1 < 0.05 and a > 0 and p2 < 0.05) else ""
    print(f"   {label:<34}{len(s):>6}{len(s)/years:>6.0f}{m1*100:>+9.2f}pp{p1:>7.3f}"
          f"{a*100:>+9.2f}%{p2:>7.3f}{beta:>6.2f}{flag}")


def main():
    cfg = load_config()
    d, feats = build(cfg)
    market = load_market(cfg)
    closes = market.closes.ffill()
    p, col, b = picks_with_returns(d, cfg, market)
    rets = daily_returns(closes)
    years = p.year.nunique()
    print(f"{len(p):,} PEAD picks with a 60-session outcome, {p.entry_date.min():%Y-%m} → "
          f"{p.entry_date.max():%Y-%m}\n")
    head = (f"   {'rule':<34}{'n':>6}{'/yr':>6}{'T1':>11}{'p':>7}"
            f"{'T2 alpha':>9}{'p':>7}{'beta':>6}")

    print("── 1. Both the model and the threshold see the future ──")
    print(head)
    mu, sd = p.cross.mean(), p.cross.std()
    gate(p, "every pick (what runs today)", years, closes, col, b, rets)
    for k in KS:
        gate(p[p.cross >= mu + k * sd], f"full-sample mean + {k}·sd", years,
             closes, col, b, rets)

    print("\n── 2. Honest threshold, model still sees the future ──")
    print(head)
    m, s = trailing_stats(p.entry_date, p.cross)
    u = p[~np.isnan(m)]
    mm, ss = m[~np.isnan(m)], s[~np.isnan(m)]
    for k in KS:
        gate(u[u.cross >= mm + k * ss], f"trailing mean + {k}·sd", u.year.nunique(),
             closes, col, b, rets)

    print("\n── 3. Nothing sees the future ──")
    print(head)
    e = p[p.expand.notna()].copy()
    m, s = trailing_stats(e.entry_date, e.expand)
    u = e[~np.isnan(m)]
    mm, ss = m[~np.isnan(m)], s[~np.isnan(m)]
    yrs = u.year.nunique()
    gate(u, "every pick", yrs, closes, col, b, rets)
    for k in KS:
        gate(u[u.expand >= mm + k * ss], f"trailing mean + {k}·sd", yrs,
             closes, col, b, rets)

    print("\n   Each block removes one source of hindsight. Compare the T2 column:")
    print("   the market-adjusted alpha is where the hindsight was living.")


if __name__ == "__main__":
    main()
