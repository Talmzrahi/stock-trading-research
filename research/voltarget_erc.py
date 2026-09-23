# =====================================================================
#  Correlation-aware weighting for the diversified book.
#
#  CONSTRUCTION on data already seen. Both one-shot samples are spent.
#
#  Inverse-volatility weighting ignores correlation. US and international
#  equity run at ~0.85 correlation, so the 21% + 21% the current book
#  holds in them contributes far more than 42% of portfolio risk. Equal
#  risk contribution (ERC) solves for weights where each bucket supplies
#  the same share of variance, which inverse-vol cannot express.
#
#  Fixed point: w <- sqrt(w / (Sigma w)), renormalised. At convergence
#  w_i * (Sigma w)_i is equal across buckets, which is the ERC condition.
#
#  Covariance uses a trailing year -- correlation needs more data than
#  volatility -- and weights are set at month end and applied from the
#  next trading day, so nothing is informed by the return it earns.
#
#  RESULT (2026-09-23): correlation-aware weighting makes the book WORSE,
#  and the mechanism is estimation noise rather than a bug. At a 10%
#  target: inverse-vol 8.42%/yr, excess Sharpe 0.62, drawdown -23%; ERC
#  6.55%/yr, 0.36, -41%; minimum variance 6.75%/yr, 0.49, -26%.
#
#  Three diagnostics say it is the method:
#    - weight turnover: ERC mean 0.084 and max 1.392 (a whole portfolio
#      flipping in a month) against inverse-vol's 0.025 and 0.209
#    - the covariance is NOT ill-conditioned: condition number median 11,
#      max 40, so this is not numerical
#    - shrinking the covariance toward its diagonal rescues ERC by
#      turning it back INTO inverse-vol: shrinkage 0.0 -> excess Sharpe
#      0.36, 0.3 -> 0.49, 0.9 -> 0.62, identical to inverse-vol
#
#  So the correlation information is actively harmful: correlations are
#  estimated far more noisily than volatilities, and an optimiser
#  amplifies that error. Ignoring the off-diagonal is the robust choice,
#  not a simplification to be apologised for.
#
#  My prior was the opposite -- US and international equity correlate
#  0.86, so I expected ERC to CUT combined equity weight. It raised it,
#  from 42% to 62%. Recorded because the prediction was wrong.
#
#    .venv\Scripts\python.exe research\voltarget_erc.py
# =====================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from voltarget_multi import BUCKETS, LOOK, COST, MARGIN, load, month_ends  # noqa: E402

COV_WIN = 252


def erc_weights(cov, iters=500, tol=1e-10):
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        m = cov @ w
        m = np.where(m <= 0, 1e-12, m)
        nw = np.sqrt(w / m)
        nw = nw / nw.sum()
        if np.max(np.abs(nw - w)) < tol:
            return nw
        w = nw
    return w


def min_var_weights(cov):
    n = cov.shape[0]
    inv = np.linalg.pinv(cov + np.eye(n) * 1e-10)
    w = inv @ np.ones(n)
    w = np.clip(w, 0, None)                 # long only
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


def scheme_weights(B, me, how):
    """Month-end weights for one scheme, shifted so they apply next day."""
    out = pd.DataFrame(index=B.index, columns=B.columns, dtype=float)
    vals = B.to_numpy()
    for t in np.flatnonzero(me):
        if t < COV_WIN:
            continue
        win = vals[t - COV_WIN + 1:t + 1]
        live = ~np.isnan(win).any(axis=0)
        if live.sum() < 2:
            continue
        cov = np.cov(win[:, live], rowvar=False) * 252
        if how == "invvol":
            d = np.sqrt(np.diag(cov))
            w = (1 / d) / (1 / d).sum()
        elif how == "erc":
            w = erc_weights(cov)
        else:
            w = min_var_weights(cov)
        out.iloc[t, np.flatnonzero(live)] = w
    return out.shift(1).ffill()


def build(B, w, rf, me, target, cap):
    rp = (w * B).sum(axis=1, min_count=1)
    pv = rp.rolling(LOOK).std() * np.sqrt(252)
    lev = (target / pv).clip(lower=0, upper=cap).where(me).shift(1).ffill()
    rate = np.where(lev.to_numpy() <= 1.0, rf.to_numpy() / 252,
                    (rf.to_numpy() + MARGIN) / 252)
    turn = w.diff().abs().sum(axis=1).fillna(0.0) + lev.diff().abs().fillna(0.0)
    return (lev * rp + (1 - lev) * rate - turn * COST).dropna()


def report(r, rf_d, label):
    r = r.loc["2005":]
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    ex = r - rf_d.reindex(r.index).fillna(0.0)
    ann_ex = (1 + ex).prod() ** (252 / len(ex)) - 1
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    print(f"   {label:<38}{ann*100:>8.2f}%{vol*100:>7.1f}%{ann/vol:>8.2f}"
          f"{ann_ex/vol:>9.2f}{dd*100:>8.0f}%{eq.iloc[-1]:>8.1f}x")
    return ann, ann_ex / vol


def main():
    px, tb = load()
    idx = px["SPY"].index
    rets = pd.DataFrame({s: v.pct_change() for s, v in px.items()}).reindex(idx)
    B = pd.DataFrame({n: rets[s].mean(axis=1, skipna=True) for n, s in BUCKETS.items()})
    B = B[B.notna().any(axis=1)]
    me = month_ends(B.index)
    rf = tb.reindex(B.index).ffill().fillna(0.0)
    rf_d = rf / 252

    print("2005+, margin charged at T-bill+1.5%. CONSTRUCTION on seen data.\n")
    print(f"   {'':<38}{'return':>9}{'vol':>7}{'Sharpe':>8}{'excess':>9}{'maxDD':>8}{'growth':>8}")
    spy = rets.SPY.fillna(0.0)
    report(spy, rf_d, "SPY buy & hold")
    report(0.6 * spy + 0.4 * rets.IEF.fillna(0.0), rf_d, "60/40 SPY/IEF")

    schemes = {h: scheme_weights(B, me, h) for h in ("invvol", "erc", "minvar")}
    names = {"invvol": "inverse-vol (current)", "erc": "equal risk contribution",
             "minvar": "minimum variance"}
    for tgt, cap in [(0.10, 1.5), (0.15, 2.0), (0.20, 2.5)]:
        print(f"\n   ── volatility target {tgt:.0%}, cap {cap:g}x ──")
        for h, w in schemes.items():
            report(build(B, w, rf, me, tgt, cap), rf_d, names[h])

    print("\n   average bucket weights, 2005+:")
    for h, w in schemes.items():
        ws = w.loc["2005":].mean()
        print(f"      {names[h]:<26}" + "  ".join(f"{c} {ws[c]:.0%}" for c in w.columns))
    eq_corr = B[["US equity", "Intl equity"]].loc["2005":].corr().iloc[0, 1]
    print(f"\n   US vs international equity correlation, 2005+: {eq_corr:.2f}")


if __name__ == "__main__":
    main()
