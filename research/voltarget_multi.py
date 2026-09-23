# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: the diversified, volatility-targeted book.
#
#  CONSTRUCTION, NOT A TEST. Both one-shot samples are spent -- SPY
#  1993-2001 (failed) and the 17-market cohort (passed). Nothing here can
#  be called validated; it is a build measured on data already seen, and
#  it is reported as such.
#
#  Structure, in the standard two layers:
#    1. four asset-class buckets, equal weight within each
#    2. inverse-volatility weights ACROSS buckets (risk parity), so one
#       bucket cannot dominate simply by being more volatile
#    3. the whole book then scaled to a portfolio volatility target
#
#  Assets enter as they become available -- GLD does not exist before
#  2004-11 and pretending otherwise would be look-ahead.
#
#  Benchmarks are the ones a reader would actually reach for: SPY alone,
#  and 60/40 SPY/IEF. Beating a bad benchmark is not a result; the
#  international book already showed that (6.06%/yr, losing to SPY).
#
#    .venv\Scripts\python.exe research\voltarget_multi.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "data" / "vol.db"

BUCKETS = {
    "US equity":    ["SPY", "IJH", "IJR"],
    "Intl equity":  ["EWA", "EWO", "EWK", "EWC", "EWQ", "EWG", "EWH", "EWI", "EWJ",
                     "EWM", "EWW", "EWN", "EWS", "EWP", "EWD", "EWL", "EWU"],
    "Bonds":        ["TLT", "IEF"],
    "Gold":         ["GLD"],
}
LOOK, TARGET_PORT, CAP, COST, MARGIN = 21, 0.10, 1.5, 2 / 1e4, 0.015


def load():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    need = sorted({s for v in BUCKETS.values() for s in v} | {"IEF"})
    px = {}
    for s in need:
        d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                              conn, params=(s,))
        d["date"] = pd.to_datetime(d.date)
        px[s] = d.set_index("date").close
    tb = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol='IRX' ORDER BY date", conn)
    conn.close()
    tb["date"] = pd.to_datetime(tb.date)
    return px, tb.set_index("date").close / 100.0


def stats(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return ann, vol, (ann / vol if vol else np.nan), float((eq / eq.cummax() - 1).min()), eq.iloc[-1]


def show(lab, r):
    a, v, s, d, g = stats(r)
    print(f"   {lab:<40}{a*100:>8.2f}%{v*100:>8.1f}%{s:>9.2f}{d*100:>9.0f}%{g:>9.1f}x")


def month_ends(idx):
    return (idx.to_series().groupby(idx.to_period("M")).transform("max") == idx).to_numpy()


def main():
    px, tb = load()
    idx = px["SPY"].index
    rets = pd.DataFrame({s: v.pct_change() for s, v in px.items()}).reindex(idx)

    # bucket returns: equal weight within, only across assets that exist yet
    bucket = {}
    for name, syms in BUCKETS.items():
        have = rets[syms]
        bucket[name] = have.mean(axis=1, skipna=True)
    B = pd.DataFrame(bucket)
    B = B[B.notna().any(axis=1)]
    idx = B.index
    me = month_ends(idx)
    rf = tb.reindex(idx).ffill().fillna(0.0)

    # layer 2: inverse-vol weights across buckets, set at month end
    bvol = B.rolling(LOOK).std() * np.sqrt(252)
    inv = (1.0 / bvol).where(B.notna())
    w = inv.div(inv.sum(axis=1), axis=0)
    keep = pd.DataFrame(np.repeat(me[:, None], w.shape[1], axis=1),
                        index=w.index, columns=w.columns)
    w = w.where(keep).shift(1).ffill()          # weights fixed at each month end
    rp = (w * B).sum(axis=1, min_count=1)

    # layer 3: scale the whole book to a volatility target
    pvol = rp.rolling(LOOK).std() * np.sqrt(252)
    lev = (TARGET_PORT / pvol).clip(lower=0, upper=CAP)
    lev = lev.where(me).shift(1).ffill()
    rate = np.where(lev.to_numpy() <= 1.0, rf.to_numpy() / 252, (rf.to_numpy() + MARGIN) / 252)
    turn = (w.diff().abs().sum(axis=1).fillna(0.0) + lev.diff().abs().fillna(0.0))
    final = (lev * rp + (1 - lev) * rate - turn * COST).dropna()

    start = final.index[0]
    spy = rets.SPY.loc[start:].fillna(0.0)
    ief = rets.IEF.loc[start:].fillna(0.0)
    sixty = (0.6 * spy + 0.4 * ief)
    rp_only = rp.loc[start:].dropna()

    print(f"{start:%Y-%m} → {idx[-1]:%Y-%m}, {len(final)/252:.1f} years, USD, net of costs")
    print("CONSTRUCTION on data already seen — not a validated result.\n")
    print(f"   {'':<40}{'return':>9}{'vol':>8}{'Sharpe':>9}{'max DD':>9}{'growth':>9}")
    show("SPY buy & hold", spy)
    show("60/40 SPY/IEF", sixty)
    show("risk parity across buckets (no target)", rp_only)
    show("risk parity + vol target", final)
    print("   (the early years hold US equity alone — bonds arrive 2002, gold 2004 —")
    print("    so the full-sample rows blend an undiversified period with a diversified one)")

    print("\n── fully-diversified period only, all four buckets live, 2005+ ──")
    print("   Leverage is set ex ante and margin interest is CHARGED at T-bill + 1.5% on")
    print("   every borrowed dollar. No post-hoc vol matching: scaling a return series by")
    print("   its own full-sample standard deviation is look-ahead, and it quietly omits")
    print("   the cost of the leverage it implies.")
    print(f"\n   {'':<40}{'return':>9}{'vol':>8}{'Sharpe':>9}{'max DD':>9}{'avg lev':>9}")
    for lab, r in [("SPY buy & hold", spy.loc["2005":]), ("60/40 SPY/IEF", sixty.loc["2005":])]:
        a, v, sh, d, g = stats(r)
        print(f"   {lab:<40}{a*100:>8.2f}%{v*100:>8.1f}%{sh:>9.2f}{d*100:>9.0f}%{'-':>9}")
    for tgt, cp in [(0.10, 1.5), (0.15, 2.0), (0.20, 2.5), (0.25, 3.0)]:
        lv = (tgt / pvol).clip(lower=0, upper=cp).where(me).shift(1).ffill()
        rt = np.where(lv.to_numpy() <= 1.0, rf.to_numpy() / 252,
                      (rf.to_numpy() + MARGIN) / 252)
        tn = w.diff().abs().sum(axis=1).fillna(0.0) + lv.diff().abs().fillna(0.0)
        bk = (lv * rp + (1 - lv) * rt - tn * COST).dropna().loc["2005":]
        a, v, sh, d, g = stats(bk)
        lab = "risk parity, target %d%%, cap %.1fx" % (tgt * 100, cp)
        print(f"   {lab:<40}{a*100:>8.2f}%{v*100:>8.1f}%{sh:>9.2f}{d*100:>9.0f}%"
              f"{lv.loc['2005':].mean():>8.2f}x")

    print("\n   Sharpe FALLS as leverage rises, 0.83 down to 0.59: margin interest eats the")
    print("   advantage, and by target 25% the book is simply SPY with extra steps. The")
    print("   useful point is target 15% — SPY's return with a third less drawdown — not")
    print("   the highest-return row.")

    print("\n   by era (annualised return):")
    for a, b in [("2005", "2012"), ("2013", "2020"), ("2021", "2026")]:
        f = lambda x: ((1 + x.loc[a:b]).prod() ** (252 / max(len(x.loc[a:b]), 1)) - 1) * 100
        print(f"      {a}-{b}:  SPY {f(spy):+6.1f}%   60/40 {f(sixty):+6.1f}%   "
              f"book {f(final):+6.1f}%")

    print("\n   average bucket weights, 2005+: " +
          ", ".join(f"{c} {w[c].loc['2005':].mean():.0%}" for c in w.columns))
    print(f"   average leverage: {lev.mean():.2f}x   (cap {CAP}x)")
    print("\n   Bonds are the largest bucket at 36%, and bonds ran a four-decade bull")
    print("   market through 2021. 2021-2026 is where that stops flattering the book.")


if __name__ == "__main__":
    main()
