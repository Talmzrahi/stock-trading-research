# ═══════════════════════════════════════════════════════════════════════
#  International replication of the volatility-targeting test.
#
#  Rule, universe and verdict fixed in research/prereg_voltarget_intl.md,
#  committed first (e584e7f). This file only executes it.
#
#  The SPY holdout failed and is spent. All 17 iShares MSCI single-country
#  funds launched 1996-03-18 are tested, every one reported.
#
#    .venv\Scripts\python.exe research\voltarget_intl.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DB   = ROOT / "data" / "vol.db"
POSTURES = [("conservative", 0.10, 1.0), ("standard", 0.12, 1.5), ("aggressive", 0.15, 2.0)]
LOOKBACKS, TARGETS = [10, 21, 42, 63], [0.10, 0.12, 0.15]
BASE_LOOK, BASE_TARGET, BASE_CAP = 21, 0.12, 1.5
MARGIN_SPREAD = 0.015


def load():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cov = pd.read_sql_query(
        "SELECT symbol, note FROM coverage WHERE kind='country_etf' ORDER BY note", conn)
    px = {}
    for s in cov.symbol:
        d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                              conn, params=(s,))
        d["date"] = pd.to_datetime(d.date)
        px[s] = d.set_index("date").close
    tb = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol='IRX' ORDER BY date", conn)
    conn.close()
    tb["date"] = pd.to_datetime(tb.date)
    return cov, px, tb.set_index("date").close / 100.0


def run(ret, tb, target, cap, look, cost_bps=2.0):
    idx = ret.index
    fc = ret.rolling(look).std() * np.sqrt(252)
    raw = (target / fc).clip(lower=0, upper=cap)
    me = idx.to_series().groupby(idx.to_period("M")).transform("max") == idx
    w = raw.where(me.to_numpy()).shift(1).ffill()
    rf = tb.reindex(idx).ffill().fillna(0.0)
    rate = np.where(w.to_numpy() <= 1.0, rf.to_numpy() / 252,
                    (rf.to_numpy() + MARGIN_SPREAD) / 252)
    gross = w * ret + (1 - w) * rate
    return (gross - w.diff().abs().fillna(0.0) * cost_bps / 1e4).dropna()


def stats(r):
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return ann, vol, (ann / vol if vol else np.nan), float((eq / eq.cummax() - 1).min())


def main():
    cov, px, tb = load()
    print(f"{len(cov)} markets, {px[cov.symbol.iloc[0]].index[0]:%Y-%m-%d} → "
          f"{px[cov.symbol.iloc[0]].index[-1]:%Y-%m-%d}\n")

    rows = []
    for _, c in cov.iterrows():
        ret = px[c.symbol].pct_change().fillna(0.0)
        b_ret, b_vol, b_sh, b_dd = stats(ret)
        s_ret, s_vol, s_sh, s_dd = stats(run(ret, tb, BASE_TARGET, BASE_CAP, BASE_LOOK))
        s10 = stats(run(ret, tb, BASE_TARGET, BASE_CAP, BASE_LOOK, 10.0))[2]
        rows.append(dict(sym=c.symbol, name=c.note, b_sh=b_sh, s_sh=s_sh,
                         d_sh=s_sh - b_sh, b_dd=b_dd, s_dd=s_dd,
                         d_dd=(abs(b_dd) - abs(s_dd)) / abs(b_dd),
                         b_ret=b_ret, s_ret=s_ret, d10=s10 - b_sh))
    R = pd.DataFrame(rows)

    print(f"{'market':<14}{'buy&hold':>10}{'targeted':>10}{'ΔSharpe':>10}"
          f"{'b&h DD':>9}{'tgt DD':>9}{'ΔDD':>8}")
    for _, r in R.sort_values("d_sh", ascending=False).iterrows():
        print(f"{r['name']:<14}{r.b_sh:>10.2f}{r.s_sh:>10.2f}{r.d_sh:>+10.2f}"
              f"{r.b_dd*100:>8.0f}%{r.s_dd*100:>8.0f}%{r.d_dd*100:>7.0f}%")

    med_sh, n_up = R.d_sh.median(), int((R.d_sh > 0).sum())
    med_dd, med_10 = R.d_dd.median(), R.d10.median()
    print(f"\n{'median':<14}{R.b_sh.median():>10.2f}{R.s_sh.median():>10.2f}"
          f"{med_sh:>+10.2f}{R.b_dd.median()*100:>8.0f}%{R.s_dd.median()*100:>8.0f}%"
          f"{med_dd*100:>7.0f}%")

    print("\n── other postures (median across markets) ──")
    for name, tgt, cap in POSTURES:
        d = [stats(run(px[s].pct_change().fillna(0.0), tb, tgt, cap, BASE_LOOK))[2]
             - stats(px[s].pct_change().fillna(0.0))[2] for s in cov.symbol]
        print(f"   {name:<14}{tgt:.0%}/{cap:g}x   median ΔSharpe {np.median(d):+.2f}")

    print("\n── parameter grid: median ΔSharpe across all 17 markets ──")
    print(f"   {'lookback':>9}" + "".join(f"{t:>10.0%}" for t in TARGETS))
    cells = 0
    for look in LOOKBACKS:
        row = f"   {look:>9}"
        for tgt in TARGETS:
            d = np.median([stats(run(px[s].pct_change().fillna(0.0), tb, tgt, BASE_CAP, look))[2]
                           - stats(px[s].pct_change().fillna(0.0))[2] for s in cov.symbol])
            cells += d > 0
            row += f"{d:>+9.2f}{'*' if d > 0 else ' '}"
        print(row)
    print(f"   positive cells: {cells}/12   (prereg requires >= 8)")

    c = [med_sh >= 0.10, n_up >= 12, med_dd >= 0.20, med_10 >= 0.10, cells >= 8]
    print(f"\n{'='*70}\n   PRE-REGISTERED VERDICT\n{'='*70}")
    for i, (lab, val, ok) in enumerate([
            ("median ΔSharpe >= +0.10", f"{med_sh:+.2f}", c[0]),
            ("markets improving >= 12/17", f"{n_up}/17", c[1]),
            ("median drawdown cut >= 20%", f"{med_dd*100:+.0f}%", c[2]),
            ("median ΔSharpe at 10bps >= +0.10", f"{med_10:+.2f}", c[3]),
            ("parameter cells positive >= 8/12", f"{cells}/12", c[4])], 1):
        print(f"   {i}  {lab:<36}{val:>8}   {'PASS' if ok else 'FAIL'}")
    print(f"\n   ══ {'PASS' if all(c) else 'FAIL'} ══   (all five required)")


if __name__ == "__main__":
    main()
