# ======================================================================
#  Volatility targeting on emerging markets, and on the recent regime.
#  Fixed in research/prereg_voltarget_em.md, committed first.
#  Mechanics from research/mechanics.py.
#
#    .venv\Scripts\python.exe research\voltarget_em.py
# ======================================================================

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research"))
from mechanics import hold_from_month_end, performance  # noqa: E402

DB = ROOT / "data" / "vol.db"
TARGET, CAP, LOOK, COST, MARGIN = 0.12, 1.5, 21, 2 / 1e4, 0.015


def load():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cov = pd.read_sql_query("SELECT symbol, note FROM coverage WHERE kind='em_country' "
                            "ORDER BY note", conn)
    px = {}
    for s in cov.symbol:
        d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                              conn, params=(s,))
        d["date"] = pd.to_datetime(d.date)
        px[s] = d.set_index("date").close
    t = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol='IRX' ORDER BY date", conn)
    conn.close()
    t["date"] = pd.to_datetime(t.date)
    return cov, px, t.set_index("date").close / 100 / 252


def pair(px, rf, lo=None, hi=None):
    p = px.loc[lo:hi] if (lo or hi) else px
    if len(p) < 500:
        return None, None, None
    eq = p.pct_change().fillna(0.0)
    r = rf.reindex(eq.index).ffill().fillna(0.0)
    w = hold_from_month_end((TARGET / (eq.rolling(LOOK).std() * np.sqrt(252))).clip(0, CAP),
                            eq.index)
    rate = np.where(w.to_numpy() <= 1.0, r.to_numpy(), r.to_numpy() + MARGIN / 252)
    tg = (w * eq + (1 - w) * rate - w.diff().abs().fillna(0.0) * COST).dropna()
    return eq.loc[tg.index], tg, r


def block(title, cov, px, rf, lo, hi):
    print("\n" + "=" * 92 + f"\n{title}\n" + "=" * 92)
    print(f"   {'market':<18}{'b&h ret':>9}{'exSh':>7}{'maxDD':>7}  |"
          f"{'tgt ret':>9}{'exSh':>7}{'maxDD':>7}{'ΔexSh':>8}{'ΔDD':>7}")
    ds, dd = [], []
    for _, c in cov.iterrows():
        bh, tg, r = pair(px[c.symbol], rf, lo, hi)
        if bh is None:
            print(f"   {c.note:<18}too short"); continue
        b, t = performance(bh, r), performance(tg, r)
        x = t["sharpe"] - b["sharpe"]
        y = (abs(b["max_dd"]) - abs(t["max_dd"])) / abs(b["max_dd"])
        ds.append(x); dd.append(y)
        print(f"   {c.note:<18}{b['ret']*100:>8.2f}%{b['sharpe']:>7.2f}{b['max_dd']*100:>7.0f}%"
              f"  |{t['ret']*100:>8.2f}%{t['sharpe']:>7.2f}{t['max_dd']*100:>7.0f}%"
              f"{x:>+8.2f}{y*100:>7.0f}%")
    n = len(ds)
    print(f"   {'median':<18}{'':>23}  |{'':>23}{np.median(ds):>+8.2f}{np.median(dd)*100:>7.0f}%")
    c1, c2, c3 = np.median(ds) >= 0.10, sum(v > 0 for v in ds) >= 10, np.median(dd) >= 0.20
    print(f"\n   1  median ΔexSharpe >= +0.10    {np.median(ds):+.3f}   {'PASS' if c1 else 'FAIL'}")
    print(f"   2  funds improving >= 10/{n}     {sum(v>0 for v in ds)}/{n}   {'PASS' if c2 else 'FAIL'}")
    print(f"   3  median ΔDD >= 20%            {np.median(dd)*100:+.0f}%   {'PASS' if c3 else 'FAIL'}")
    ok = c1 and c2 and c3
    print(f"   ══ {'PASS' if ok else 'FAIL'} ══")
    return dict(sh=np.median(ds), up=sum(v > 0 for v in ds), n=n,
                dd=np.median(dd), c1=c1, c2=c2, c3=c3, ok=ok)


def main():
    cov, px, rf = load()
    D = block("TEST D — 14 emerging markets, full history each", cov, px, rf, None, None)
    E = block("TEST E — the same 14, 2015-01 to 2026-09 (the recent regime)",
              cov, px, rf, "2015-01-01", None)

    print("\n" + "=" * 92 + "\n   PREDICTIONS MADE IN ADVANCE\n" + "=" * 92)
    p1 = D["c3"] and E["c3"]
    p2 = not (D["c1"] and E["c1"])
    p3 = E["sh"] < D["sh"]
    print(f"   drawdown criterion passes in both      D {D['c3']}, E {E['c3']}"
          f"        {'CORRECT' if p1 else 'WRONG'}")
    print(f"   Sharpe criterion fails in at least one D {D['c1']}, E {E['c1']}"
          f"        {'CORRECT' if p2 else 'WRONG'}")
    print(f"   E does worse than D on Sharpe          {E['sh']:+.3f} vs {D['sh']:+.3f}"
          f"   {'CORRECT' if p3 else 'WRONG'}")
    print(f"\n   {sum((p1,p2,p3))}/3 predictions correct")
    if D["c3"] and E["c3"] and not (D["c1"] and E["c1"]):
        print("\n   → consistent with §8d: a drawdown tool, not a return tool")
    elif D["c1"] and E["c1"]:
        print("\n   → §8d is WRONG and must be revised: Sharpe improved in both")
    if not (D["c3"] and E["c3"]):
        print("\n   → the one consistent finding of the programme has broken")


if __name__ == "__main__":
    main()
