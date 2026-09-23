# ======================================================================
#  Three fresh tests of volatility targeting.
#  Rule, samples, verdicts and one directional hypothesis are fixed in
#  research/prereg_voltarget_deep.md, committed first (6d9e57d).
#  Mechanics come from research/mechanics.py.
#
#    .venv\Scripts\python.exe research\voltarget_deep.py
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
TARGET, CAP, LOOK, COST = 0.12, 1.5, 21, 2 / 1e4
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
LONG_C  = {"N225": ("Japan", "1965", "1995"), "GSPTSE": ("Canada", "1979", "1995"),
           "FTSE": ("UK", "1984", "1995"), "HSI": ("Hong Kong", "1986", "1995"),
           "GDAXI": ("Germany (total ret)", "1987", "1995")}


def series(sym):
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                          conn, params=(sym,))
    conn.close()
    d["date"] = pd.to_datetime(d.date)
    return d.set_index("date").close


def rf_daily(index):
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol='IRX' ORDER BY date", conn)
    conn.close()
    d["date"] = pd.to_datetime(d.date)
    return (d.set_index("date").close / 100 / 252).reindex(index).ffill().fillna(0.0)


def run(px, q, lo, hi):
    """Buy-and-hold and targeted returns, with an assumed dividend yield q
    added to the equity leg (0 for a total-return series)."""
    px = px.loc[lo:hi]
    if len(px) < 500:
        return None, None
    eq = px.pct_change().fillna(0.0) + q / 252          # total-return equity leg
    rf = rf_daily(eq.index)
    raw = (TARGET / (eq.rolling(LOOK).std() * np.sqrt(252))).clip(0, CAP)
    w = hold_from_month_end(raw, eq.index)
    rate = np.where(w.to_numpy() <= 1.0, rf.to_numpy(), rf.to_numpy() + 0.015 / 252)
    tgt = (w * eq + (1 - w) * rate - w.diff().abs().fillna(0.0) * COST).dropna()
    return eq.loc[tgt.index], tgt


def line(lab, bh, tg, rf):
    b, t = performance(bh, rf), performance(tg, rf)
    d_sh = t["sharpe"] - b["sharpe"]
    d_dd = (abs(b["max_dd"]) - abs(t["max_dd"])) / abs(b["max_dd"])
    print(f"   {lab:<22}{b['ret']*100:>7.2f}%{b['sharpe']:>7.2f}{b['max_dd']*100:>7.0f}%"
          f"  |{t['ret']*100:>8.2f}%{t['sharpe']:>7.2f}{t['max_dd']*100:>7.0f}%"
          f"{d_sh:>+8.2f}{d_dd*100:>7.0f}%")
    return d_sh, d_dd, b["sharpe"]


HEAD = (f"   {'':<22}{'b&h ret':>8}{'exSh':>7}{'maxDD':>7}  |{'tgt ret':>9}"
        f"{'exSh':>7}{'maxDD':>7}{'ΔexSh':>8}{'ΔDD':>7}")
rows = []


def main():
    print("Three fresh samples. Verdicts at an assumed 4% dividend yield on")
    print("price-only series, the conservative choice.\n")

    # ── Test A ──
    print("=" * 96 + "\nTEST A — S&P 500 price index, 1928-1992 (65 years never touched)\n" + "=" * 96)
    print(HEAD)
    res_a = {}
    for q in (0.0, 0.02, 0.04):
        bh, tg = run(series("GSPC"), q, "1928", "1992")
        res_a[q] = line(f"assumed yield {q:.0%}", bh, tg, rf_daily(bh.index))
    rows.append(("GSPC 1928-92", res_a[0.04][2], res_a[0.04][0]))
    a1, a2 = res_a[0.04][0] >= 0.10, res_a[0.04][1] >= 0.20

    # ── Test B ──
    print("\n" + "=" * 96 + "\nTEST B — 9 SPDR sectors, 1999-2026 (total return, no yield needed)\n" + "=" * 96)
    print(HEAD)
    ds, dd = [], []
    for s in SECTORS:
        bh, tg = run(series(s), 0.0, "1999", "2026")
        x, y, bsh = line(s, bh, tg, rf_daily(bh.index))
        ds.append(x); dd.append(y); rows.append((s, bsh, x))
    b1 = float(np.median(ds)) >= 0.10
    b2 = int(sum(v > 0 for v in ds)) >= 7
    b3 = float(np.median(dd)) >= 0.20
    print(f"   {'median':<22}{'':>22}  |{'':>24}{np.median(ds):>+8.2f}{np.median(dd)*100:>7.0f}%")
    print(f"   sectors improving: {sum(v>0 for v in ds)}/9")

    # ── Test C ──
    print("\n" + "=" * 96 + "\nTEST C — long national indices, pre-1996 only\n" + "=" * 96)
    print(HEAD)
    cs = []
    for sym, (name, lo, hi) in LONG_C.items():
        q = 0.0 if sym == "GDAXI" else 0.04
        bh, tg = run(series(sym), q, lo, hi)
        if bh is None:
            print(f"   {name:<22}too short"); continue
        x, y, bsh = line(f"{name} {lo}-", bh, tg, rf_daily(bh.index))
        cs.append(x); rows.append((name, bsh, x))
    c1 = float(np.median(cs)) >= 0.10
    c2 = int(sum(v > 0 for v in cs)) >= 3
    print(f"   {'median':<22}{'':>22}  |{'':>24}{np.median(cs):>+8.2f}")
    print(f"   markets improving: {sum(v>0 for v in cs)}/{len(cs)}")

    # ── verdicts ──
    A, Bv, C = (a1 and a2), (b1 and b2 and b3), (c1 and c2)
    print("\n" + "=" * 96 + "\n   PRE-REGISTERED VERDICTS\n" + "=" * 96)
    print(f"   A  S&P 500 1928-1992      ΔexSharpe {res_a[0.04][0]:+.2f} (need +0.10), "
          f"ΔDD {res_a[0.04][1]*100:+.0f}% (need +20%)   {'PASS' if A else 'FAIL'}")
    print(f"   B  9 sectors             median {np.median(ds):+.2f}, {sum(v>0 for v in ds)}/9 up, "
          f"DD {np.median(dd)*100:+.0f}%   {'PASS' if Bv else 'FAIL'}")
    print(f"   C  national pre-1996     median {np.median(cs):+.2f}, "
          f"{sum(v>0 for v in cs)}/{len(cs)} up   {'PASS' if C else 'FAIL'}")
    n = sum((A, Bv, C))
    print(f"\n   ══ {n}/3 pass — mechanism {'STANDS' if n >= 2 else 'DOES NOT STAND'} ══")
    print("   (not independent: sectors sit inside the US market, the indices correlate)")

    # ── the directional hypothesis ──
    R = pd.DataFrame(rows, columns=["series", "bh_sharpe", "improvement"]).dropna()
    r = R.bh_sharpe.corr(R.improvement)
    print(f"\n   HYPOTHESIS (recorded before the fact): targeting helps most where")
    print(f"   buy-and-hold is worst, so r <= -0.30 across the series.")
    print(f"      n = {len(R)} series, r = {r:+.3f}   "
          f"{'CONFIRMED' if r <= -0.30 else 'NOT CONFIRMED'}")
    print(f"      strongest markets: " +
          ", ".join(f"{x.series} (Sh {x.bh_sharpe:.2f}, Δ{x.improvement:+.2f})"
                    for _, x in R.nlargest(3, "bh_sharpe").iterrows()))
    print(f"      weakest markets:   " +
          ", ".join(f"{x.series} (Sh {x.bh_sharpe:.2f}, Δ{x.improvement:+.2f})"
                    for _, x in R.nsmallest(3, "bh_sharpe").iterrows()))


if __name__ == "__main__":
    main()
