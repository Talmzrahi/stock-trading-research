# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: the volatility-targeting test.
#
#  The rule, the samples and the verdict are fixed in
#  research/prereg_voltarget.md, committed first (2eb5b85). This file
#  only executes it.
#
#  Order matters and is deliberate:
#    1. development battery, 2002-2026 -- ALREADY SEEN, declared in the
#       pre-registration as not evidence. Printed for completeness.
#    2. the premise check on the holdout, BEFORE any return is scored.
#    3. the holdout verdict, 1993-2001. One shot.
#
#    .venv\Scripts\python.exe research\voltarget_test.py
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
MARGIN_SPREAD = 0.015            # over the 13-week T-bill, on borrowed notional
DEV, HOLD = ("2002-01-01", "2026-12-31"), ("1993-01-01", "2001-12-31")


def load():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    w = {}
    for s in ("SPY", "SHY", "VIX", "IRX"):
        d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                              conn, params=(s,))
        d["date"] = pd.to_datetime(d.date)
        w[s] = d.set_index("date").close
    conn.close()
    idx = w["SPY"].index
    spy = w["SPY"].pct_change().fillna(0.0)
    cash = w["SHY"].pct_change().reindex(idx).fillna(0.0)     # 0% before SHY exists, per the prereg
    vix = w["VIX"].reindex(idx).ffill()
    tb = (w["IRX"].reindex(idx).ffill() / 100).fillna(0.0)
    return idx, spy, cash, vix, tb


def weights(idx, spy, vix, target, cap, look, spec):
    """Month-end weight from information known at that close, applied from the
    next trading day — so no weight is ever informed by the return it earns."""
    fc = (spy.rolling(look).std() * np.sqrt(252)) if spec == "A" else (vix / 100.0)
    raw = (target / fc).clip(lower=0, upper=cap)
    month_end = idx.to_series().groupby(idx.to_period("M")).transform("max") == idx
    return raw.where(month_end.to_numpy()).shift(1).ffill()


def run(idx, spy, cash, vix, tb, target, cap, look, spec="A", cost_bps=2.0):
    w = weights(idx, spy, vix, target, cap, look, spec)
    rate = np.where(w.to_numpy() <= 1.0, cash.to_numpy(), (tb.to_numpy() + MARGIN_SPREAD) / 252)
    gross = w * spy + (1 - w) * rate                      # borrowing shows up as a negative leg
    turn = w.diff().abs().fillna(0.0)
    return (gross - turn * cost_bps / 1e4).dropna()


def stats(r):
    if len(r) < 60:
        return dict(ret=np.nan, vol=np.nan, sharpe=np.nan, dd=np.nan)
    ann = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return dict(ret=ann, vol=vol, sharpe=ann / vol if vol else np.nan,
                dd=float((eq / eq.cummax() - 1).min()))


def line(lab, s):
    print(f"   {lab:<34}{s['ret']*100:>8.2f}%{s['vol']*100:>8.1f}%"
          f"{s['sharpe']:>9.2f}{s['dd']*100:>10.1f}%")


def window(idx, spy, cash, vix, tb, lo, hi):
    m = (idx >= lo) & (idx <= hi)
    return idx[m], spy[m], cash[m], vix[m], tb[m]


def battery(pack, title, note=""):
    idx, spy, cash, vix, tb = pack
    print(f"\n{'='*78}\n{title}\n{note}{'='*78}")
    print(f"   {'':<34}{'return':>9}{'vol':>8}{'Sharpe':>9}{'max DD':>10}")
    bench = stats(spy)
    line("SPY buy and hold", bench)
    for name, tgt, cap in POSTURES:
        for spec in ("A", "B"):
            s = stats(run(idx, spy, cash, vix, tb, tgt, cap, BASE_LOOK, spec))
            line(f"{name} {tgt:.0%}/{cap:g}x, spec {spec}", s)
    return bench


def main():
    idx, spy, cash, vix, tb = load()
    dev = window(idx, spy, cash, vix, tb, *DEV)
    hold = window(idx, spy, cash, vix, tb, *HOLD)

    battery(dev, "1. DEVELOPMENT 2002-2026 — ALREADY SEEN, NOT EVIDENCE",
            "   Declared in the pre-registration. Printed for completeness only.\n")

    # ── 2. the premise, measured on the holdout BEFORE returns are scored ──
    hi, hs = hold[0], hold[1]
    rv = hs.rolling(21).std() * np.sqrt(252)
    ac = rv.corr(rv.shift(-21))
    vr = rv.corr(hs.shift(-21).rolling(21).mean())
    print(f"\n{'='*78}\n2. PREMISE ON THE HOLDOUT (measured before any return is scored)\n{'='*78}")
    print(f"   volatility autocorrelation, 21 days ahead : {ac:+.3f}   "
          f"(prereg: fails below ~0.30)")
    print(f"   realised vol vs NEXT 21 days' return      : {vr:+.3f}   "
          f"(prereg: fails above ~0.30 in absolute value)")
    ok = ac >= 0.30 and abs(vr) <= 0.30
    print(f"   → the mechanism {'HOLDS' if ok else 'DOES NOT HOLD'} in this era")

    # ── 3. the holdout verdict ──
    bench = battery(hold, "3. HOLDOUT 1993-2001 — ONE SHOT",
                    f"   {len(hi):,} trading days, never examined before now.\n")
    base = stats(run(*hold, BASE_TARGET, BASE_CAP, BASE_LOOK, "A"))
    d_sharpe = base["sharpe"] - bench["sharpe"]
    d_dd = (abs(bench["dd"]) - abs(base["dd"])) / abs(bench["dd"])

    print(f"\n   ── parameter grid, standard cap 1.5x, spec A (plateau test) ──")
    print(f"   {'lookback':>9}" + "".join(f"{t:>12.0%}" for t in TARGETS))
    wins = 0
    for look in LOOKBACKS:
        row = f"   {look:>9}"
        for tgt in TARGETS:
            s = stats(run(*hold, tgt, BASE_CAP, look, "A"))
            better = s["sharpe"] > bench["sharpe"]
            wins += better
            row += f"{s['sharpe']:>11.2f}{'*' if better else ' '}"
        print(row)
    print(f"   cells beating SPY: {wins}/12   (prereg requires >= 8)   * = beats SPY")

    print(f"\n   ── cost sensitivity, standard posture ──")
    costs = {b: stats(run(*hold, BASE_TARGET, BASE_CAP, BASE_LOOK, "A", b))["sharpe"]
             for b in (2, 5, 10)}
    print("   " + "   ".join(f"{b}bps: Sharpe {v:.2f}" for b, v in costs.items()))

    c1, c2 = d_sharpe >= 0.10, d_dd >= 0.20
    c3, c4 = wins >= 8, costs[10] - bench["sharpe"] >= 0.10
    print(f"\n{'='*78}\n   PRE-REGISTERED VERDICT\n{'='*78}")
    print(f"   1  Sharpe improvement >= +0.10     {d_sharpe:+.2f}   {'PASS' if c1 else 'FAIL'}")
    print(f"   2  drawdown reduced >= 20%         {d_dd*100:+.0f}%   {'PASS' if c2 else 'FAIL'}")
    print(f"   3  plateau, >= 8 of 12 cells       {wins}/12   {'PASS' if c3 else 'FAIL'}")
    print(f"   4  survives 10bps                  {costs[10]-bench['sharpe']:+.2f}   "
          f"{'PASS' if c4 else 'FAIL'}")
    print(f"\n   ══ {'PASS' if all((c1,c2,c3,c4)) else 'FAIL'} ══   "
          f"(all four required; a fail stops the build)")


if __name__ == "__main__":
    main()
