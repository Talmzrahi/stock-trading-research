# ═══════════════════════════════════════════════════════════════════════
#  The pre-registered out-of-sample test — research/prereg_midsmall.md.
#
#  Run ONCE, after research/oos_midsmall_data.py. The specification and
#  decision rule live in the pre-registration and are not tuned here.
#
#  One implementation detail the pre-registration did not spell out: a
#  position whose price series ends inside the 60 sessions (acquired,
#  delisted) is valued at its last available close, as the trading engine's
#  no-data exit does, rather than dropped — dropping it would quietly
#  remove the worst outcomes.
#
#    python research/oos_midsmall.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.data import load_closes, load_earnings  # noqa: E402
from trader.events import build_events, point_in_time  # noqa: E402
from trader.market_calendar import trading_days  # noqa: E402
from trader.signals.sue import SueSignal  # noqa: E402

DB_FILE   = ROOT / "data" / "oos_midsmall.db"
HOLD      = 60
COST      = 20 / 1e4
COST_HIGH = 40 / 1e4
PRIMARY   = 0.95
MIN_N     = 300


def clustered(g, col, cost):
    m = (g[col] - cost).groupby(g.entry_date).mean()
    if len(m) < 5:
        return np.nan, np.nan, len(m)
    return float(m.mean()), float(stats.ttest_1samp(m, 0).pvalue), len(m)


def main():
    conn = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    closes, earn = load_closes(conn), load_earnings(conn)
    u = pd.read_sql_query("""SELECT h.as_of, h.symbol, h.idx, i.firm FROM universe_history h
                             LEFT JOIN universe_ids i USING (as_of, symbol, idx)""", conn)
    conn.close()
    u["as_of"] = pd.to_datetime(u.as_of)
    u["firm"] = u.firm.fillna("SYM:" + u.symbol)
    universe = u[["as_of", "symbol", "firm"]].drop_duplicates()

    cal = trading_days(closes.index.min(), closes.index.max())
    closes = closes.reindex(cal)
    ev = build_events(earn, cal, closes, universe)
    ev["pctl"] = SueSignal().score(ev)
    ev = ev[ev.pit & ev.pctl.notna() & (ev.entry_idx + HOLD < len(cal))].copy()

    _, in400 = point_in_time(ev.symbol, ev.entry_date, u[u.idx == "400"][["as_of", "symbol", "firm"]])
    ev["member_of"] = np.where(in400, "400", "600")
    ev["etf"] = np.where(in400, "IJH", "IJR")

    ff = closes.ffill()
    col = {s: i for i, s in enumerate(ff.columns)}
    px = ff.to_numpy()
    e, x = ev.entry_idx.to_numpy(), ev.entry_idx.to_numpy() + HOLD
    c = ev.symbol.map(col).to_numpy()
    stock = px[x, c] / px[e, c] - 1
    etf = np.array([px[xi, col[t]] / px[ei, col[t]] - 1 for ei, xi, t in zip(e, x, ev.etf)])
    spy = px[x, col["SPY"]] / px[e, col["SPY"]] - 1
    ev["excess"], ev["excess_spy"] = stock - etf, stock - spy
    ev = ev[np.isfinite(ev.excess)]

    print(f"Point-in-time events with a trailing percentile: {len(ev):,} "
          f"({ev.entry_date.min().date()} → {ev.entry_date.max().date()}), "
          f"{ev.firm.nunique():,} firms; S&P 400 {int((ev.member_of == '400').sum()):,}, "
          f"S&P 600 {int((ev.member_of == '600').sum()):,}")

    top = ev[ev.pctl >= PRIMARY]
    mean, p, dates = clustered(top, "excess", COST)
    verdict = ("INCONCLUSIVE" if len(top) < MIN_N else
               "PASS" if mean > 0 and p < 0.05 else "FAIL")
    print(f"\n══ PRIMARY (pre-registered): top 5%, {HOLD}-session excess vs IJH/IJR, net 20bps ══")
    print(f"   events {len(top):,}   dates {dates:,}   mean {mean*100:+.3f}pp   p={p:.4f}")
    print(f"══ VERDICT: {verdict} ══")

    print("\n── Reported, not decisive ─────────────────────────────────────")
    for cut in [0.90, 0.95, 0.97, 0.98]:
        g = ev[ev.pctl >= cut]
        m, pp, _ = clustered(g, "excess", COST)
        m40, p40, _ = clustered(g, "excess", COST_HIGH)
        ms, ps, _ = clustered(g, "excess_spy", COST)
        print(f"   top {round((1-cut)*100):>2}%  n={len(g):>5,}  vs ETF {m*100:+.3f}pp (p={pp:.3f})  "
              f"@40bps {m40*100:+.3f}pp (p={p40:.3f})  vs SPY {ms*100:+.3f}pp (p={ps:.3f})")
    for name, g in top.groupby("member_of"):
        m, pp, _ = clustered(g, "excess", COST)
        print(f"   top 5% S&P {name}: n={len(g):,}  {m*100:+.3f}pp (p={pp:.3f})")
    thirds = np.array_split(np.sort(top.entry_date.unique()), 3)
    for part in thirds:
        g = top[top.entry_date.isin(part)]
        m, pp, _ = clustered(g, "excess", COST)
        print(f"   top 5% {pd.Timestamp(part[0]).date()} → {pd.Timestamp(part[-1]).date()}: "
              f"n={len(g):,}  {m*100:+.3f}pp (p={pp:.3f})")
    edges = [0.0, 0.5, 0.9, 0.95, 0.98, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = ev[(ev.pctl >= lo) & (ev.pctl < hi)]
        m, pp, _ = clustered(g, "excess", COST)
        print(f"   pctl {lo:.2f}–{min(hi, 1):.2f}: n={len(g):>6,}  {m*100:+.3f}pp (p={pp:.3f})")


if __name__ == "__main__":
    main()
