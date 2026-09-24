# ======================================================================
#  What separates the 2 passing tests from the 5 failing ones?
#
#  POST-HOC, ON SPENT DATA. Every sample here has already been used for a
#  pre-registered verdict. Nothing below is a test; it is an attempt to
#  explain those verdicts, and anything it suggests is a hypothesis for
#  data that does not yet exist. One story was already tried and failed:
#  "targeting helps most where buy-and-hold is worst", r = -0.089.
#
#  The mechanism being probed. Targeting cuts exposure when volatility is
#  high. Volatility is persistent, so that reliably cuts variance. Whether
#  it ALSO helps return depends on what follows high volatility: more
#  losses (a slow bear) or a rebound (a V-shape). So the candidate
#  differentiator is the correlation between month-end volatility and the
#  NEXT month's return, per series -- negative should help, positive hurt.
#
#  61 series across the 7 tests. They are not independent: countries and
#  sectors overlap in time, and EM full/recent are the same markets twice.
#
#    .venv\Scripts\python.exe research\voltarget_synthesis.py
# ======================================================================

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from mechanics import hold_from_month_end, performance  # noqa: E402

DB = ROOT / "data" / "vol.db"
TARGET, CAP, LOOK, COST, MARGIN = 0.12, 1.5, 21, 2 / 1e4, 0.015


def _conn():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def px(sym):
    with _conn() as c:
        d = pd.read_sql_query("SELECT date, close FROM prices WHERE symbol=? ORDER BY date",
                              c, params=(sym,))
    d["date"] = pd.to_datetime(d.date)
    return d.set_index("date").close


def kind(k):
    with _conn() as c:
        return pd.read_sql_query("SELECT symbol, note FROM coverage WHERE kind=? ORDER BY symbol",
                                 c, params=(k,))


RF = px("IRX") / 100 / 252


def pair(s, lo, hi, q):
    p = s.loc[lo:hi]
    eq = p.pct_change().fillna(0.0) + q / 252
    rf = RF.reindex(eq.index).ffill().fillna(0.0)
    w = hold_from_month_end((TARGET / (eq.rolling(LOOK).std() * np.sqrt(252))).clip(0, CAP),
                            eq.index)
    rate = np.where(w.to_numpy() <= 1.0, rf.to_numpy(), rf.to_numpy() + MARGIN / 252)
    tg = (w * eq + (1 - w) * rate - w.diff().abs().fillna(0.0) * COST).dropna()
    return eq.loc[tg.index], tg, rf.loc[tg.index]


def describe(bh, tg, rf):
    b, t = performance(bh, rf), performance(tg, rf)
    rv = bh.rolling(LOOK).std() * np.sqrt(252)
    m = bh.index.to_period("M")
    rv_m = rv.groupby(m).last()
    r_m = (1 + bh).groupby(m).prod() - 1
    vol_ret = rv_m.corr(r_m.shift(-1))          # the mechanism
    persist = rv_m.corr(rv_m.shift(-1))
    eq = (1 + bh).cumprod()
    trough = (eq / eq.cummax() - 1).idxmin()
    peak = eq.loc[:trough].idxmax()
    months_down = (trough - peak).days / 30.44
    keep = ~((bh.index >= peak) & (bh.index <= trough + pd.Timedelta(days=182)))
    b2, t2 = performance(bh[keep], rf[keep]), performance(tg[keep], rf[keep])
    return dict(
        d_sh=t["sharpe"] - b["sharpe"],
        d_dd=(abs(b["max_dd"]) - abs(t["max_dd"])) / abs(b["max_dd"]),
        bh_sh=b["sharpe"], bh_ret=b["ret"], tg_ret=t["ret"],
        calmar_b=b["ret"] / abs(b["max_dd"]), calmar_t=t["ret"] / abs(t["max_dd"]),
        vol_ret=vol_ret, persist=persist, months_down=months_down,
        worst=f"{peak:%Y-%m}", d_sh_ex=t2["sharpe"] - b2["sharpe"])


def samples():
    out = [("SPY 1993-2001", "FAIL", "SPY", "SPY", "1993", "2001", 0.0),
           ("S&P 1928-1992", "PASS", "GSPC", "S&P 500", "1928", "1992", 0.04)]
    for _, r in kind("country_etf").iterrows():
        out.append(("17 countries", "PASS", r.symbol, r.note, "1996", "2026", 0.0))
    for _, r in kind("sector_etf").iterrows():
        out.append(("9 sectors", "FAIL", r.symbol, r.note, "1999", "2026", 0.0))
    for sym, name, lo, q in [("N225", "Japan", "1965", .04), ("GSPTSE", "Canada", "1979", .04),
                             ("FTSE", "UK", "1984", .04), ("HSI", "Hong Kong", "1986", .04),
                             ("GDAXI", "Germany", "1987", 0.0)]:
        out.append(("national pre-1996", "FAIL", sym, name, lo, "1995", q))
    for _, r in kind("em_country").iterrows():
        out.append(("EM full history", "FAIL", r.symbol, r.note, None, None, 0.0))
        out.append(("EM 2015-2026", "FAIL", r.symbol, r.note, "2015", None, 0.0))
    return out


ORDER = ["S&P 1928-1992", "17 countries", "SPY 1993-2001", "9 sectors",
         "national pre-1996", "EM full history", "EM 2015-2026"]


def main():
    rows = []
    for test, verdict, sym, name, lo, hi, q in samples():
        bh, tg, rf = pair(px(sym), lo, hi, q)
        if len(bh) < 500:
            continue
        rows.append(dict(test=test, verdict=verdict, series=name, **describe(bh, tg, rf)))
    R = pd.DataFrame(rows)
    print(f"{len(R)} series across 7 tests. POST-HOC on spent data: hypotheses, not findings.\n")

    print("-- 1. Per test (medians) --")
    print(f"   {'test':<20}{'verdict':>8}{'dExSh':>8}{'dDD':>7}{'vol->next ret':>15}"
          f"{'persist':>9}{'worst DD took':>15}{'dExSh ex-worst':>16}")
    for t in ORDER:
        g = R[R.test == t]
        print(f"   {t:<20}{g.verdict.iloc[0]:>8}{g.d_sh.median():>+8.3f}"
              f"{g.d_dd.median()*100:>6.0f}%{g.vol_ret.median():>+15.3f}"
              f"{g.persist.median():>9.2f}{g.months_down.median():>11.0f} mo"
              f"{g.d_sh_ex.median():>+16.3f}")

    print("\n-- 2. What predicts the Sharpe improvement, across all series? --")
    for col, lab in [("vol_ret", "vol -> next-month return (the mechanism)"),
                     ("months_down", "months the worst drawdown took to bottom"),
                     ("persist", "volatility persistence"),
                     ("bh_sh", "buy-and-hold Sharpe (the failed story)")]:
        r = R.d_sh.corr(R[col])
        rs = R.d_sh.corr(R[col], method="spearman")
        print(f"   {lab:<44} r = {r:+.3f}   rank r = {rs:+.3f}")

    print("\n-- 3. Passing tests vs failing tests (medians over their series) --")
    for v in ("PASS", "FAIL"):
        g = R[R.verdict == v]
        print(f"   {v}  n={len(g):>2}   vol->next ret {g.vol_ret.median():+.3f}   "
              f"worst DD took {g.months_down.median():.0f} mo   "
              f"persistence {g.persist.median():.2f}   b&h Sharpe {g.bh_sh.median():.2f}")

    print("\n-- 4. Is the improvement one episode? Remove each series' worst drawdown --")
    for t in ORDER:
        g = R[R.test == t]
        print(f"   {t:<20} dExSh {g.d_sh.median():+.3f}  ->  without its worst episode "
              f"{g.d_sh_ex.median():+.3f}")
    w = R[R.test == "S&P 1928-1992"].iloc[0]
    print(f"   (S&P 1928-1992's worst drawdown began {w.worst})")

    print("\n-- 5. Consistency, metric by metric (share of series that improve) --")
    print(f"   {'drawdown smaller':<26}{(R.d_dd > 0).mean():>6.0%} of {len(R)} series")
    print(f"   {'excess Sharpe higher':<26}{(R.d_sh > 0).mean():>6.0%}")
    print(f"   {'return per unit DD higher':<26}{(R.calmar_t > R.calmar_b).mean():>6.0%}")
    print(f"   {'raw return LOWER':<26}{(R.tg_ret < R.bh_ret).mean():>6.0%}"
          f"   (the price of the insurance)")

    R.to_csv(ROOT / "data" / "voltarget_synthesis.csv", index=False)


if __name__ == "__main__":
    main()
