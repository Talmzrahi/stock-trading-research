# ═══════════════════════════════════════════════════════════════════════
#  Replay the live strategy over the last N years and compare it to SPY.
#
#  Same engine, config and costs the daily run uses — this is the backtest
#  path of the one code path, not a separate model.
#
#    python research/simulate.py            # last 5 years
#    python research/simulate.py --years 3
# ═══════════════════════════════════════════════════════════════════════

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import cagr, load_market, run  # noqa: E402
from trader.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5)
    args = ap.parse_args()

    cfg = load_config()
    print(f"Strategy: top {round((1 - cfg.cutoff) * 100)}% surprise · hold {cfg.hold_days} sessions · "
          f"stop {cfg.stop_k} sd · ${cfg.initial_capital:,.0f} start · "
          f"{cfg.cost_bps_side * 2:.0f}bps round trip\nLoading market …", flush=True)
    market = load_market(cfg)
    first = market.cal[-1] - pd.Timedelta(days=int(365.25 * args.years))
    start = int(market.cal.searchsorted(first))
    res = run(market, cfg, start=start)

    eq = res.equity.equity
    spy = market.closes[cfg.benchmark].reindex(eq.index).ffill()
    spy_eq = cfg.initial_capital * spy / spy.iloc[0]

    print(f"\n── Year by year, {eq.index[0]:%Y-%m-%d} → {eq.index[-1]:%Y-%m-%d} ─────────────")
    print(f"   {'year':<6}{'strategy':>11}{'S&P 500':>11}{'difference':>13}{'trades':>8}{'ends with':>12}")
    rows = []
    for year, seg in eq.groupby(eq.index.year):
        base = eq.loc[:seg.index[0]]
        prev = base.iloc[-2] if len(base) > 1 else cfg.initial_capital
        s_ret = seg.iloc[-1] / prev - 1
        sp = spy_eq.loc[seg.index]
        sp_prev = spy_eq.loc[:seg.index[0]].iloc[-2] if len(spy_eq.loc[:seg.index[0]]) > 1 else cfg.initial_capital
        b_ret = sp.iloc[-1] / sp_prev - 1
        n = int(((res.trades.entry_date.dt.year == year).sum()) if len(res.trades) else 0)
        rows.append((year, s_ret, b_ret))
        print(f"   {year:<6}{s_ret*100:>+10.2f}%{b_ret*100:>+10.2f}%{(s_ret-b_ret)*100:>+12.2f}pp"
              f"{n:>8}{seg.iloc[-1]:>12,.0f}")

    r = eq.pct_change().dropna()
    rb = spy_eq.pct_change().dropna()
    beat = sum(1 for _, s, b in rows if s > b)
    print(f"\n── Overall ────────────────────────────────────────────────────")
    print(f"   {'':<22}{'strategy':>14}{'S&P 500':>14}")
    for label, a, b in [
        ("final value", f"${eq.iloc[-1]:,.0f}", f"${spy_eq.iloc[-1]:,.0f}"),
        ("total return", f"{(eq.iloc[-1]/cfg.initial_capital-1)*100:+.1f}%",
                         f"{(spy_eq.iloc[-1]/cfg.initial_capital-1)*100:+.1f}%"),
        ("annualised", f"{cagr(eq)*100:+.2f}%", f"{cagr(spy_eq)*100:+.2f}%"),
        ("volatility", f"{r.std()*math.sqrt(252)*100:.1f}%", f"{rb.std()*math.sqrt(252)*100:.1f}%"),
        ("worst drawdown", f"{(eq/eq.cummax()-1).min()*100:.1f}%", f"{(spy_eq/spy_eq.cummax()-1).min()*100:.1f}%"),
        ("Sharpe (rf=0)", f"{r.mean()/r.std()*math.sqrt(252):.2f}", f"{rb.mean()/rb.std()*math.sqrt(252):.2f}"),
    ]:
        print(f"   {label:<22}{a:>14}{b:>14}")
    print(f"   {'years beating SPY':<22}{f'{beat} of {len(rows)}':>14}")

    tr = res.trades
    if len(tr):
        print(f"\n   {len(tr)} trades · {(tr.alpha > 0).mean()*100:.0f}% beat the market · "
              f"mean {tr.alpha.mean()*100:+.2f}pp vs market per trade · "
              f"median hold {tr.hold_days.median():.0f} sessions")
        print(f"   exits: " + ", ".join(f"{k} {v}" for k, v in tr.reason.value_counts().items()))
        print(f"   average stocks held {res.equity.n_positions.mean():.1f} "
              f"(max {int(res.equity.n_positions.max())}), "
              f"{(res.equity.stocks/eq).mean()*100:.0f}% of equity in stocks, rest in SPY")
        best = tr.nlargest(3, "alpha")[["symbol", "entry_date", "alpha"]]
        worst = tr.nsmallest(3, "alpha")[["symbol", "entry_date", "alpha"]]
        print("   best:  " + ", ".join(f"{r.symbol} {r.alpha*100:+.0f}pp ({r.entry_date:%b %Y})" for r in best.itertuples()))
        print("   worst: " + ", ".join(f"{r.symbol} {r.alpha*100:+.0f}pp ({r.entry_date:%b %Y})" for r in worst.itertuples()))


if __name__ == "__main__":
    main()
