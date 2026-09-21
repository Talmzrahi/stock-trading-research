# ═══════════════════════════════════════════════════════════════════════
#  What a long-short version of the text signal would have earned.
#  DEVELOPMENT DATA ONLY, exploratory — this is not a gate test.
#
#  Why it exists: on the full development set the text score's
#  top-minus-bottom fifth is +1.12pp over 60 sessions (p<0.005), but by
#  decile the money is on the SHORT side — the worst-read releases fall,
#  the best-read ones barely rise, and the top 5% a long-only rule would
#  buy earns −0.46pp. This project is long-only, so that half is
#  unreachable. Before anyone builds shorting, this measures what the
#  unreachable half is actually worth, net of what shorting costs.
#
#  The book: each event enters at the next close after its reaction is
#  complete and is held HOLD sessions. Long the top decile, short the
#  bottom decile, equal-weighted within each side and dollar-neutral.
#
#  Costs charged: COST per side per trade (entry and exit), plus BORROW a
#  year on the short book. Short borrow on S&P 500 names is usually cheap,
#  but it is not free and it is not optional.
#
#    python research/v3_longshort_probe.py
# ═══════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402
from inference_check import market_alpha  # noqa: E402
from v3_readers_eval import out_of_fold  # noqa: E402
from v3_signal_probe import BASE, load, spread  # noqa: E402

HOLD    = 60
DECILES = 10
COST    = 10 / 1e4        # per side, per trade
BORROW  = 0.01            # a year, on the short book
MIN_DAYS = 5


def book(picks, side, returns, col, n_days):
    """Daily equal-weighted return of one side's open positions."""
    total, count = np.zeros(n_days), np.zeros(n_days)
    for e, sym in zip(picks.gap_entry_idx.to_numpy(), picks.symbol):
        c = col.get(sym)
        if c is None:
            continue
        window = slice(e + 1, e + HOLD + 1)
        total[window] += returns[window, c]
        count[window] += 1
        total[e + 1] -= COST                       # entry
        total[min(e + HOLD, n_days - 1)] -= COST   # exit
    open_ = count > 0
    daily = np.zeros(n_days)
    daily[open_] = side * total[open_] / count[open_]
    return daily, count


def main():
    cfg = load_config()
    d = load(cfg)
    text = [c for c in d.columns if any(c.startswith(p) for p in
            ("distilroberta_fin", "finbert", "minilm", "spread", "lm_tone"))]
    d["score"] = out_of_fold(d, d[BASE + text].to_numpy(float), d.reaction.to_numpy())
    d = d[d.fwd60.notna()].copy()
    d["dec"] = pd.qcut(d.score.rank(method="first"), DECILES, labels=False) + 1

    market = load_market(cfg)
    closes = market.closes.ffill()
    col = {s: i for i, s in enumerate(closes.columns)}
    returns = closes.pct_change().fillna(0.0).to_numpy()
    n_days = len(market.cal)

    print(f"{len(d):,} development releases, {d.entry_date.min()} → {d.entry_date.max()}")
    print(f"\n── 60-session excess return vs SPY, by text-score decile ──")
    g = d.groupby("dec").fwd60.agg(["size", "mean"])
    for dec, r in g.iterrows():
        print(f"   decile {dec:>2}: n={int(r['size']):>5,}  {r['mean'] * 100:+6.2f}pp")
    print(f"   all releases: {d.fwd60.mean() * 100:+.2f}pp  "
          "(equal-weighted stocks lagged the cap-weighted index over this period)")

    print("\n── Per trade, long top decile vs short bottom decile ──")
    for hi, lo in [(10, 1), (9, 1), (10, 2)]:
        sel = d[d.dec.isin([hi, lo])]
        diff, p = spread(sel.fwd60, (sel.dec == hi).astype(float), sel.quarter, n=2)
        print(f"   decile {hi} minus decile {lo}: {diff * 100:+.2f}pp per trade "
              f"(p={p:.3f}, quarter-clustered, n={len(sel):,})")

    print("\n── As a portfolio: dollar-neutral, held 60 sessions, net of costs ──")
    long_d, short_d = d[d.dec == DECILES], d[d.dec == 1]
    lo_daily, lo_n = book(long_d, +1, returns, col, n_days)
    sh_daily, sh_n = book(short_d, -1, returns, col, n_days)
    borrow = np.where(sh_n > 0, BORROW / 252, 0.0)
    daily = (lo_daily + sh_daily) / 2 - borrow / 2
    live = (lo_n > 0) & (sh_n > 0)
    idx = pd.DatetimeIndex(market.cal)[live]
    ser = pd.Series(daily[live], index=idx)
    monthly = pd.DataFrame({"port": (1 + ser).groupby(idx.to_period("M")).prod() - 1})
    monthly["bench"] = ((1 + pd.Series(returns[live, col[cfg.benchmark]], index=idx))
                        .groupby(idx.to_period("M")).prod() - 1)
    monthly = monthly[ser.groupby(idx.to_period("M")).size() >= MIN_DAYS]
    a, p, beta, months = market_alpha(monthly)
    ann = monthly.port.mean() * 12
    vol = monthly.port.std() * np.sqrt(12)
    t = stats.ttest_1samp(monthly.port, 0)
    print(f"   {months} months live · average {int(lo_n[live].mean())} long / "
          f"{int(sh_n[live].mean())} short positions")
    print(f"   return {ann * 100:+.2f}%/yr · volatility {vol * 100:.1f}% · "
          f"Sharpe {ann / vol if vol else float('nan'):.2f} · t={t.statistic:.2f} (p={t.pvalue:.3f})")
    print(f"   alpha vs SPY {a * 100:+.2f}%/yr (p={p:.3f}), beta {beta:+.2f}")

    print("\n── By year (is it steady or one good spell?) ──")
    by_year = monthly.port.groupby(monthly.index.year).agg(["size", lambda s: (1 + s).prod() - 1])
    by_year.columns = ["months", "return"]
    line = "   " + "  ".join(f"{y}:{r['return'] * 100:+.0f}%" for y, r in by_year.iterrows())
    print(line)
    print(f"   positive years: {(by_year['return'] > 0).sum()} of {len(by_year)}")

    print("\nDevelopment data, many looks, no shorting in this system today. "
          "Treat as a reason to test, not a result.")


if __name__ == "__main__":
    main()
