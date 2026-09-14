# ═══════════════════════════════════════════════════════════════════════
#  PEAD, specified properly.
#
#  backtest.py measured drift with a binary flag (surprise > 2%) over at
#  most 10 days and found +0.124pp at 5d. Two problems with that:
#
#    1. Binarising a continuous signal throws away magnitude — a 2.1%
#       surprise and a 40% surprise counted the same, as did a small miss
#       and a disaster. Dichotomisation costs power and understates size.
#    2. surprise_pct divides by |estimate|, so a firm expected to earn
#       $0.01 that earns $0.03 books a +200% "surprise". Near-zero
#       estimates dominate the tail for no economic reason.
#    3. The drift in the literature runs 60+ days; 10 days sees a slice.
#
#  This re-specifies with price-scaled SUE (Livnat & Mendenhall style),
#  decile sorts, and horizons to 60 days. Deciles are cut WITHIN each
#  calendar quarter so breakpoints never use future cross-sections.
#
#    python research/pead.py
# ═══════════════════════════════════════════════════════════════════════

import importlib.util
import sqlite3
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parent.parent
DB_FILE   = ROOT / "data" / "research.db"
BENCHMARK = "SPY"
HORIZONS  = [1, 3, 5, 10, 20, 40, 60]
COST_BPS  = 10.0
N_DECILES = 10

_spec = importlib.util.spec_from_file_location("bt", Path(__file__).resolve().parent / "backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def build(wide, earn):
    """Same conservative entry timing as backtest.py, but carrying a
    continuous SUE and horizons long enough to see the whole drift."""
    cal = wide.index
    ann_et = earn["announced_at"].dt.tz_convert("America/New_York")
    ann_day = pd.to_datetime(ann_et.dt.date)
    before_open = ann_et.dt.time < dtime(9, 30)

    pos = cal.searchsorted(ann_day, side="left")
    ev = earn.copy()
    ev["entry_idx"] = np.where(before_open, pos, cal.searchsorted(ann_day, side="right"))
    ev["signal_idx"] = pos - 1
    ev = ev[(ev.signal_idx >= 0) & (ev.entry_idx < len(cal) - max(HORIZONS))]

    col_of = {s: i for i, s in enumerate(wide.columns)}
    ev = ev[ev.symbol.isin(col_of)]
    rows = ev.entry_idx.to_numpy()
    sig = ev.signal_idx.to_numpy()
    cols = ev.symbol.map(col_of).to_numpy()
    px = wide.to_numpy()
    spy = wide[BENCHMARK].to_numpy()

    ev["entry_date"] = cal[rows]
    ev["price"] = px[sig, cols]
    for h in HORIZONS:
        ev[f"ret{h}"] = (px[rows + h, cols] / px[rows, cols] - 1.0) - (spy[rows + h] / spy[rows] - 1.0)

    # Price-scaled surprise: robust to near-zero estimates, unlike surprise_pct.
    ev["sue"] = (ev.eps_actual - ev.eps_estimate) / ev.price
    ev = ev[ev.price > 0].replace([np.inf, -np.inf], np.nan)
    ev = ev.dropna(subset=["sue"] + [f"ret{h}" for h in HORIZONS])

    # Deciles cut within each quarter: breakpoints use only that quarter's
    # cross-section, never the full-sample distribution.
    ev["q"] = ev.entry_date.dt.to_period("Q")
    ev["decile"] = (ev.groupby("q")["sue"]
                      .transform(lambda s: pd.qcut(s.rank(method="first"), N_DECILES,
                                                   labels=False) + 1
                                 if s.notna().sum() >= N_DECILES else np.nan))
    return ev.dropna(subset=["decile"])


def date_clustered_diff(ev, col, hi_mask, lo_mask):
    """Collapse to one observation per (date, group) before testing, since
    same-day events share a common market component."""
    hi = ev[hi_mask].groupby("entry_date")[col].mean()
    lo = ev[lo_mask].groupby("entry_date")[col].mean()
    if len(hi) < 5 or len(lo) < 5:
        return np.nan, np.nan, len(hi), len(lo)
    t, p = stats.ttest_ind(hi, lo, equal_var=False)
    return hi.mean() - lo.mean(), p, len(hi), len(lo)


def main():
    conn = sqlite3.connect(DB_FILE)
    wide, _vix, earn = bt.load(conn)
    conn.close()

    ev = build(wide, earn)
    print(f"Events: {len(ev):,}  ({ev.entry_date.min().date()} → {ev.entry_date.max().date()})")
    print(f"Symbols: {ev.symbol.nunique()}   quarters: {ev.q.nunique()}")
    print(f"SUE (price-scaled) percentiles: "
          f"p10={ev.sue.quantile(.10):+.5f}  p50={ev.sue.quantile(.50):+.5f}  "
          f"p90={ev.sue.quantile(.90):+.5f}")

    print("\n── Mean market-adjusted return by SUE decile (%) ──────────────")
    print("   monotonic across deciles is the signature of a real effect")
    hdr = "   dec" + "".join(f"{f'{h}d':>9}" for h in HORIZONS) + f"{'n':>8}"
    print(hdr)
    tab = ev.groupby("decile")
    for d, g in tab:
        line = f"   {int(d):>3}" + "".join(f"{g[f'ret{h}'].mean()*100:>+9.3f}" for h in HORIZONS)
        print(line + f"{len(g):>8,}")

    print("\n── Top decile minus bottom decile ─────────────────────────────")
    print(f"   {'h':>4} {'spread':>9} {'p':>8} {'net of cost':>12} {'dates hi/lo':>14}")
    top, bot = ev.decile == N_DECILES, ev.decile == 1
    spreads = {}
    for h in HORIZONS:
        d, p, nh, nl = date_clustered_diff(ev, f"ret{h}", top, bot)
        spreads[h] = d
        net = d - 2 * COST_BPS / 1e4          # long top + short bottom = two round trips
        print(f"   {h:>4} {d*100:>+8.3f}pp {p:>8.3f} {net*100:>+11.3f}pp {f'{nh}/{nl}':>14}")

    print("\n── Long-only top decile (vs market) ───────────────────────────")
    print(f"   {'h':>4} {'gross':>9} {'net of cost':>12} {'ann. (4 ev/yr)':>16}")
    for h in HORIZONS:
        g = ev.loc[top, f"ret{h}"].mean()
        net = g - COST_BPS / 1e4
        print(f"   {h:>4} {g*100:>+8.3f}pp {net*100:>+11.3f}pp {net*4*100:>+15.2f}%")

    print("\n── What the old binary specification cost ─────────────────────")
    old = ev[ev.surprise_pct > 2.0]
    oth = ev[~(ev.surprise_pct > 2.0)]
    for h in [5, 60]:
        d, p, _, _ = date_clustered_diff(ev, f"ret{h}", ev.surprise_pct > 2.0,
                                         ~(ev.surprise_pct > 2.0))
        print(f"   binary beat>2%   h={h:<3} {d*100:>+7.3f}pp (p={p:.3f})   "
              f"decile spread {spreads[h]*100:>+7.3f}pp   "
              f"ratio {abs(spreads[h]/d) if d and not np.isnan(d) else float('nan'):.1f}x")

    print("\n── Monotonicity check ─────────────────────────────────────────")
    for h in [5, 20, 60]:
        means = ev.groupby("decile")[f"ret{h}"].mean()
        rho, p = stats.spearmanr(means.index.astype(float), means.values)
        print(f"   h={h:<3} Spearman rho(decile, mean return) = {rho:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
