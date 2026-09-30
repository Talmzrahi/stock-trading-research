# ======================================================================
#  The numbers behind docs/charts/ 1, 3 and 4 (2026-09-30).
#
#  Uses the project's own definitions, read-only: the volatility-targeted
#  book from voltarget_synthesis.pair (12% target, 1.5x cap, 21-day vol,
#  set at month-end, margin charged), the bear-market flags from
#  bear_regimes.flags, and noise_search's seeded coin-flip rules, replayed
#  in the same order so the best-p curve matches the script exactly.
#
#    chart 1  SPY growth of $1: buy-and-hold, the model only in hindsight
#             bear markets, the model only below the 200-day average
#    chart 3  drawdowns: buy-and-hold vs the model always on
#    chart 4  best p-value among the first n random rules, n = 1..1,000
#
#  Charts 2 and 5 are recorded results (bear_regimes.py output; the
#  PEAD backtests in ROADMAP.md and PROJECT_STATE.md section 3).
#
#    .venv\Scripts\python.exe research\chart_data.py   -> data/chart_data.json
# ======================================================================
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from mechanics import performance  # noqa: E402
from voltarget_synthesis import pair, px  # noqa: E402
from bear_regimes import flags  # noqa: E402
import noise_search as ns  # noqa: E402

OUT = ROOT / "data" / "chart_data.json"


def spy_books():
    bh, tg, rf = pair(px("SPY"), "1993", None, 0.0)
    eq, hind, ma, dd20, eps = flags(bh)
    books = {
        "sp500": bh,
        "hindsight": pd.Series(np.where(hind, tg, bh), index=bh.index),
        "realtime": pd.Series(np.where(ma, tg, bh), index=bh.index),
        "always": tg,
    }
    return books, rf, eps


def noise_curve():
    with redirect_stdout(io.StringIO()):
        ns.experiment_a()                  # consumes the RNG exactly as the script does
    s = px("GSPC").loc["1928":]
    eq = (s.pct_change().fillna(0.0) + 0.04 / 252).to_numpy()
    idx = s.index
    rf = ns.RF.reindex(idx).ffill().fillna(0.0).to_numpy()
    bid = ns.blocks(idx)
    nb = bid.max() - bid.min() + 1
    mkt_m, rf_m = ns.monthly(idx, eq, rf)

    def run(on_block):
        h = on_block[bid - bid.min()].astype(float)
        return ns.t2(ns.monthly(idx, h * eq + (1 - h) * rf)[0], mkt_m, rf_m)

    res = [run(ns.RNG.random(nb) < 0.5) for _ in range(1000)]
    a = np.array([x[0] for x in res])
    p = np.array([x[1] for x in res])
    best = np.minimum.accumulate(np.where(a > 0, p, 1.0))
    return best, int(((a > 0) & (p < 0.05)).sum())


def main():
    books, rf, eps = spy_books()
    out = {"stats": {}, "bear_episodes": [[str(a.date()), str(b.date())] for a, b in eps]}
    print("SPY, buy-and-hold vs the volatility-targeted model")
    for k, r in books.items():
        p = performance(r, rf)
        out["stats"][k] = dict(cagr=p["ret"], max_dd=p["max_dd"], final=float((1 + r).prod()))
        print(f"   {k:<10} {p['ret']*100:6.2f}%/yr   worst {p['max_dd']*100:6.1f}%   "
              f"$1 -> ${(1 + r).prod():.2f}")
    m = books["sp500"].index.to_period("M")
    out["growth"] = {k: {str(q): float(v) for q, v in (1 + books[k]).cumprod().groupby(m).last().items()}
                     for k in ("sp500", "hindsight", "realtime")}
    out["drawdown"] = {}
    for k in ("sp500", "always"):
        e = (1 + books[k]).cumprod()
        out["drawdown"][k] = {str(q): float(v) for q, v in (e / e.cummax() - 1).groupby(m).min().items()}

    best, passed = noise_curve()
    out["noise_best_p"] = best.tolist()
    print(f"\nCoin-flip rules: {passed} of 1,000 pass; best p after n rules:")
    for n in (1, 10, 34, 100, 323, 1000):
        print(f"   n = {n:>5}   p = {best[n - 1]:.4f}")

    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nwritten: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
