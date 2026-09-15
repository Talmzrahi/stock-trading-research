# ═══════════════════════════════════════════════════════════════════════
#  Stress test of the gate-selected strategy, before real money.
#
#  The live configuration (top 2%, 8-sd stop) was the best of 20 cells on
#  the full 2011-2026 history. How much of that is hindsight?
#
#  PRE-REGISTERED (2026-09-15, written before the first run):
#
#  1. WALK-FORWARD. Run all 20 configurations once over the full window
#     (same engine, $1,000, 10bps round trip, same start as the gate). For
#     each test year Y = 2014 … 2026, apply research/selection.py's rule to
#     the equity curves from the window start through Dec 31 of Y-1 only.
#     Named eras don't exist inside short training windows, so the
#     "no worse in 2 of 3 sub-periods" check uses the training window split
#     into three equal thirds. Year Y is traded with the pick: the
#     walk-forward curve takes that configuration's daily returns for Y.
#     (Switching assumes the new book is adopted on Jan 1 at no cost — this
#     understates turnover cost slightly.)
#
#     Reported over 2014-01-01 → end, against: the hindsight pick, the
#     top-10%/no-stop baseline, the average of all 20 configurations
#     (selection with no skill), and SPY.
#
#     Headline: survival share = (WF CAGR - SPY) / (hindsight CAGR - SPY).
#     PASS iff walk-forward CAGR > SPY CAGR. A FAIL is a recommendation not
#     to put real money in; it does not change config/strategy.json.
#
#  2. COSTS. The live configuration at 10 / 20 / 30 / 50 bps round trip.
#
#  Consistency check: the shared rule applied to the full window with the
#  gate's named eras must reproduce config/strategy.json.
#
#    python research/stress_test.py
# ═══════════════════════════════════════════════════════════════════════

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from selection import CUTOFFS, STOPS, select  # noqa: E402
from trader.backtest import ERAS, cagr, load_market, run, summarize  # noqa: E402
from trader.config import Config, load_config  # noqa: E402
from trader.engine import Engine  # noqa: E402

TEST_START = 2014
COST_SIDES = [5, 10, 15, 25]      # bps per side → 10/20/30/50 round trip
REPORTS    = ROOT / "reports"


def thirds(eq):
    n = len(eq)
    cuts = [0, n // 3, 2 * n // 3, n - 1]
    return [eq.iloc[cuts[i]:cuts[i + 1] + 1] for i in range(3)]


def growth_curve(r):
    r = r.dropna()
    base = pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)])
    return pd.concat([base, (1 + r).cumprod()])


def max_dd(eq):
    return float((eq / eq.cummax() - 1).min())


def label(c, k):
    return f"top {round((1 - c) * 100)}% / stop {'—' if k is None else f'{k:g}'}"


def main(cutoffs=None):
    cutoffs = cutoffs or CUTOFFS
    base, live = Config(), load_config()
    live_key = (live.cutoff, live.stop_k)
    print("Loading market …", flush=True)
    market = load_market(base)
    start = Engine(base, market.cal, market.closes, market.events).default_start()

    curves = {}
    for c in cutoffs:
        for k in STOPS:
            curves[(c, k)] = run(market, replace(base, cutoff=c, stop_k=k), start=start).equity.equity
            print(f"   ran {label(c, k)}", flush=True)
    rets = {key: eq.pct_change() for key, eq in curves.items()}
    idx = curves[(cutoffs[0], None)].index
    spy_r = market.closes[base.benchmark].reindex(idx).pct_change()

    full = {key: {"cagr": cagr(eq), "eras": [cagr(eq.loc[a:b]) for _, a, b in ERAS]}
            for key, eq in curves.items()}
    fc, fk, _, _ = select(full, cutoffs=cutoffs)
    ok = (fc, fk) == live_key
    print(f"\nConsistency: shared rule on the full window picks {label(fc, fk)}; "
          f"config/strategy.json has {label(*live_key)} — {'match' if ok else 'MISMATCH'}")

    # ── 1. Walk-forward ───────────────────────────────────────────────
    rows, wf = [], []
    for year in range(TEST_START, idx[-1].year + 1):
        train = {}
        for key, eq in curves.items():
            tr = eq.loc[:f"{year - 1}-12-31"]
            train[key] = {"cagr": cagr(tr), "eras": [cagr(s) for s in thirds(tr)]}
        c, k, _, _ = select(train, cutoffs=cutoffs)
        yr = slice(f"{year}-01-01", f"{year}-12-31")
        seg = rets[(c, k)].loc[yr]
        wf.append(seg)
        comp = lambda r: float((1 + r.loc[yr].dropna()).prod() - 1)
        rows.append(dict(year=year, pick=label(c, k), walk_forward=comp(rets[(c, k)]),
                         hindsight=comp(rets[live_key]), baseline=comp(rets[(cutoffs[0], None)]),
                         spy=comp(spy_r)))
    picks = pd.DataFrame(rows)

    test = slice(f"{TEST_START}-01-01", None)
    series = {
        "walk-forward": pd.concat(wf),
        f"hindsight ({label(*live_key)})": rets[live_key].loc[test],
        f"baseline ({label(cutoffs[0], None)})": rets[(cutoffs[0], None)].loc[test],
        "grid averaged (no skill)": pd.DataFrame(rets).mean(axis=1).loc[test],
        "SPY": spy_r.loc[test],
    }
    stats = {}
    for name, r in series.items():
        g = growth_curve(r)
        stats[name] = (cagr(g), max_dd(g), float(g.iloc[-1] - 1))

    print(f"\n── 1. Walk-forward, {TEST_START} → {idx[-1].date()} ─────────────────────────")
    print(f"   {'year':<6}{'picked (trained on prior years only)':<30}{'WF':>9}{'hindsight':>11}"
          f"{'baseline':>10}{'SPY':>9}")
    for r in picks.itertuples():
        print(f"   {r.year:<6}{r.pick:<30}{r.walk_forward*100:>+8.2f}%{r.hindsight*100:>+10.2f}%"
              f"{r.baseline*100:>+9.2f}%{r.spy*100:>+8.2f}%")
    same = (picks.pick == label(*live_key)).sum()
    print(f"   Walk-forward picked the live configuration in {same} of {len(picks)} years.")

    print(f"\n   {'':<34}{'CAGR':>9}{'max DD':>9}{'total':>10}")
    for name, (g, dd, tot) in stats.items():
        print(f"   {name:<34}{g*100:>+8.2f}%{dd*100:>+8.1f}%{tot*100:>+9.0f}%")

    wf_c, spy_c = stats["walk-forward"][0], stats["SPY"][0]
    hind_c = stats[f"hindsight ({label(*live_key)})"][0]
    survival = (wf_c - spy_c) / (hind_c - spy_c) if hind_c != spy_c else float("nan")
    print(f"\n   Survival share of the excess return: {survival*100:.0f}%")
    print(f"══ PRE-REGISTERED VERDICT: {'PASS' if wf_c > spy_c else 'FAIL'} ══  "
          f"walk-forward {wf_c*100:+.2f}% vs SPY {spy_c*100:+.2f}% CAGR")

    # ── 2. Costs ──────────────────────────────────────────────────────
    print(f"\n── 2. Cost sensitivity, {label(*live_key)}, full window ─────────────────")
    for side in COST_SIDES:
        cfg = replace(live, cost_bps_side=side)
        s = summarize(run(market, cfg, start=start), market, cfg)
        print(f"   {2*side:>3} bps round trip   CAGR {s['cagr']*100:+.2f}%   "
              f"mean alpha/trade {s['mean_alpha']*100:+.2f}%   max DD {s['max_drawdown']*100:+.1f}%",
              flush=True)

    REPORTS.mkdir(exist_ok=True)
    picks.to_csv(REPORTS / "stress_walk_forward.csv", index=False)


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or None)
