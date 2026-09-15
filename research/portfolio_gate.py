# ═══════════════════════════════════════════════════════════════════════
#  Portfolio gate: choose the cutoff and the exit rule on net total return.
#
#  Runs the live engine (trader/) over the point-in-time history for a
#  grid of cutoffs and volatility trailing stops, $1,000 start, 10bps
#  round-trip on stocks, idle cash in SPY, identical start date for every
#  run.
#
#  PRE-REGISTERED SELECTION (2026-09-15, written before the first run):
#
#    Metric: CAGR of the net equity curve over the full window.
#    Eras:   2010-2013, 2014-2019, 2020-2026.
#
#    1. Cutoff (no stop). Baseline = top 10%. A tighter cutoff c qualifies
#       if its CAGR beats the baseline, EVERY cutoff between 10% and c also
#       beats the baseline (returns must rise with strictness, not spike at
#       one setting), and it does no worse than the baseline in at least
#       2 of 3 eras. Pick the qualifying cutoff with the highest CAGR;
#       otherwise keep 10%. (Event-level significance for every cutoff was
#       already shown in research/pead_trailing.py: all p <= 0.019.)
#
#    2. Stop, at the chosen cutoff. Reference = no stop. A multiple k
#       qualifies if its CAGR beats the reference, both grid neighbours of
#       k also beat it (robust, not a lucky parameter), and it does no
#       worse in at least 2 of 3 eras. Pick the qualifying k with the
#       highest CAGR; otherwise no stop.
#
#  Writes config/strategy.json (the live parameters + evidence) and
#  config/tripwire_band.csv (layer 7's alarm threshold).
#
#    python research/portfolio_gate.py
# ═══════════════════════════════════════════════════════════════════════

import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import ERAS, cagr, load_market, run, summarize  # noqa: E402
from trader.config import Config, save_strategy  # noqa: E402
from trader.engine import Engine  # noqa: E402
from trader.monitor import bootstrap_band, first_fire  # noqa: E402

CUTOFFS   = [0.90, 0.95, 0.97, 0.98]
STOPS     = [None, 3, 5, 8, 12]
ERA_NAMES = [e[0] for e in ERAS]
BAND_FILE = ROOT / "config" / "tripwire_band.csv"
REPORTS   = ROOT / "reports"


def no_worse_in_eras(cand, ref, need=2):
    return sum(cand[f"cagr_{e}"] >= ref[f"cagr_{e}"] for e in ERA_NAMES) >= need


def label(c, k):
    return f"top {round((1 - c) * 100):>2}%  stop {'—' if k is None else k:>2}"


def main():
    base = Config()
    print("Loading market …")
    market = load_market(base)
    start = Engine(base, market.cal, market.closes, market.events).default_start()
    print(f"Window: {market.cal[start].date()} → {market.cal[-1].date()}\n")

    print(f"   {'config':<18}{'CAGR':>8}{'SPY':>8}{'maxDD':>8}{'Sharpe':>8}"
          f"{'tr/yr':>7}{'alpha':>8}{'held':>6}  " + "  ".join(f"{e:>9}" for e in ERA_NAMES))
    S, R = {}, {}
    for c in CUTOFFS:
        for k in STOPS:
            cfg = replace(base, cutoff=c, stop_k=k)
            res = run(market, cfg, start=start)
            s = summarize(res, market, cfg)
            S[(c, k)], R[(c, k)] = s, res
            print(f"   {label(c, k):<18}{s['cagr']*100:>+7.2f}%{s['bench_cagr']*100:>+7.2f}%"
                  f"{s['max_drawdown']*100:>+7.1f}%{s['sharpe']:>8.2f}{s['trades_per_year']:>7.0f}"
                  f"{s['mean_alpha']*100:>+7.2f}%{s['avg_positions']:>6.1f}  "
                  + "  ".join(f"{s[f'cagr_{e}']*100:>+8.2f}%" for e in ERA_NAMES), flush=True)

    # ── Step 1: cutoff ────────────────────────────────────────────────
    ref = S[(0.90, None)]
    qualifying = []
    for i, c in enumerate(CUTOFFS[1:], start=1):
        chain = all(S[(cc, None)]["cagr"] > ref["cagr"] for cc in CUTOFFS[1:i + 1])
        if chain and no_worse_in_eras(S[(c, None)], ref):
            qualifying.append(c)
    cutoff = max(qualifying, key=lambda c: S[(c, None)]["cagr"]) if qualifying else 0.90
    print(f"\nStep 1 — qualifying tighter cutoffs: "
          f"{[f'top {round((1-c)*100)}%' for c in qualifying] or 'none'}"
          f"  →  cutoff = top {round((1 - cutoff) * 100)}%")

    # ── Step 2: stop ──────────────────────────────────────────────────
    ref2 = S[(cutoff, None)]
    ks = STOPS[1:]
    qual_k = []
    for j, k in enumerate(ks):
        nbrs = [ks[x] for x in (j - 1, j + 1) if 0 <= x < len(ks)]
        if (S[(cutoff, k)]["cagr"] > ref2["cagr"]
                and all(S[(cutoff, n)]["cagr"] > ref2["cagr"] for n in nbrs)
                and no_worse_in_eras(S[(cutoff, k)], ref2)):
            qual_k.append(k)
    stop_k = max(qual_k, key=lambda k: S[(cutoff, k)]["cagr"]) if qual_k else None
    print(f"Step 2 — qualifying stops at that cutoff: {qual_k or 'none'}  →  stop = {stop_k}")

    chosen = replace(base, cutoff=cutoff, stop_k=stop_k)
    s, res = S[(cutoff, stop_k)], R[(cutoff, stop_k)]

    print("\n── Chosen configuration ───────────────────────────────────────")
    for key in ["start", "end", "final_equity", "total_return", "cagr", "bench_cagr", "vol",
                "sharpe", "max_drawdown", "bench_max_drawdown", "avg_stock_exposure",
                "avg_positions", "max_positions", "trades", "trades_per_year", "mean_alpha",
                "hit_rate", "exit_reasons"]:
        v = s[key]
        print(f"   {key:<22}{v:,.4f}" if isinstance(v, float) else f"   {key:<22}{v}")

    eq = res.equity.equity
    spy = market.closes[base.benchmark].reindex(eq.index)
    yearly = pd.DataFrame({"strategy": eq.resample("YE").last().pct_change(),
                           "SPY": spy.resample("YE").last().pct_change()})
    yearly.iloc[0] = [eq.resample("YE").last().iloc[0] / eq.iloc[0] - 1,
                      spy.resample("YE").last().iloc[0] / spy.iloc[0] - 1]
    print("\n── Calendar-year returns ──────────────────────────────────────")
    for d, r in yearly.iterrows():
        print(f"   {d.year}  strategy {r.strategy*100:>+7.2f}%   SPY {r.SPY*100:>+7.2f}%"
              f"   diff {(r.strategy - r.SPY)*100:>+7.2f}pp")

    # ── Tripwire band ─────────────────────────────────────────────────
    band = bootstrap_band(res.trades)
    BAND_FILE.parent.mkdir(parents=True, exist_ok=True)
    band.to_csv(BAND_FILE, index=False)
    print(f"\n── Tripwire band ({len(res.trades):,} backtest trades, date-clustered bootstrap) ─")
    for n in [10, 25, 50, 100, 200]:
        r = band.iloc[n - 1]
        print(f"   after {n:>3} trades: fire if mean alpha < {r.p05*100:+.2f}%   (median {r.p50*100:+.2f}%)")
    tr = res.trades.sort_values("exit_date")
    print("   If live trading had started on …")
    for since in [s["start"], "2014-01-01", "2020-01-01"]:
        sub = tr[tr.entry_date >= pd.Timestamp(since)]
        n = first_fire(sub.alpha.to_numpy(), band)
        when = "never fired" if n is None else f"fires at trade {n} ({sub.exit_date.iloc[n-1].date()})"
        print(f"     {since}: {when}")

    REPORTS.mkdir(exist_ok=True)
    res.equity.to_csv(REPORTS / "backtest_equity.csv")
    res.trades.to_csv(REPORTS / "backtest_trades.csv", index=False)

    grid = {f"cutoff={c},stop={k}": {m: S[(c, k)][m] for m in
            ["cagr", "max_drawdown", "sharpe", "trades_per_year", "mean_alpha"]
            + [f"cagr_{e}" for e in ERA_NAMES]} for c in CUTOFFS for k in STOPS}
    save_strategy(chosen, {
        "selected_on": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "script": "research/portfolio_gate.py",
        "rule": "max CAGR subject to monotone-vs-baseline, robust neighbours, >= baseline in 2 of 3 eras",
        "baseline": asdict(base),
        "chosen_summary": {k: v for k, v in s.items()},
        "grid": grid,
    })
    print(f"\nWrote config/strategy.json and {BAND_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
