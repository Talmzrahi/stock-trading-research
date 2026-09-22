# ═══════════════════════════════════════════════════════════════════════
#  The PEAD gate on the full S&P 1500 — a power exercise, not a new test.
#
#  Every result in this project has failed from lack of power rather than
#  from being clearly wrong. The S&P 500 and S&P 400/600 samples are both
#  already built and both already examined; pooling them roughly triples
#  the sample and asks what the gate says when it can actually see.
#
#  NOT A CLEAN TEST. Both halves have been looked at (Phase 2c ran the
#  pre-registered cutoff test on the mid/small half, Phase 2e re-checked
#  it). This sharpens the ESTIMATE; it does not create fresh evidence.
#
#  One-shot data: untouched. This reads oos_midsmall.db (prices and
#  earnings, development-grade since Phase 2c). It never opens
#  data/edgar_midsmall.db, which is the one-shot text exam.
#
#  Each event's excess return is against its own matched benchmark — SPY
#  for large caps, IJH/IJR for mid and small — and net of its own sample's
#  cost assumption (10bps vs 20bps a side). Pooling excess returns across
#  differently-benchmarked events is legitimate; pooling raw returns
#  would not be.
#
#    .venv\Scripts\python.exe research\sp1500_gate.py
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
from inference_check import (HOLD, MIN_DAYS, NW_LAGS, daily_returns,  # noqa: E402
                             market_alpha, midsmall_sample, sp500_sample)

FORBIDDEN = ROOT / "data" / "edgar_midsmall.db"       # the one-shot exam


def net_events(top, col, cost):
    """One row per event: its entry date and its excess return net of cost."""
    return pd.DataFrame({"entry_date": pd.to_datetime(top.entry_date.to_numpy()),
                         "net": top[col].to_numpy(float) - cost})


def quarter_est(ev):
    """T1: mean of per-date means, standard errors clustered by quarter."""
    means = ev.groupby("entry_date").net.mean()
    q = pd.PeriodIndex(means.index, freq="Q").astype(str)
    fit = sm.OLS(means.to_numpy(), np.ones((len(means), 1))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(q)[0]}, use_t=True)
    return (float(fit.params[0]), float(fit.bse[0]), float(fit.pvalues[0]),
            len(means), q.nunique())


def daily_legs(top, closes, cost):
    """Per-day sums of position returns, matched-benchmark returns and the
    open-position count — the pieces calendar_time averages."""
    rets, dates = daily_returns(closes), closes.index
    port, bench, n = (np.zeros(len(dates)) for _ in range(3))
    for e, c, b in zip(top.entry_idx.to_numpy(), top.stock_col.to_numpy(),
                       top.bench_col.to_numpy()):
        w = slice(e + 1, e + HOLD + 1)
        port[w] += rets[w, c]
        bench[w] += rets[w, b]
        n[w] += 1
        port[e + 1] -= cost
    return pd.DataFrame({"port": port, "bench": bench, "n": n}, index=dates)


def monthly_from(legs):
    """Equal-weighted across whatever is open, compounded to months."""
    legs = legs[legs.n > 0]
    daily = pd.DataFrame({"port": legs.port / legs.n, "bench": legs.bench / legs.n},
                         index=legs.index)
    m = daily.index.to_period("M")
    monthly = (1 + daily).groupby(m).prod() - 1
    return monthly[daily.groupby(m).size() >= MIN_DAYS]


def report(label, ev, legs):
    m, se, p, dates, quarters = quarter_est(ev)
    monthly = monthly_from(legs)
    a, p2, beta, months = market_alpha(monthly)
    # the standard error on the alpha, which is the whole point of pooling
    t2 = stats.t.ppf(1 - p2 / 2, max(months - 2, 1))
    a_se = abs(a) / t2 if t2 > 0 else float("nan")
    print(f"\n── {label} ──")
    print(f"   {len(ev):,} events, {dates:,} entry dates, {quarters} quarters, {months} months")
    print(f"   T1  {m*100:+.3f}pp  SE {se*100:.3f}pp  p={p:.4f}")
    print(f"   T2  alpha {a*100:+.2f}%/yr  SE {a_se*100:.2f}%/yr  p={p2:.4f}  beta {beta:.2f}")
    print(f"   →  {'CLEARS' if (m > 0 and p < 0.05 and a > 0 and p2 < 0.05) else 'does NOT clear'}"
          f" both gate tests")
    return se, a_se


def main():
    if FORBIDDEN.exists():
        print(f"one-shot exam set present at {FORBIDDEN.name} — this script never opens it\n")

    big_top, big_col, big_cost, big_closes = sp500_sample()
    mid_top, mid_col, mid_cost, mid_closes = midsmall_sample()

    big_ev, mid_ev = net_events(big_top, big_col, big_cost), net_events(mid_top, mid_col, mid_cost)
    big_legs, mid_legs = daily_legs(big_top, big_closes, big_cost), daily_legs(mid_top, mid_closes, mid_cost)

    print(f"S&P 500     : {len(big_ev):,} events, {big_ev.entry_date.min():%Y-%m} → "
          f"{big_ev.entry_date.max():%Y-%m}, vs SPY, {big_cost*1e4:.0f}bps")
    print(f"S&P 400/600 : {len(mid_ev):,} events, {mid_ev.entry_date.min():%Y-%m} → "
          f"{mid_ev.entry_date.max():%Y-%m}, vs IJH/IJR, {mid_cost*1e4:.0f}bps")

    se_b, a_b = report("S&P 500 alone", big_ev, big_legs)
    se_m, a_m = report("S&P 400/600 alone", mid_ev, mid_legs)

    pooled_ev = pd.concat([big_ev, mid_ev], ignore_index=True)
    pooled_legs = big_legs.add(mid_legs, fill_value=0.0)
    se_p, a_p = report("S&P 1500 pooled", pooled_ev, pooled_legs)

    print("\n── What the extra data bought ──")
    print(f"   T1 standard error   S&P 500 {se_b*100:.3f}pp → pooled {se_p*100:.3f}pp "
          f"({se_b/se_p:.2f}x tighter)")
    print(f"   T2 alpha SE         S&P 500 {a_b*100:.2f}%/yr → pooled {a_p*100:.2f}%/yr "
          f"({a_b/a_p:.2f}x tighter)")
    print("\n   Both halves were already examined, so this is a sharper estimate,")
    print("   not fresh evidence. The one-shot text exam remains unspent.")


if __name__ == "__main__":
    main()
