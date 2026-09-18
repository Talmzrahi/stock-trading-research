# ═══════════════════════════════════════════════════════════════════════
#  Overlap-robust re-check of the PEAD evidence.
#
#  Rules: research/prereg_inference.md (committed first, fbddc02). Only the
#  statistics change — samples, cutoffs, costs and return definitions are
#  the original tests', and step T0 proves it by reproducing their numbers.
#
#    T0  the original date-clustered result, which must replicate
#    T1  the same estimate, standard error clustered by calendar quarter
#    T2  calendar-time portfolio: monthly alpha against the matched ETF
#    T3  the live strategy's monthly alpha against SPY (S&P 500 only)
#
#    python research/inference_check.py
# ═══════════════════════════════════════════════════════════════════════

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import load_market, run  # noqa: E402
from trader.config import load_config  # noqa: E402
from trader.data import load_closes, load_earnings, load_universe  # noqa: E402
from trader.events import build_events, point_in_time  # noqa: E402
from trader.market_calendar import trading_days  # noqa: E402
from trader.signals.sue import SueSignal, trailing_percentile  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOLD      = 60
TOP       = 0.95
NW_LAGS   = 3
MIN_DAYS  = 5          # a month needs this many days with an open position
ALPHA     = 0.05


# ── the statistics ─────────────────────────────────────────────────────

def date_means(g, col, cost):
    """The unit every original gate used: one equal-weighted mean per entry date."""
    return (g[col] - cost).groupby(g.entry_date).mean()


def quarter_clustered(means):
    """T1: the mean of per-date means — the original estimate, unchanged —
    with its standard error clustered by calendar quarter of entry (CR1,
    t on G-1 df). Returns (estimate, p, number of quarters)."""
    quarters = pd.factorize(pd.PeriodIndex(means.index, freq="Q"))[0]
    fit = sm.OLS(means.to_numpy(), np.ones(len(means))).fit(
        cov_type="cluster", cov_kwds={"groups": quarters}, use_t=True)
    return float(fit.params[0]), float(fit.pvalues[0]), int(quarters.max() + 1)


def calendar_time(entry_idx, stock_col, bench_col, returns, dates, cost):
    """T2: each day, an equal-weighted portfolio of every event inside its
    holding window (sessions entry+1 .. entry+HOLD), and the same positions'
    benchmark. Cost is taken on each position's first day. Days with nothing
    open are left out; the rest compound to calendar months.

    `returns` is a (days x symbols) array of daily returns from forward-filled
    closes, so a position whose prices stop is held flat at its last close.
    """
    port, bench, n = (np.zeros(len(dates)) for _ in range(3))
    for e, c, b in zip(entry_idx, stock_col, bench_col):
        window = slice(e + 1, e + HOLD + 1)
        port[window] += returns[window, c]
        bench[window] += returns[window, b]
        n[window] += 1
        port[e + 1] -= cost
    open_ = n > 0
    daily = pd.DataFrame({"port": port[open_] / n[open_], "bench": bench[open_] / n[open_]},
                         index=pd.DatetimeIndex(dates)[open_])
    month = daily.index.to_period("M")
    monthly = (1 + daily).groupby(month).prod() - 1
    monthly = monthly[daily.groupby(month).size() >= MIN_DAYS]
    return monthly


def market_alpha(monthly, y="port", x="bench"):
    """Monthly y on monthly x, Newey-West standard errors. Returns
    (annualised alpha, p, beta, months)."""
    fit = sm.OLS(monthly[y].to_numpy(), sm.add_constant(monthly[x].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAGS}, use_t=True)
    return float(fit.params[0] * 12), float(fit.pvalues[0]), float(fit.params[1]), len(monthly)


def daily_returns(closes):
    return closes.ffill().pct_change().fillna(0.0).to_numpy()


# ── the samples, exactly as originally tested ──────────────────────────

def sp500_sample():
    """research/pead_trailing.py, repaired universe, top 5%."""
    pead, pt = _load("pead"), _load("pead_trailing")
    conn = sqlite3.connect(pead.DB_FILE)
    wide, _vix, earn = pead.bt.load(conn)
    universe = load_universe(conn)
    conn.close()
    ev = pead.build(wide, earn)
    ev = ev[ev.entry_date >= universe.as_of.min()].copy()
    ev["firm"], ev["pit"] = point_in_time(ev.symbol, ev.entry_date, universe)
    ev["ann_day"] = ev.announced_at.dt.tz_convert("America/New_York").dt.date
    ev = ev.sort_values(["firm", "ann_day", "symbol"]).drop_duplicates(["firm", "ann_day"])
    ev = ev[ev.pit].copy()
    ev["pctl"] = trailing_percentile(ev.entry_date, ev.sue)
    t = ev[(ev.entry_date >= pt.TEST_START) & ev.pctl.notna()]
    top = t[t.pctl >= TOP].copy()

    col = {s: i for i, s in enumerate(wide.columns)}
    top["stock_col"] = top.symbol.map(col)
    top["bench_col"] = col[pead.BENCHMARK]
    return top, "ret60", pt.COST, wide


def midsmall_sample():
    """research/oos_midsmall.py primary: top 5%, vs IJH / IJR, net 20bps."""
    oos = _load("oos_midsmall")
    conn = sqlite3.connect(f"file:{oos.DB_FILE}?mode=ro", uri=True)
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
    ev = ev[ev.pit & ev.pctl.notna() & (ev.entry_idx + oos.HOLD < len(cal))].copy()
    _, in400 = point_in_time(ev.symbol, ev.entry_date,
                             u[u.idx == "400"][["as_of", "symbol", "firm"]])
    ev["etf"] = np.where(in400, "IJH", "IJR")

    ff = closes.ffill()
    col = {s: i for i, s in enumerate(ff.columns)}
    px = ff.to_numpy()
    e, x = ev.entry_idx.to_numpy(), ev.entry_idx.to_numpy() + oos.HOLD
    c = ev.symbol.map(col).to_numpy()
    etf = np.array([px[xi, col[t]] / px[ei, col[t]] - 1 for ei, xi, t in zip(e, x, ev.etf)])
    ev["excess"] = px[x, c] / px[e, c] - 1 - etf
    ev = ev[np.isfinite(ev.excess)]
    top = ev[ev.pctl >= oos.PRIMARY].copy()
    top["stock_col"] = top.symbol.map(col)
    top["bench_col"] = top.etf.map(col)
    return top, "excess", oos.COST, closes


# ── running it ─────────────────────────────────────────────────────────

def check_signal(label, sample, target, blind):
    top, col, cost, closes = sample
    means = date_means(top, col, cost)
    m0, p0 = float(means.mean()), float(stats.ttest_1samp(means, 0).pvalue)
    print(f"\n══ {label} — top 5%, {len(top):,} events, {len(means):,} entry dates "
          f"({'BLIND' if blind else 'not blind'}) ══")
    print(f"   T0 original, date-clustered   {m0 * 100:+.3f}pp  p={p0:.4f}")

    (tm, tdp, tp, tpp) = target
    if abs(m0 * 100 - tm) > 0.5 * 10 ** -tdp + 1e-9 or abs(p0 - tp) > 0.5 * 10 ** -tpp + 1e-12:
        raise SystemExit(f"   T0 does NOT replicate the recorded {tm:+.{tdp}f}pp / p={tp:.{tpp}f}"
                         " — stopping, per the pre-registration.")
    print(f"      replicates the recorded {tm:+.{tdp}f}pp / p={tp:.{tpp}f}")

    m1, p1, q = quarter_clustered(means)
    print(f"   T1 same estimate, by quarter  {m1 * 100:+.3f}pp  p={p1:.4f}  ({q} quarters)")

    monthly = calendar_time(top.entry_idx.to_numpy(), top.stock_col.to_numpy(),
                            top.bench_col.to_numpy(), daily_returns(closes), closes.index, cost)
    a2, p2, beta, months = market_alpha(monthly)
    print(f"   T2 calendar-time alpha        {a2 * 100:+.2f}%/yr  p={p2:.4f}  "
          f"beta {beta:.2f}  ({months} months)")

    survives = m1 > 0 and p1 < ALPHA and a2 > 0 and p2 < ALPHA
    print(f"   → signal {'SURVIVES' if survives else 'DOES NOT SURVIVE'}"
          f"{'' if blind else '  (not blind)'}")
    return survives


def check_strategy():
    cfg = load_config()
    market = load_market(cfg)
    res = run(market, cfg)
    eq = res.equity["equity"] if isinstance(res.equity, pd.DataFrame) else res.equity
    spy = market.closes[cfg.benchmark].reindex(eq.index).ffill()
    monthly = pd.DataFrame({"strategy": eq.resample("ME").last(),
                            "spy": spy.resample("ME").last()}).pct_change().dropna()
    a, p, beta, months = market_alpha(monthly, "strategy", "spy")
    excess = (monthly.strategy - monthly.spy).mean() * 12
    print(f"\n══ T3 live strategy vs SPY (not blind) — {eq.index.min().date()} → "
          f"{eq.index.max().date()}, {months} months ══")
    print(f"   raw excess {excess * 100:+.2f}%/yr; alpha {a * 100:+.2f}%/yr  p={p:.4f}  beta {beta:.2f}")
    beats = a > 0 and p < ALPHA
    print(f"   → strategy {'BEATS SPY' if beats else 'NOT SHOWN to beat SPY'} after market exposure")
    return beats


def main():
    primary = check_signal("S&P 400/600 out-of-sample (PRIMARY)", midsmall_sample(),
                           (2.730, 3, 0.0009, 4), blind=True)
    check_signal("S&P 500, repaired universe (secondary)", sp500_sample(),
                 (1.91, 2, 0.002, 3), blind=False)
    check_strategy()
    print(f"\n══ PRE-REGISTERED PRIMARY VERDICT: signal "
          f"{'SURVIVES' if primary else 'DOES NOT SURVIVE'} overlap-robust inference ══")


if __name__ == "__main__":
    main()
