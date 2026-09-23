# ═══════════════════════════════════════════════════════════════════════
#  Backtest mechanics, in one place, with tests.
#
#  Every research script in this project used to reimplement weight
#  timing, cost accounting and leverage charging by hand. Between
#  2026-09-21 and 2026-09-23 that produced five separate errors, each of
#  which returned a number that looked like a finding:
#
#    1  in-sample ranking      pd.qcut over the whole sample ranked each
#       (v3_drift_rule)        event against trades that had not happened.
#                              +8.85%/yr became +1.18%/yr when removed.
#    2  full-sample threshold  mean + k*sd computed over 2011-2026 let a
#       (v3_drift_rule)        2015 decision consult 2026. +9.54%/yr
#                              became +1.65%/yr, and -2.65% on a 1y window.
#    3  fitting on the future  out_of_fold trains each fold on all the
#       (v3_drift_rule)        others, later ones included. Cost about
#                              half the remaining effect.
#    4  cost sign inversion    `side * total` applied AFTER subtracting
#       (longshort_probe)      cost turned every short-side cost into a
#                              gain. +1.51%/yr became +0.58%/yr.
#    5  post-hoc vol matching  scaling a series by its own full-sample
#       (voltarget_multi)      standard deviation, charging no margin on
#                              the implied leverage. Showed 165x growth.
#
#  Four of the five are timing errors and one is an accounting error.
#  None were exotic; all were a line of arithmetic written slightly wrong
#  in a script that had no tests. Hence this module: each function below
#  is the correct version of something already got wrong, the docstring
#  names which, and tests/test_mechanics.py encodes each as a regression.
#
#  Use these rather than rewriting them. The point is not elegance, it is
#  that the next script cannot make error 4 again.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ── timing ─────────────────────────────────────────────────────────────

def month_end_mask(index):
    """True on the last trading day of each month. Calendar month ends are
    not trading days, so this is computed on the index itself."""
    idx = pd.DatetimeIndex(index)
    return (idx.to_series().groupby(idx.to_period("M")).transform("max") == idx).to_numpy()


def hold_from_month_end(raw, index):
    """A weight decided at each month-end close, applied from the NEXT
    trading day and held until the following month end.

    The `.shift(1)` is the whole point: without it the weight set on day t
    earns day t's return, which it was partly computed from. Prevents
    error 1/2-class timing mistakes at the portfolio level.
    """
    raw = pd.Series(np.asarray(raw, dtype=float), index=pd.DatetimeIndex(index))
    me = month_end_mask(raw.index)
    return raw.where(me).shift(1).ffill()


def trailing_stats(dates, values, window_days=1095, min_obs=30):
    """Mean and standard deviation of every value dated strictly EARLIER
    than each observation, within a trailing window.

    Strictly earlier matters: same-day observations must not inform their
    own threshold. This is the fix for error 2 — a full-sample mean and sd
    let a decision consult its own future. NaN where history is short.
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    v = np.asarray(values, dtype=float)
    order = np.argsort(d, kind="stable")
    d_s, v_s = d[order], v[order]

    starts = np.searchsorted(d_s, d_s - np.timedelta64(int(window_days), "D"), side="left")
    _, first = np.unique(d_s, return_index=True)
    bounds = np.append(first, len(d_s))

    m_s = np.full(len(v_s), np.nan)
    s_s = np.full(len(v_s), np.nan)
    for f, last in zip(bounds[:-1], bounds[1:]):
        hist = v_s[starts[f]:f]              # f is the first row of this date
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_obs:
            continue
        m_s[f:last] = hist.mean()
        s_s[f:last] = hist.std(ddof=1)

    mean = np.empty_like(m_s)
    sd = np.empty_like(s_s)
    mean[order], sd[order] = m_s, s_s
    return mean, sd


# ── accounting ─────────────────────────────────────────────────────────

def position_book(entry_idx, stock_col, returns, hold, side=1, cost=0.0, n_days=None):
    """Daily equal-weighted return of a book of fixed-horizon positions,
    plus the open-position count.

    `side` multiplies the RETURN (+1 long, -1 short). Cost is ALWAYS a
    subtraction, charged on entry and exit. Applying `side` after the cost
    subtraction turns short-side costs into gains — that was error 4, and
    it overstated a strategy by roughly +0.9pp/yr at a 60-session hold and
    ~2.5pp/yr at 20 sessions.
    """
    n_days = n_days if n_days is not None else returns.shape[0]
    total, count = np.zeros(n_days), np.zeros(n_days)
    for e, c in zip(np.asarray(entry_idx), np.asarray(stock_col)):
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        c = int(c)
        window = slice(e + 1, e + hold + 1)
        total[window] += side * returns[window, c]
        count[window] += 1
        total[e + 1] -= cost                            # entry
        total[min(e + hold, n_days - 1)] -= cost        # exit
    daily = np.zeros(n_days)
    open_ = count > 0
    daily[open_] = total[open_] / count[open_]
    return daily, count


def levered_return(weight, asset_return, rf_daily, margin_spread=0.015):
    """Return of `weight` in an asset with the remainder in cash — or
    borrowed, when weight exceeds 1.

    Below 1 the idle fraction earns the risk-free rate; above 1 the
    borrowed fraction PAYS rf + spread. Leverage is never free, and
    omitting its cost was half of error 5.
    """
    w = np.asarray(weight, dtype=float)
    rf = np.asarray(rf_daily, dtype=float)
    rate = np.where(w <= 1.0, rf, rf + margin_spread / TRADING_DAYS)
    return w * np.asarray(asset_return, dtype=float) + (1 - w) * rate


# ── measurement ────────────────────────────────────────────────────────

def performance(returns, rf_daily=None):
    """Annualised return, volatility, Sharpe and maximum drawdown.

    Sharpe is EXCESS of the risk-free rate whenever `rf_daily` is given,
    because raw return/vol flatters whichever strategy holds more cash —
    the risk-free rate sits in its numerator. Both are returned so the
    difference stays visible rather than being quietly chosen.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return dict(ret=np.nan, vol=np.nan, sharpe=np.nan, sharpe_raw=np.nan,
                    max_dd=np.nan, growth=np.nan, n=len(r))
    ann = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + r).cumprod()
    raw = ann / vol if vol else np.nan
    excess = raw
    if rf_daily is not None:
        rf = pd.Series(rf_daily).reindex(r.index).fillna(0.0)
        ex = r - rf
        excess = ((1 + ex).prod() ** (TRADING_DAYS / len(ex)) - 1) / vol if vol else np.nan
    return dict(ret=float(ann), vol=float(vol), sharpe=float(excess),
                sharpe_raw=float(raw), max_dd=float((eq / eq.cummax() - 1).min()),
                growth=float(eq.iloc[-1]), n=len(r))


def years_to_significance(sharpe, alpha_z=1.96):
    """Years of live data needed for a strategy of this Sharpe to be
    distinguishable from zero. Quoted before proposing any shadow book:
    at Sharpe 0.36 it is ~30 years, which is not a test.
    """
    if not sharpe or sharpe <= 0:
        return float("inf")
    return (alpha_z / sharpe) ** 2
