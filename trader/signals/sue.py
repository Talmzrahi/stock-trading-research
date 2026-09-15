# ═══════════════════════════════════════════════════════════════════════
#  SUE signal — price-scaled earnings surprise, ranked point-in-time.
#
#  research/pead.py cut deciles within each calendar quarter, which uses
#  events reported LATER in that quarter to rank earlier ones. A live
#  system reporting in week 1 of earnings season cannot know week 6's
#  surprises. Here each event is ranked only against events dated
#  strictly before it, within a trailing window — exactly what could have
#  been computed on the day.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

WINDOW_DAYS = 365     # ~4 quarters of history
MIN_HISTORY = 200     # below this the percentile is too noisy to trade on


def sue(eps_actual, eps_estimate, price):
    """Price-scaled surprise. Unlike (actual - estimate) / |estimate|, it does
    not explode when the estimate is near zero."""
    return (eps_actual - eps_estimate) / price


def trailing_percentile(dates, values, window_days=WINDOW_DAYS, min_history=MIN_HISTORY):
    """Fraction of trailing-window values below each value, in [0, 1].

    The history for an event on date t is every value dated in
    [t - window_days, t), so same-day events never rank against each other
    and nothing from the future leaks in. NaN where history is too short
    or the value itself is missing.
    """
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    v = np.asarray(values, dtype=float)
    order = np.argsort(d, kind="stable")
    d_s, v_s = d[order], v[order]

    starts = np.searchsorted(d_s, d_s - np.timedelta64(window_days, "D"), side="left")
    _, first = np.unique(d_s, return_index=True)
    bounds = np.append(first, len(d_s))

    out_s = np.full(len(v_s), np.nan)
    for f, last in zip(bounds[:-1], bounds[1:]):
        hist = v_s[starts[f]:f]               # f is the first index of this date
        hist = np.sort(hist[~np.isnan(hist)])
        if len(hist) < min_history:
            continue
        x = v_s[f:last]
        out_s[f:last] = np.where(np.isnan(x), np.nan,
                                 np.searchsorted(hist, x, side="left") / len(hist))

    out = np.empty_like(out_s)
    out[order] = out_s
    return out
