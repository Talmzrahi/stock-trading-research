# ═══════════════════════════════════════════════════════════════════════
#  NYSE trading calendar, rule-based.
#
#  The live run needs to know whether today is a session and when the
#  market-on-close cutoff falls, before today's bar exists. Rules plus a
#  list of one-off closures; tests check the result against every SPY
#  session in research.db. Unscheduled future closures (a state funeral,
#  a hurricane) cannot be known — the runner treats a day with no SPY bar
#  as a non-session when settling, so a miss costs a wasted run, not a
#  wrong fill.
# ═══════════════════════════════════════════════════════════════════════

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

# Closures not covered by the holiday rules.
SPECIAL_CLOSURES = {
    date(2004, 6, 11),                       # Reagan funeral
    date(2007, 1, 2),                        # Ford funeral
    date(2012, 10, 29), date(2012, 10, 30),  # Hurricane Sandy
    date(2018, 12, 5),                       # G.H.W. Bush funeral
    date(2025, 1, 9),                        # Carter funeral
}

MOC_CUTOFF        = time(15, 50)   # Alpaca / NYSE closing-auction entry cutoff
MOC_CUTOFF_EARLY  = time(12, 50)   # on 13:00 early-close days


def _easter(y):
    """Anonymous Gregorian algorithm."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def _nth_weekday(y, month, weekday, n):
    d = date(y, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y, month, weekday):
    d = date(y + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d):
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(y):
    h = {
        _nth_weekday(y, 1, 0, 3),            # MLK Day
        _nth_weekday(y, 2, 0, 3),            # Washington's Birthday
        _easter(y) - timedelta(days=2),      # Good Friday
        _last_weekday(y, 5, 0),              # Memorial Day
        _observed(date(y, 7, 4)),            # Independence Day
        _nth_weekday(y, 9, 0, 1),            # Labor Day
        _nth_weekday(y, 11, 3, 4),           # Thanksgiving
        _observed(date(y, 12, 25)),          # Christmas
    }
    # New Year's: a Saturday Jan 1 is NOT moved back into the prior year.
    ny = date(y, 1, 1)
    if ny.weekday() == 6:
        h.add(date(y, 1, 2))
    elif ny.weekday() < 5:
        h.add(ny)
    if y >= 2022:
        h.add(_observed(date(y, 6, 19)))     # Juneteenth
    return h


def is_trading_day(d):
    d = pd.Timestamp(d).date()
    return d.weekday() < 5 and d not in holidays(d.year) and d not in SPECIAL_CLOSURES


def is_early_close(d):
    d = pd.Timestamp(d).date()
    if not is_trading_day(d):
        return False
    thanksgiving = _nth_weekday(d.year, 11, 3, 4)
    return (d == thanksgiving + timedelta(days=1)
            or (d.month == 12 and d.day == 24)
            or (d.month == 7 and d.day == 3))


def trading_days(start, end):
    """All sessions in [start, end] as a normalized DatetimeIndex."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return pd.DatetimeIndex([d for d in days if is_trading_day(d)])


def next_trading_day(d):
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_trading_day(d):
        d += pd.Timedelta(days=1)
    return d


def now_et():
    return datetime.now(ET)


def moc_cutoff(d):
    d = pd.Timestamp(d).date()
    return datetime.combine(d, MOC_CUTOFF_EARLY if is_early_close(d) else MOC_CUTOFF, tzinfo=ET)
