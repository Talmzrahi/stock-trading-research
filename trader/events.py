# ═══════════════════════════════════════════════════════════════════════
#  Earnings events with point-in-time timing — layer 1 → layer 2 handoff.
#
#  Same conservative timing as research/pead.py: a release before the
#  09:30 open can be traded at that day's close; anything at or after the
#  open waits for the next session's close, so the announcement jump is
#  never captured.
# ═══════════════════════════════════════════════════════════════════════

from datetime import time as dtime

import numpy as np
import pandas as pd

MARKET_OPEN = dtime(9, 30)


def build_events(earn, calendar, closes, universe):
    """One row per announcement: timing, surprise inputs, index membership.

    entry_idx   calendar index of the first close a trader could act on
    signal_idx  last session strictly before the announcement day; its
                close scales the surprise
    pit         firm was in the S&P 500 (latest snapshot on/before entry)
    """
    cal = pd.DatetimeIndex(calendar)
    e = earn.reset_index(drop=True)
    ann_et = e["announced_at"].dt.tz_convert("America/New_York")
    ann_day = pd.to_datetime(ann_et.dt.date)
    before_open = (ann_et.dt.time < MARKET_OPEN).to_numpy()
    left = cal.searchsorted(ann_day, side="left")
    right = cal.searchsorted(ann_day, side="right")

    ev = e[["symbol", "announced_at", "eps_estimate", "eps_actual"]].copy()
    ev["ann_date"] = ann_day.to_numpy()
    ev["entry_idx"] = np.where(before_open, left, right)
    ev["signal_idx"] = left - 1
    ev = ev[(ev.signal_idx >= 0) & (ev.entry_idx < len(cal))].copy()
    ev["entry_date"] = cal[ev.entry_idx.to_numpy()]
    ev["key"] = ev.symbol + "|" + pd.to_datetime(ev.ann_date).dt.strftime("%Y-%m-%d")
    ev = ev.sort_values("announced_at").drop_duplicates("key", keep="last")

    aligned = closes.reindex(index=cal)
    col_of = {s: i for i, s in enumerate(aligned.columns)}
    ev = ev[ev.symbol.isin(list(col_of))].copy()
    vals = aligned.to_numpy()
    ev["price"] = vals[ev.signal_idx.to_numpy(), ev.symbol.map(col_of).to_numpy()]
    ev = ev[ev.price > 0].copy()

    ev["firm"], ev["pit"] = point_in_time(ev.symbol, ev.entry_date, universe)
    # One event per firm per day, so share classes (GOOG/GOOGL) count once.
    ev = ev.sort_values(["firm", "ann_date", "symbol"]).drop_duplicates(["firm", "ann_date"])
    return ev.sort_values(["entry_idx", "symbol"]).reset_index(drop=True)


def point_in_time(symbols, dates, universe):
    """(firm, was-a-member) for each (symbol, date).

    Membership is matched by firm when the universe carries firm identity,
    so a ticker rename (BK → BNY) doesn't turn the firm's pre-rename history
    into non-member events. A ticker carries its firm backwards without
    limit but forwards only a year past its last listing: after a ticker
    leaves, its data may belong to someone else (AA's history runs on into
    the 2016 Alcoa spin-off, not the firm that was the member).
    Shared by the live system and the research gates so all three agree.
    """
    symbols, dates = list(symbols), pd.DatetimeIndex(dates)
    if "firm" in universe.columns:
        latest = universe.sort_values("as_of").drop_duplicates("symbol", keep="last")
        firm_of = dict(zip(latest.symbol, latest.firm))
        until = {s: d + pd.Timedelta(days=365) for s, d in zip(latest.symbol, latest.as_of)}
        firms = [firm_of[s] if s in firm_of and d <= until[s] else f"SYM:{s}"
                 for s, d in zip(symbols, dates)]
        groups = universe.groupby("as_of")["firm"].apply(set).sort_index()
    else:
        firms = symbols
        groups = universe.groupby("as_of")["symbol"].apply(set).sort_index()
    snaps, members = pd.DatetimeIndex(groups.index), groups.tolist()
    k = snaps.searchsorted(dates, side="right") - 1
    return firms, [i >= 0 and f in members[i] for f, i in zip(firms, k)]
