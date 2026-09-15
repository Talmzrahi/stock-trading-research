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

    groups = universe.groupby("as_of")["symbol"].apply(set).sort_index()
    snaps, members = pd.DatetimeIndex(groups.index), groups.tolist()
    k = snaps.searchsorted(ev.entry_date.to_numpy(), side="right") - 1
    ev["pit"] = [i >= 0 and s in members[i] for s, i in zip(ev.symbol, k)]

    return ev.sort_values(["entry_idx", "symbol"]).reset_index(drop=True)
