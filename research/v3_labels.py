# ═══════════════════════════════════════════════════════════════════════
#  v3 labels: the market's reaction to each earnings release (idea 1:
#  the market labels the text, not human sentiment annotators).
#
#  Design: research/design_sentiment_v3.md, "Timing".
#
#    reaction  stock return minus SPY, from the last close BEFORE the
#              announcement to the entry close (trader/events.py) — only
#              post-announcement price moves are inside the window
#
#  The gap signal (idea 2) needs the reaction to be known when the trade
#  is decided. The daily run decides at 14:30 ET and there is no free
#  intraday history, so the reaction is only complete after the entry
#  close: a gap trade enters at the NEXT session's close. `gap_entry_idx`
#  records that, and `readable` whether the 8-K was public by the 14:30
#  decision on that day.
#
#  S&P 500 only: this is development data. The S&P 400/600 exam set is
#  not labelled until its pre-registration is committed.
#
#    python research/v3_labels.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402

EDGAR_DB = ROOT / "data" / "edgar.db"
V3_DB    = ROOT / "data" / "v3.db"
ET       = "America/New_York"
CLOSE    = pd.Timedelta(hours=16)
DECISION = pd.Timedelta(hours=14, minutes=30)     # scripts/install_task.ps1


def pre_announcement_idx(announced_at, cal):
    """Index of the last session whose 16:00 ET close is before the
    announcement. Pre-open and intraday releases: the previous session.
    After-close releases: that same day's session."""
    ann = announced_at.dt.tz_convert(ET)
    day = pd.to_datetime(ann.dt.tz_localize(None).dt.normalize())
    left = cal.searchsorted(day, side="left")
    is_session = (left < len(cal)) & (cal[np.minimum(left, len(cal) - 1)] == day)
    after_close = (ann.dt.tz_localize(None) - day) >= CLOSE
    return np.where(is_session & after_close.to_numpy(), left, left - 1)


def build(cfg):
    market = load_market(cfg)
    cal, ev = market.cal, market.events
    ev = ev[ev.pit & ev.conviction.notna()].copy()
    ev["pre_idx"] = pre_announcement_idx(ev.announced_at, cal)
    ev["gap_entry_idx"] = ev.entry_idx + 1
    ev = ev[(ev.pre_idx >= 0) & (ev.gap_entry_idx < len(cal)) & (ev.pre_idx < ev.entry_idx)]

    closes = market.closes.ffill()
    col = {s: i for i, s in enumerate(closes.columns)}
    px = closes.to_numpy()
    p, e, c = ev.pre_idx.to_numpy(), ev.entry_idx.to_numpy(), ev.symbol.map(col).to_numpy()
    b = col[cfg.benchmark]
    ev["reaction"] = (px[e, c] / px[p, c] - 1) - (px[e, b] / px[p, b] - 1)
    ev["sue"] = (ev.eps_actual - ev.eps_estimate) / ev.price

    conn = sqlite3.connect(f"file:{EDGAR_DB}?mode=ro", uri=True, timeout=60)
    links = pd.read_sql_query(
        """SELECT e.event_key, e.cik, e.accession, f.accepted_at FROM event_filings e
           JOIN filings f USING (accession) WHERE f.body IS NOT NULL""", conn)
    conn.close()
    ev = ev.merge(links, left_on="key", right_on="event_key")
    decided = (pd.DatetimeIndex(cal[ev.gap_entry_idx.to_numpy()]) + DECISION).tz_localize(ET)
    ev["readable"] = pd.to_datetime(ev.accepted_at, utc=True).to_numpy() <= decided.tz_convert("UTC")
    ev = ev[np.isfinite(ev.reaction)]

    out = ev.assign(pre_date=cal[ev.pre_idx.to_numpy()].strftime("%Y-%m-%d"),
                    entry_date=ev.entry_date.dt.strftime("%Y-%m-%d"),
                    gap_entry_date=cal[ev.gap_entry_idx.to_numpy()].strftime("%Y-%m-%d"),
                    announced_at=ev.announced_at.dt.strftime("%Y-%m-%dT%H:%M:%S%z"))
    return out[["event_key", "symbol", "firm", "cik", "accession", "announced_at", "pre_date",
                "entry_date", "gap_entry_date", "pre_idx", "entry_idx", "gap_entry_idx", "reaction",
                "sue", "conviction", "readable"]]


def main():
    cfg = load_config()
    labels = build(cfg)
    out = sqlite3.connect(V3_DB, timeout=60)
    out.execute("PRAGMA journal_mode=WAL")
    labels.to_sql("labels", out, if_exists="replace", index=False)
    out.close()

    r = labels.reaction
    print(f"{len(labels):,} S&P 500 events with a release, "
          f"{labels.entry_date.min()} → {labels.entry_date.max()}")
    print(f"   reaction: median {r.median() * 100:+.2f}pp, middle 90% "
          f"[{r.quantile(.05) * 100:+.1f}, {r.quantile(.95) * 100:+.1f}]pp, "
          f"|reaction| median {r.abs().median() * 100:.2f}pp")
    print(f"   release readable by the gap trade's decision: {labels.readable.mean():.1%}")
    print(f"   reaction window, sessions: "
          f"{(labels.entry_idx - labels.pre_idx).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
