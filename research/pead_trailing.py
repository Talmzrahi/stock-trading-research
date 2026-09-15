# ═══════════════════════════════════════════════════════════════════════
#  Gate re-test: PEAD with point-in-time (trailing) decile cutoffs.
#
#  pead.py's quarterly deciles rank each event against the whole quarter,
#  including firms that report weeks later — unknowable live. This re-runs
#  the top-decile test with cutoffs from the prior 365 days only
#  (trader/signals/sue.py), on the point-in-time universe.
#
#  PRE-REGISTERED (2026-09-15, before running):
#    PASS iff top-decile 60d return, net of 10bps, is > 0 with p < 0.05,
#    date-clustered, point-in-time universe. FAIL stops the build.
#
#  RESULT (2026-09-15): PASS — +1.225pp net, p=0.003, 1,071 dates. The
#  PIT snapshots start 2010-04 (not 2007 as universe.py's range suggests),
#  so with a 200-event trailing burn-in the sample runs 2010-05 → 2026-06.
#  Trailing and quarterly cutoffs agree on 90% of top-decile events.
#  2014-2019 remains negative (-0.51pp, p=0.31).
#
#  RE-RUN (2026-09-15) after repairing the universe — 100 departed firms
#  added back (research/ingest_departed.py) and membership matched by firm
#  through ticker renames (research/universe_ids.py): FAIL — top decile
#  60d +0.767pp net, p=0.063, 1,185 dates. The first PASS leaned on a
#  survivors-only universe. Tighter cutoffs still show drift (top 5%
#  +1.91pp p=0.002; top 2% +1.96pp p=0.052) but picking one now is
#  post-hoc and was not the pre-registered test.
#
#  The cutoff grid and within-decile buckets are informational only — the
#  tighter-cutoff decision is made later on portfolio net total return.
#
#    python research/pead_trailing.py
# ═══════════════════════════════════════════════════════════════════════

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.data import load_universe  # noqa: E402
from trader.events import point_in_time  # noqa: E402
from trader.signals.sue import trailing_percentile  # noqa: E402

_spec = importlib.util.spec_from_file_location("pead", Path(__file__).resolve().parent / "pead.py")
pead = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pead)

TEST_START = pd.Timestamp("2008-01-01")
COST       = pead.COST_BPS / 1e4
HOLD       = 60
ALPHA      = 0.05
TOP        = 0.90
CUTOFFS    = [0.90, 0.95, 0.97, 0.98]


def clustered(g, col, cost=0.0):
    """Mean of per-date means and its one-sample p — same-day events share a
    market shock, so the date is the independent unit."""
    m = (g[col] - cost).groupby(g.entry_date).mean()
    if len(m) < 5:
        return np.nan, np.nan, len(m)
    return m.mean(), stats.ttest_1samp(m, 0).pvalue, len(m)


def main():
    conn = sqlite3.connect(pead.DB_FILE)
    wide, _vix, earn = pead.bt.load(conn)
    universe = load_universe(conn)
    conn.close()
    ev = pead.build(wide, earn)
    ev = ev[ev.entry_date >= universe.as_of.min()].copy()
    # Same membership rule as the live system: by firm, through renames.
    ev["firm"], ev["pit"] = point_in_time(ev.symbol, ev.entry_date, universe)
    ev["ann_day"] = ev.announced_at.dt.tz_convert("America/New_York").dt.date
    ev = ev.sort_values(["firm", "ann_day", "symbol"]).drop_duplicates(["firm", "ann_day"])

    # Pool = point-in-time members only, matching what the live system sees.
    ev = ev[ev.pit].copy()
    print(f"Point-in-time member events: {len(ev):,} across {ev.firm.nunique()} firms")
    ev["pctl"] = trailing_percentile(ev.entry_date, ev.sue)
    t = ev[(ev.entry_date >= TEST_START) & ev.pctl.notna()].copy()
    years = (t.entry_date.max() - t.entry_date.min()).days / 365.25

    print(f"Events: {len(t):,} point-in-time, {t.entry_date.min().date()} → "
          f"{t.entry_date.max().date()} ({years:.1f}y)")

    trail = t[t.pctl >= TOP]
    qcut = t[t.decile == pead.N_DECILES]
    both = len(trail.index.intersection(qcut.index))
    print(f"Top decile: trailing {len(trail):,}  quarterly {len(qcut):,}  "
          f"overlap {both:,} ({both / len(qcut) * 100:.0f}% of quarterly)")

    print("\n── Top decile, net of cost (%) — trailing vs quarterly cutoffs ─")
    print(f"   {'h':>4} {'trailing':>10} {'p':>7} {'quarterly':>11} {'p':>7}")
    for h in pead.HORIZONS:
        a, pa, _ = clustered(trail, f"ret{h}", COST)
        b, pb, _ = clustered(qcut, f"ret{h}", COST)
        print(f"   {h:>4} {a*100:>+9.3f}  {pa:>7.3f} {b*100:>+10.3f}  {pb:>7.3f}")

    print(f"\n── Stability by era, trailing top decile, {HOLD}d net ─────────────")
    bins = [TEST_START, pd.Timestamp("2014-01-01"), pd.Timestamp("2020-01-01"),
            pd.Timestamp("2027-01-01")]
    trail = trail.assign(era=pd.cut(trail.entry_date, bins=bins, right=False,
                                    labels=["2008-2013", "2014-2019", "2020-2026"]))
    for era, g in trail.groupby("era", observed=True):
        m, p, n = clustered(g, f"ret{HOLD}", COST)
        print(f"   {era}  {m*100:>+7.3f}pp  p={p:.3f}  events={len(g):,}  dates={n}")

    print(f"\n── Cutoff grid, {HOLD}d net (informational) ──────────────────────")
    print(f"   {'cutoff':>8} {'net':>9} {'p':>7} {'entries/yr':>11} {'~held':>7}")
    for c in CUTOFFS:
        g = t[t.pctl >= c]
        m, p, _ = clustered(g, f"ret{HOLD}", COST)
        per_yr = len(g) / years
        print(f"   top {(1-c)*100:>3.0f}% {m*100:>+8.3f}  {p:>7.3f} {per_yr:>11.0f} "
              f"{per_yr * HOLD / 252:>7.1f}")

    print(f"\n── Does conviction scale inside the top decile? ({HOLD}d net) ────")
    edges = CUTOFFS + [1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = t[(t.pctl >= lo) & (t.pctl < hi)]
        m, p, _ = clustered(g, f"ret{HOLD}", COST)
        print(f"   pctl {lo:.2f}–{min(hi, 1):.2f}  {m*100:>+7.3f}pp  p={p:.3f}  n={len(g):,}")

    m, p, n = clustered(t[t.pctl >= TOP], f"ret{HOLD}", COST)
    verdict = "PASS" if (m > 0 and p < ALPHA) else "FAIL"
    print(f"\n══ PRE-REGISTERED VERDICT: {verdict} ══")
    print(f"   top decile {HOLD}d net {m*100:+.3f}pp, p={p:.4f}, {n} dates "
          f"(rule: net > 0 and p < {ALPHA})")


if __name__ == "__main__":
    main()
