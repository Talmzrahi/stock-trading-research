# ═══════════════════════════════════════════════════════════════════════
#  Phase 1 hypothesis test
#
#  H1: stocks that beat EPS estimates while VIX is elevated above its
#      trailing average earn excess forward returns beyond what a plain
#      earnings-beat effect explains.
#  H0: no difference between beats in elevated-VIX vs. normal-VIX regimes.
#
#  Run after research/ingest.py:
#    python research/backtest.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

DB_FILE      = Path(__file__).resolve().parent.parent / "data" / "research.db"
BENCHMARK    = "SPY"
HORIZONS     = [1, 3, 5, 10]
VIX_WINDOW   = 60      # trading days for the trailing VIX average
VIX_Z        = 1.0     # elevated = VIX at least this many sd above trailing mean
BEAT_PCT     = 2.0     # "beat" = EPS surprise above this %
COST_BPS     = 10.0    # round-trip transaction cost assumption
OOS_SPLIT    = "2017-01-01"
N_PERM       = 5000
RNG          = np.random.default_rng(42)


# ── Data loading ──────────────────────────────────────────────────────
def load(conn):
    prices = pd.read_sql_query("SELECT symbol, date, close FROM prices", conn)
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot(index="date", columns="symbol", values="close").sort_index()

    vix = pd.read_sql_query("SELECT date, close FROM vix", conn)
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date")["close"].sort_index()

    earn = pd.read_sql_query(
        """SELECT symbol, announced_at, eps_estimate, eps_actual, surprise_pct
           FROM earnings
           WHERE eps_actual IS NOT NULL AND eps_estimate IS NOT NULL""",
        conn,
    )
    earn["announced_at"] = pd.to_datetime(earn["announced_at"], utc=True, format="mixed")
    return wide, vix, earn


def build_events(wide, vix, earn):
    """Attach entry dates, forward market-adjusted returns, and VIX regime.

    Timing is deliberately conservative: an announcement at or after the
    open means entry at the NEXT session's close, so the initial
    announcement jump is never captured. What remains is the drift, which
    is exactly what the hypothesis claims to exploit.
    """
    cal = wide.index
    if BENCHMARK not in wide.columns:
        raise SystemExit(f"benchmark {BENCHMARK} missing from prices table")

    # VIX regime, computed only from data available up to each date.
    vix_al = vix.reindex(cal).ffill()
    roll = vix_al.rolling(VIX_WINDOW)
    vix_z = ((vix_al - roll.mean()) / roll.std()).rename("vix_z")

    ann_et = earn["announced_at"].dt.tz_convert("America/New_York")
    ann_day = pd.to_datetime(ann_et.dt.date)
    before_open = ann_et.dt.time < dtime(9, 30)

    pos = cal.searchsorted(ann_day, side="left")
    entry_idx = np.where(before_open, pos, cal.searchsorted(ann_day, side="right"))
    signal_idx = pos - 1                      # last session strictly before the news

    ev = earn.copy()
    ev["entry_idx"] = entry_idx
    ev["signal_idx"] = signal_idx
    ev = ev[(ev.signal_idx >= 0) & (ev.entry_idx < len(cal) - max(HORIZONS))]

    ev["entry_date"] = cal[ev.entry_idx.to_numpy()]
    ev["vix_z"] = vix_z.to_numpy()[ev.signal_idx.to_numpy()]
    ev = ev.dropna(subset=["vix_z"])

    # Forward returns, vectorised over the whole price matrix.
    col_of = {s: i for i, s in enumerate(wide.columns)}
    ev = ev[ev.symbol.isin(col_of)]
    rows = ev.entry_idx.to_numpy()
    cols = ev.symbol.map(col_of).to_numpy()
    px = wide.to_numpy()
    spy = wide[BENCHMARK].to_numpy()

    for h in HORIZONS:
        fwd = px[rows + h, cols] / px[rows, cols] - 1.0
        mkt = spy[rows + h] / spy[rows] - 1.0
        ev[f"ret{h}"] = fwd - mkt          # market-adjusted

    ev["beat"] = ev.surprise_pct > BEAT_PCT
    ev["elevated"] = ev.vix_z >= VIX_Z
    return ev.dropna(subset=[f"ret{h}" for h in HORIZONS])


# ── Inference ─────────────────────────────────────────────────────────
def date_level(ev, col):
    """Collapse to one observation per entry date.

    Every stock reporting on a given date shares the same VIX regime, and
    their returns share a common market component. The honest unit of
    observation is therefore the date, not the individual event.
    """
    # Group on (date, regime) rather than date alone: a same-day entry can
    # carry either regime flag when one report was pre-open and another came
    # after the prior close.
    g = (ev.groupby(["entry_date", "elevated"])
           .agg(ret=(col, "mean"), n=(col, "size"))
           .reset_index())
    return g


def permutation_p(dl, n_perm=N_PERM):
    """Shuffle the elevated/normal label across DATES, preserving clustering."""
    obs = dl.loc[dl.elevated, "ret"].mean() - dl.loc[~dl.elevated, "ret"].mean()
    flags = dl.elevated.to_numpy().copy()
    rets = dl.ret.to_numpy()
    k = flags.sum()
    if k == 0 or k == len(flags):
        return obs, np.nan
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(flags)
        diffs[i] = rets[perm].mean() - rets[~perm].mean()
    return obs, float((np.abs(diffs) >= abs(obs)).mean())


def interaction(ev, h):
    """The actual hypothesis test: does beating earnings pay MORE when VIX is
    elevated, over and above the regime's own effect on all stocks?

    The elevated-vs-normal contrast on beats alone cannot answer this — it
    conflates the interaction with a regime effect that lifts beats and
    non-beats alike. The difference-in-differences (this interaction term)
    isolates it, and as a within-universe difference it is also robust to the
    survivorship bias in the constituent list.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    d = ev.assign(beat_i=ev.beat.astype(int), elev_i=ev.elevated.astype(int),
                  date_id=ev.entry_date.astype("category").cat.codes)
    m = smf.ols(f"ret{h} ~ beat_i * elev_i", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.date_id})
    return {"h": h,
            "interaction": m.params["beat_i:elev_i"], "p_int": m.pvalues["beat_i:elev_i"],
            "beat": m.params["beat_i"], "p_beat": m.pvalues["beat_i"],
            "regime": m.params["elev_i"], "p_regime": m.pvalues["elev_i"]}


def contrast(ev, h, label):
    col = f"ret{h}"
    beats = ev[ev.beat]
    dl = date_level(beats, col)
    hi, lo = dl[dl.elevated], dl[~dl.elevated]
    if len(hi) < 5 or len(lo) < 5:
        print(f"  {label:>10s} h={h:<3d} insufficient data (hi={len(hi)} lo={len(lo)} dates)")
        return None

    diff, p_perm = permutation_p(dl)
    t, p_t = stats.ttest_ind(hi.ret, lo.ret, equal_var=False)
    net = hi.ret.mean() - COST_BPS / 1e4

    print(f"  {label:>10s} h={h:<3d} "
          f"elev={hi.ret.mean()*100:+6.3f}%  norm={lo.ret.mean()*100:+6.3f}%  "
          f"diff={diff*100:+6.3f}pp  "
          f"p_clust={p_t:.3f}  p_perm={p_perm:.3f}  "
          f"net_of_cost={net*100:+6.3f}%  "
          f"dates={len(hi)}/{len(lo)}  events={len(beats):,}")
    return {"h": h, "diff": diff, "p_perm": p_perm, "p_clust": p_t,
            "elev_mean": hi.ret.mean(), "net": net,
            "n_hi": len(hi), "n_lo": len(lo)}


def main():
    if not DB_FILE.exists():
        raise SystemExit(f"{DB_FILE} not found — run research/ingest.py first")
    conn = sqlite3.connect(DB_FILE)
    wide, vix, earn = load(conn)
    conn.close()

    print(f"Loaded: {wide.shape[1]} symbols, {len(wide)} sessions, {len(earn):,} earnings records")
    ev = build_events(wide, vix, earn)
    print(f"Usable events: {len(ev):,}  ({ev.entry_date.min().date()} → {ev.entry_date.max().date()})")
    print(f"Config: beat>{BEAT_PCT}%  elevated=VIX z>={VIX_Z} ({VIX_WINDOW}d)  cost={COST_BPS}bps\n")

    print("── 2x2 mean market-adjusted return, 5-day horizon ─────────────")
    tab = ev.groupby(["beat", "elevated"])["ret5"].agg(["mean", "size"])
    for (b, e), r in tab.iterrows():
        print(f"  beat={str(b):<5s} elevated={str(e):<5s}  mean={r['mean']*100:+6.3f}%  n={int(r['size']):,}")

    print("\n── PRIMARY TEST: beat x elevated interaction ──────────────────")
    print("   H1 needs a POSITIVE interaction: beats rewarded extra during fear.")
    print("   Decomposed into regime effect, plain beat effect, and interaction.")
    inter = [interaction(ev, h) for h in HORIZONS]
    for r in inter:
        if r is None:
            print("   statsmodels not installed — skipped")
            break
        print(f"   h={r['h']:<3d} interaction={r['interaction']*100:+7.4f}pp p={r['p_int']:.3f}   "
              f"| beat={r['beat']*100:+6.3f}pp p={r['p_beat']:.3f}   "
              f"| regime={r['regime']*100:+6.3f}pp p={r['p_regime']:.3f}")

    print("\n── SECONDARY: beats in elevated vs normal VIX ─────────────────")
    print("   Tradeable contrast, but it CONFLATES the interaction with the")
    print("   regime effect — do not read it as support for H1 on its own.")
    full = [contrast(ev, h, "full") for h in HORIZONS]

    print("\n── OUT-OF-SAMPLE SPLIT ────────────────────────────────────────")
    ins = ev[ev.entry_date < OOS_SPLIT]
    oos = ev[ev.entry_date >= OOS_SPLIT]
    print(f"   in-sample {ins.entry_date.min().date()}→{ins.entry_date.max().date()} ({len(ins):,} events)")
    for h in HORIZONS:
        contrast(ins, h, "in-samp")
    print(f"   out-of-sample {oos.entry_date.min().date()}→{oos.entry_date.max().date()} ({len(oos):,} events)")
    for h in HORIZONS:
        contrast(oos, h, "OOS")

    print("\n── THRESHOLD SENSITIVITY (5-day, full sample) ─────────────────")
    print("   reporting the whole grid, not the best cell")
    print(f"   {'beat%':>6s} {'vixZ':>5s} {'contrast':>9s} {'p_perm':>7s} "
          f"{'interact':>9s} {'p_int':>7s} {'dates hi/lo':>13s}")
    for bt in [0.0, 2.0, 5.0]:
        for vz in [0.5, 1.0, 1.5, 2.0]:
            sub = ev.copy()
            sub["beat"] = sub.surprise_pct > bt
            sub["elevated"] = sub.vix_z >= vz
            dl = date_level(sub[sub.beat], "ret5")
            hi, lo = dl[dl.elevated], dl[~dl.elevated]
            if len(hi) < 5 or len(lo) < 5:
                print(f"   {bt:6.1f} {vz:5.1f} {'--':>9s} {'--':>7s} {'--':>9s} {'--':>7s}"
                      f" {f'{len(hi)}/{len(lo)}':>13s}")
                continue
            d, p = permutation_p(dl, n_perm=1000)
            ir = interaction(sub, 5)
            istr = f"{ir['interaction']*100:+9.3f}" if ir else f"{'--':>9s}"
            ipstr = f"{ir['p_int']:7.3f}" if ir else f"{'--':>7s}"
            print(f"   {bt:6.1f} {vz:5.1f} {d*100:+9.3f} {p:7.3f} {istr} {ipstr}"
                  f" {f'{len(hi)}/{len(lo)}':>13s}")

    print("\n" + "═" * 63)
    hits = [r for r in inter if r and r["p_int"] < 0.05 and r["interaction"] > 0]
    if hits:
        print(f"VERDICT: interaction is positive and significant at "
              f"{len(hits)}/{len(HORIZONS)} horizons — consistent with H1.")
        print("Confirm it holds across the sensitivity grid and the OOS split.")
    else:
        print("VERDICT: H1 NOT SUPPORTED. The beat x elevated interaction is not")
        print("positively significant at any horizon, so beating earnings is not")
        print("rewarded any more during elevated VIX than at normal times.")
        print("Any elevated-vs-normal gap in the secondary contrast is a regime")
        print("effect that lifts beats and non-beats alike, not the mechanism H1")
        print("proposes — and that regime effect is itself inflated by the")
        print("survivorship bias in the constituent list.")


if __name__ == "__main__":
    main()
