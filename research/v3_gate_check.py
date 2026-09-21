# ═══════════════════════════════════════════════════════════════════════
#  Does the text signal clear the gate — on DEVELOPMENT data?
#
#  The mandatory gate (CLAUDE.md) asks two things of anything with
#  overlapping holding periods:
#
#    T1  the estimate with standard errors clustered by calendar quarter
#    T2  calendar-time, market-adjusted alpha: an equal-weighted portfolio
#        of open positions, monthly returns regressed on SPY, Newey-West
#
#  Both are the same functions the PEAD re-check used
#  (research/inference_check.py), so the two signals are judged alike.
#
#  This is a rehearsal, not the test. Development data has been looked at
#  many times; passing here only says the exam is worth spending. The exam
#  is the S&P 400/600 releases, once, after a pre-registration.
#
#  The rule tested: buy the releases whose predicted reaction is in the
#  top TOP of the model's training distribution, at the next close after
#  the reaction is complete, hold HOLD sessions.
#
#    python research/v3_gate_check.py
# ═══════════════════════════════════════════════════════════════════════

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from trader.backtest import load_market  # noqa: E402
from trader.config import load_config  # noqa: E402
from inference_check import calendar_time, daily_returns, market_alpha, quarter_clustered  # noqa: E402
from v3_readers_eval import N_FOLDS, out_of_fold  # noqa: E402
from v3_signal_probe import BASE, load  # noqa: E402

V3_DB = ROOT / "data" / "v3.db"
HOLD  = 60
TOP   = 0.95
COST  = 10 / 1e4          # one side, as the S&P 500 event gate assumed


def main():
    cfg = load_config()
    d = load(cfg)
    text_cols = [c for c in d.columns if any(c.startswith(p) for p in
                 ("distilroberta_fin", "finbert", "minilm", "spread", "lm_tone"))]
    d["score"] = out_of_fold(d, d[BASE + text_cols].to_numpy(float), d.reaction.to_numpy())
    d["pctl"] = d.score.rank(pct=True)
    market = load_market(cfg)
    closes = market.closes.ffill()
    col = {s: i for i, s in enumerate(closes.columns)}
    n = len(market.cal)

    picks = d[(d.pctl >= TOP) & (d.gap_entry_idx + HOLD < n)].copy()
    picks["entry_date"] = pd.to_datetime(picks.gap_entry_date)
    px = closes.to_numpy()
    g, c, b = picks.gap_entry_idx.to_numpy(), picks.symbol.map(col).to_numpy(), col[cfg.benchmark]
    picks["excess"] = ((px[g + HOLD, c] / px[g, c] - 1) - (px[g + HOLD, b] / px[g, b] - 1))
    picks = picks[np.isfinite(picks.excess)]
    print(f"{len(picks):,} development trades: top {round((1 - TOP) * 100)}% of the text score, "
          f"held {HOLD} sessions from the next close ({picks.entry_date.min():%Y-%m-%d} → "
          f"{picks.entry_date.max():%Y-%m-%d})")

    means = (picks.excess - COST).groupby(picks.entry_date).mean()
    from scipy import stats
    m0, p0 = float(means.mean()), float(stats.ttest_1samp(means, 0).pvalue)
    print(f"\n   date-clustered (the old, too-lenient test) {m0 * 100:+.3f}pp  p={p0:.4f}")
    m1, p1, q = quarter_clustered(means)
    print(f"   T1 quarter-clustered                       {m1 * 100:+.3f}pp  p={p1:.4f}  "
          f"({q} quarters)")

    monthly = calendar_time(picks.gap_entry_idx.to_numpy() - 1,      # calendar_time enters at idx+1
                            picks.symbol.map(col).to_numpy(),
                            np.full(len(picks), b), daily_returns(closes), closes.index, COST)
    a, p2, beta, months = market_alpha(monthly)
    print(f"   T2 calendar-time alpha vs SPY              {a * 100:+.2f}%/yr  p={p2:.4f}  "
          f"beta {beta:.2f}  ({months} months)")

    passes = m1 > 0 and p1 < 0.05 and a > 0 and p2 < 0.05
    print(f"\n   → on development data the rule {'CLEARS' if passes else 'does NOT clear'} "
          f"both gate tests")
    print("   Development data, many looks: this says only whether the one-shot exam is "
          "worth spending.")


if __name__ == "__main__":
    main()
