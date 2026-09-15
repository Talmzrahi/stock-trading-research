# ═══════════════════════════════════════════════════════════════════════
#  Layer 7 — monitoring: the "signal has stopped working" tripwire.
#
#  The band is built from backtest trades: if live trades were drawn like
#  historical ones, where would the running mean alpha after N trades
#  fall? Below the 5th percentile, the tripwire fires. It only alerts —
#  early live samples are noisy and the decision stays with a human.
#
#  Caveat: the band comes from the same history the strategy was selected
#  on, so it is if anything too optimistic — it will fire a little early,
#  which is the safe direction for a tripwire.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

MIN_TRADES = 10
N_MAX      = 500


def bootstrap_band(trades, n_max=N_MAX, draws=5000, q=0.05, seed=42):
    """Percentiles of the running mean alpha after N trades.

    Resamples whole entry dates, not trades: positions entered on the same
    day share a market shock, and treating them as independent would make
    the band too narrow.
    """
    groups = [g.to_numpy() for _, g in trades.groupby("entry_date")["alpha"]]
    sizes = np.array([len(g) for g in groups])
    flat = np.concatenate(groups)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(seed)
    denom = np.arange(1, n_max + 1)

    means = np.empty((draws, n_max))
    for b in range(draws):
        picks = rng.integers(len(groups), size=n_max)     # every group has >= 1 trade
        x = np.concatenate([flat[offs[i]:offs[i + 1]] for i in picks])[:n_max]
        means[b] = np.cumsum(x) / denom
    return pd.DataFrame({"n": denom,
                         "p05": np.quantile(means, q, axis=0),
                         "p50": np.median(means, axis=0),
                         "p95": np.quantile(means, 1 - q, axis=0)})


def band_row(band, n):
    return band.iloc[min(n, len(band)) - 1]


def tripwire_status(alphas, band, min_trades=MIN_TRADES):
    n = len(alphas)
    if n < min_trades:
        return {"state": "warming_up", "n": n, "needed": min_trades}
    row = band_row(band, n)
    mean = float(np.mean(alphas))
    return {"state": "FIRED" if mean < row.p05 else "ok", "n": n, "mean_alpha": mean,
            "p05": float(row.p05), "p50": float(row.p50)}


def first_fire(alphas, band, min_trades=MIN_TRADES):
    """1-based trade count at which the running mean first drops below the
    band, or None."""
    x = np.asarray(alphas, dtype=float)
    running = np.cumsum(x) / np.arange(1, len(x) + 1)
    for n in range(min_trades, len(x) + 1):
        if running[n - 1] < band_row(band, n).p05:
            return n
    return None
