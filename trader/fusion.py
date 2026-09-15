# ═══════════════════════════════════════════════════════════════════════
#  Layer 4 — fusion: validated signals → one conviction per event.
#
#  With one signal this is a pass-through. With several it is a weighted
#  mean of the signals that have an opinion, plus an `agree` flag (every
#  opinion on the same side of 0.5). Acting on conflict — the original
#  thesis — is deliberately NOT implemented: that rule has to pass the
#  gate like any other, and it can't be tested until a second signal has.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


class Fusion:
    def __init__(self, signals, weights=None):
        self.signals = list(signals)
        self.weights = weights or {s.name: 1.0 for s in self.signals}

    def score(self, events):
        scores = pd.DataFrame({s.name: s.score(events) for s in self.signals}, index=events.index)
        w = pd.Series(self.weights, dtype=float)[scores.columns]
        have = scores.notna()
        wsum = have.mul(w, axis=1).sum(axis=1)
        conviction = scores.fillna(0).mul(w, axis=1).sum(axis=1) / wsum.replace(0, np.nan)
        n_have = have.sum(axis=1)
        n_high = (scores > 0.5).sum(axis=1)
        out = scores.add_prefix("score_")
        out["conviction"] = conviction
        out["agree"] = (n_high == 0) | (n_high == n_have)
        out["n_signals"] = n_have
        return out
