# ═══════════════════════════════════════════════════════════════════════
#  The text signal.
#
#  It reads a company's earnings release, keeps only what is new against
#  that company's own previous releases, and predicts the announcement
#  reaction from that text plus the earnings surprise. The score is where
#  that prediction sits in the training distribution, so "top 5%" means
#  on day one what it meant in the fit.
#
#  This signal does NOT trade the live account. It runs in shadow mode —
#  its own simulated account, its own record — until it passes the gate
#  in CLAUDE.md. On the development data the prediction showed only
#  borderline continuation (+0.95pp top-minus-bottom fifth at 60
#  sessions, p=0.07), which is worth watching live and not worth trading.
#
#  Scoring needs the filing, so the daily run does the reading and hands
#  the finished scores in. That keeps this class pure: events in, scores
#  out, no network, no models, and the same object serves a backtest fed
#  from stored scores.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

from .base import Signal


class TextSignal(Signal):
    """Percentile of the model's predicted reaction, by event key."""

    name = "text"

    def __init__(self, scores=None):
        self.scores = dict(scores or {})

    def score(self, events):
        out = events.key.map(self.scores).astype(float) if len(events) else pd.Series(dtype=float)
        return pd.Series(np.asarray(out, dtype=float), index=events.index, name=self.name)
