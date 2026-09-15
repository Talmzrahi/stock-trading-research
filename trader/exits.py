# ═══════════════════════════════════════════════════════════════════════
#  Exit rules — independent of the entry signal.
#
#  Each rule sees a position and closes through t-1 (all a pre-close run
#  can see) and says whether to sell at close t. Rules are checked in
#  order; the first that fires names the exit. New rules plug in here and
#  must beat the plain time exit in research/portfolio_gate.py first.
# ═══════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

STALE_SESSIONS = 5


@dataclass
class Position:
    symbol: str
    event_key: str
    entry_idx: int
    exit_due_idx: int
    conviction: float
    late_days: int = 0
    qty: float = 0.0
    entry_fill: float = float("nan")
    resets: int = 0


class ExitRule(ABC):
    name: str

    def prepare(self, closes):
        """Precompute anything derived from the full close matrix."""

    @abstractmethod
    def check(self, pos, t, market) -> bool:
        ...


class NoDataExit(ExitRule):
    """Price series stopped (acquired, delisted, bad ticker): get out at the
    last known price rather than hold a position nobody can value."""
    name = "no_data"

    def check(self, pos, t, market):
        lo = t - STALE_SESSIONS
        if lo <= pos.entry_idx:
            return False
        return bool(np.isnan(market.px[lo:t, market.col[pos.symbol]]).all())


class TimeExit(ExitRule):
    name = "time"

    def check(self, pos, t, market):
        return t >= pos.exit_due_idx


class VolTrailingStop(ExitRule):
    """Sell when the close falls more than k daily standard deviations below
    the highest close since entry. Scales to each stock's own volatility."""
    name = "stop"

    def __init__(self, k, window=20):
        self.k = k
        self.window = window
        self.vol = None

    def prepare(self, closes):
        logret = np.log(closes).diff()
        self.vol = logret.rolling(self.window, min_periods=self.window // 2).std().to_numpy()

    def check(self, pos, t, market):
        if t - 1 <= pos.entry_idx:
            return False
        c = market.col[pos.symbol]
        path = market.px[pos.entry_idx:t, c]
        last, vol = path[-1], self.vol[t - 1, c]
        if np.isnan(last) or not vol > 0:
            return False
        return last / np.nanmax(path) - 1 < -self.k * vol


def rules_for(cfg):
    rules = [NoDataExit(), TimeExit()]
    if cfg.stop_k:
        rules.append(VolTrailingStop(cfg.stop_k, cfg.stop_vol_window))
    return rules
