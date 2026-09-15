# ═══════════════════════════════════════════════════════════════════════
#  Strategy parameters.
#
#  Defaults are the validated baseline. research/portfolio_gate.py writes
#  the gate-selected values, with the evidence behind them, to
#  config/strategy.json; the backtest and the live run both load that
#  file, so the rule that passed is the rule that trades.
# ═══════════════════════════════════════════════════════════════════════

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
STRATEGY_FILE = ROOT / "config" / "strategy.json"


@dataclass(frozen=True)
class Config:
    # Signal / entry
    cutoff: float = 0.90              # enter when trailing SUE percentile >= cutoff
    window_days: int = 365            # trailing ranking window
    min_history: int = 200

    # Exits
    hold_days: int = 60               # trading days; a re-trigger resets the clock
    stop_k: float | None = None       # vol trailing stop, multiple of daily sd; None = off
    stop_vol_window: int = 20

    # Portfolio
    slot_mult: float = 1.0            # slots = expected concurrent holdings x slot_mult
    min_slots: int = 5
    cash_buffer: float = 0.005        # equity fraction kept as cash for fill drift
    min_order: float = 1.0            # dollars — Alpaca's fractional minimum

    # Costs
    cost_bps_side: float = 5.0        # 10bps round trip on stocks
    benchmark_cost_bps_side: float = 1.0

    # Live
    max_late_days: int = 2
    initial_capital: float = 1000.0
    benchmark: str = "SPY"


def load_config(path=STRATEGY_FILE):
    if not Path(path).exists():
        return Config()
    params = json.loads(Path(path).read_text(encoding="utf-8")).get("params", {})
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in params.items() if k in known})


def save_strategy(cfg, evidence, path=STRATEGY_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": asdict(cfg), "evidence": evidence}, indent=2),
                    encoding="utf-8")
