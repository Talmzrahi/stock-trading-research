# ═══════════════════════════════════════════════════════════════════════
#  Portfolio backtest — the live engine, replayed over history.
#
#  Every session: Engine.step with closes through t-1, fill at close t in
#  an in-memory Ledger, book the fills. Nothing here decides anything the
#  live run doesn't; the only difference is that history has no missed
#  days, so late entries never occur.
# ═══════════════════════════════════════════════════════════════════════

import math
from dataclasses import dataclass

import pandas as pd

from .broker import Ledger
from .data import connect, load_closes, load_earnings, load_universe
from .engine import Engine, EngineState
from .events import build_events
from .fusion import Fusion
from .market_calendar import trading_days
from .signals.sue import SueSignal

ERAS = [("2010-2013", None, "2013-12-31"),
        ("2014-2019", "2014-01-01", "2019-12-31"),
        ("2020-2026", "2020-01-01", None)]


@dataclass
class Market:
    cal: pd.DatetimeIndex
    closes: pd.DataFrame
    events: pd.DataFrame


@dataclass
class Result:
    equity: pd.DataFrame
    trades: pd.DataFrame
    decisions: list


def signal_stack(cfg):
    return Fusion([SueSignal(cfg.window_days, cfg.min_history)])


def market_from(cfg, cal, closes, earn, universe):
    events = build_events(earn, cal, closes, universe)
    return Market(cal, closes, events.join(signal_stack(cfg).score(events)))


def load_market(cfg, conn=None, end=None):
    own = conn is None
    conn = conn or connect()
    try:
        closes, earn, uni = load_closes(conn), load_earnings(conn), load_universe(conn)
    finally:
        if own:
            conn.close()
    cal = trading_days(closes.index.min(), end or closes.index.max())
    return market_from(cfg, cal, closes.reindex(cal), earn, uni)


def cost_model(cfg):
    return lambda s: cfg.benchmark_cost_bps_side if s == cfg.benchmark else cfg.cost_bps_side


def run(market, cfg, start=None, end=None, keep_decisions=False):
    eng = Engine(cfg, market.cal, market.closes, market.events)
    ledger = Ledger(cfg.initial_capital, cost_model(cfg), cfg.min_order, keep_history=False)
    state = EngineState()
    col, bench = eng.col, cfg.benchmark
    t0 = eng.default_start() if start is None else start
    t1 = len(market.cal) - 1 if end is None else end

    rows, decisions = [], []
    for t in range(t0, t1 + 1):
        prev, cur = eng.ff[t - 1], eng.ff[t]
        d = eng.step(t, state, ledger, lambda s: prev[col[s]] if s in col else None)
        if keep_decisions:
            decisions.extend(d)
        fills = ledger.settle(eng.dates[t], lambda s: cur[col[s]] if s in col else None)
        eng.on_fills(t, fills, state)

        stocks = bench_v = 0.0
        for s, q in ledger.holdings().items():
            if s == bench:
                bench_v += q * cur[col[s]]
            else:
                stocks += q * cur[col[s]]
        rows.append((market.cal[t], ledger.cash(), stocks, bench_v, len(state.positions)))

    eq = pd.DataFrame(rows, columns=["date", "cash", "stocks", "bench", "n_positions"]).set_index("date")
    eq["equity"] = eq.cash + eq.stocks + eq.bench
    return Result(eq, pd.DataFrame([vars(x) for x in state.trades]), decisions)


def cagr(series):
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 and s.iloc[0] > 0 else float("nan")


def summarize(res, market, cfg):
    eq = res.equity.equity
    r = eq.pct_change().dropna()
    bench = market.closes[cfg.benchmark].reindex(eq.index)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    tr = res.trades
    out = {
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": cagr(eq),
        "bench_cagr": cagr(bench),
        "vol": float(r.std() * math.sqrt(252)),
        "sharpe": float(r.mean() / r.std() * math.sqrt(252)) if r.std() > 0 else float("nan"),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "bench_max_drawdown": float((bench / bench.cummax() - 1).min()),
        "avg_stock_exposure": float((res.equity.stocks / eq).mean()),
        "avg_positions": float(res.equity.n_positions.mean()),
        "max_positions": int(res.equity.n_positions.max()),
        "trades": len(tr),
        "trades_per_year": len(tr) / years,
        "mean_alpha": float(tr.alpha.mean()) if len(tr) else float("nan"),
        "hit_rate": float((tr.alpha > 0).mean()) if len(tr) else float("nan"),
        "exit_reasons": tr.reason.value_counts().to_dict() if len(tr) else {},
    }
    for name, a, b in ERAS:
        out[f"cagr_{name}"] = cagr(eq.loc[a:b])
        out[f"bench_cagr_{name}"] = cagr(bench.loc[a:b])
    return out
