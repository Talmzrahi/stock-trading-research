# ═══════════════════════════════════════════════════════════════════════
#  Daily report — reports/YYYY-MM-DD.md. What fired, why, what's held,
#  and whether the signal still looks alive.
# ═══════════════════════════════════════════════════════════════════════

import math

ACTIONS = {
    "enter":             "BUY",
    "exit_time":         "SELL — hold period complete",
    "exit_stop":         "SELL — volatility trailing stop",
    "exit_no_data":      "SELL — price data stopped",
    "reset_clock":       "HOLD — new top signal restarted the clock",
    "too_late":          "SKIP — signal seen too late",
    "no_cash":           "SKIP — no cash left today",
    "retrigger_ignored": "SKIP — position already exiting",
}


def pct(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


def money(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:,.2f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def render(ctx):
    cfg, acct = ctx["cfg"], ctx["account"]
    stop = "off" if not cfg.stop_k else f"{cfg.stop_k:g} daily sd"
    L = [f"# Daily run — {ctx['today']}" + ("  (DRY RUN — nothing saved)" if ctx["dry_run"] else ""),
         "",
         f"_Run {ctx['now']:%Y-%m-%d %H:%M} ET · simulated broker · entry: top "
         f"{round((1 - cfg.cutoff) * 100)}% SUE surprise · hold {cfg.hold_days} sessions · "
         f"trailing stop {stop}_",
         "", f"**{ctx['status']}**", ""]

    trip = ctx["tripwire"]
    if trip.get("state") == "FIRED":
        L += ["> ## 🚨 TRIPWIRE FIRED",
              f"> Mean alpha over the last {trip['n']} closed trades is {pct(trip['mean_alpha'])}, "
              f"below the backtest's 5th-percentile band ({pct(trip['p05'])}). The signal may have "
              "stopped working. Nothing was changed automatically — review before the next run.", ""]

    if ctx["warnings"]:
        L += ["## ⚠ Warnings", ""] + [f"- {w}" for w in ctx["warnings"]] + [""]

    L += ["## Account", ""]
    L += table(["", ""], [
        ["Equity", money(acct["equity"])],
        ["Since inception", f"{pct(acct['ret_since'])} (SPY {pct(acct['spy_since'])}) — started "
                            f"{acct['inception']} with {money(acct['initial'])}"],
        ["Stocks", f"{money(acct['stocks'])} ({pct(acct['exposure'], 0)} of equity, "
                   f"{len(ctx['positions'])} positions)"],
        ["SPY (idle cash)", money(acct["bench"])],
        ["Cash", money(acct["cash"])],
        ["Prices as of", acct["prices_as_of"]],
    ])
    L.append("")

    acts = [d for d in ctx["decisions"] if d.action != "below_cutoff"]
    below = sorted((d for d in ctx["decisions"] if d.action == "below_cutoff"),
                   key=lambda d: -d.conviction)
    L += ["## Today's decisions", ""]
    if acts:
        L += table(["Action", "Symbol", "Surprise percentile", "Detail"],
                   [[ACTIONS.get(d.action, d.action), d.symbol, f"{d.conviction:.3f}", d.detail]
                    for d in acts])
    else:
        L.append("No entries or exits today.")
    if below:
        top = ", ".join(f"{d.symbol} ({d.conviction:.3f})" for d in below[:5])
        L += ["", f"{len(below)} new report(s) below the {cfg.cutoff:.2f} cutoff — highest: {top}."]
    L.append("")

    if ctx["submitted"]:
        L += ["## Orders submitted (market-on-close)", ""]
        L += table(["Symbol", "Side", "Amount", "Reason"],
                   [[o.symbol, o.side.upper(), money(o.notional) if o.notional else "all shares",
                     o.tag] for o in ctx["submitted"]])
        L.append("")

    if ctx["fills"]:
        L += ["## Fills since last run", ""]
        L += table(["Session", "Symbol", "Side", "Qty", "Price", "Status"],
                   [[o.when, o.symbol, o.side.upper(), f"{o.fill_qty:.4f}" if o.fill_qty else "—",
                     money(o.fill_price), o.status] for o in ctx["fills"]])
        L.append("")

    L += ["## Open positions", ""]
    if ctx["positions"]:
        L += table(["Symbol", "Entered", "Sessions held", "Exit due", "Value", "Return",
                    "vs SPY", "Room to stop"],
                   [[p["symbol"], p["entry"], p["held"], p["due"], money(p["value"]),
                     pct(p["ret"]), pct(p["alpha"]), pct(p["stop_gap"], 1)]
                    for p in ctx["positions"]])
    else:
        L.append("None — all capital is in SPY until a signal fires.")
    L.append("")

    tr = ctx["trades"]
    L += ["## Closed trades", ""]
    if len(tr):
        L.append(f"{len(tr)} closed · mean alpha {pct(tr.alpha.mean())} per trade "
                 f"(backtest {pct(ctx['expected_alpha'])}) · hit rate {pct((tr.alpha > 0).mean(), 0)}")
        L.append("")
        L += table(["Symbol", "Entered", "Exited", "Reason", "Return", "Alpha"],
                   [[r.symbol, f"{r.entry_date:%Y-%m-%d}", f"{r.exit_date:%Y-%m-%d}", r.reason,
                     pct(r.ret), pct(r.alpha)] for r in tr.tail(10).itertuples()])
    else:
        L.append("None yet.")
    L.append("")

    L += ["## Tripwire", ""]
    if trip.get("state") == "warming_up":
        L.append(f"Warming up: {trip['n']} of {trip['needed']} closed trades needed before it can fire. "
                 "At ~24 trades a year this takes months — the backtest band is wide early on.")
    elif trip.get("state") in ("ok", "FIRED"):
        L.append(f"**{trip['state']}** — mean alpha {pct(trip['mean_alpha'])} over {trip['n']} trades; "
                 f"fires below {pct(trip['p05'])}; backtest median at this count {pct(trip['p50'])}.")
    else:
        L.append("No band file — run `python research/portfolio_gate.py`.")
    L.append("")

    up = ctx["upcoming"]
    L += ["## Reporting in the next 7 days", ""]
    if len(up):
        for d, g in up.groupby("date"):
            L.append(f"- **{d:%a %b %d}** ({len(g)}): " + ", ".join(g.symbol.tolist()[:40])
                     + (" …" if len(g) > 40 else ""))
    else:
        L.append("No S&P 500 reports scheduled.")
    L.append("")

    if ctx["notes"]:
        L += ["## Run notes", ""] + [f"- {n}" for n in ctx["notes"]] + [""]
    return "\n".join(L)
