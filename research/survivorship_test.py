# ═══════════════════════════════════════════════════════════════════════
#  How much does the missing-delisted-firms hole matter?
#
#  yfinance drops dead tickers, so 207 S&P 500 and 543 mid/small firms that
#  were once index members have no data at all. The usual worry is that
#  they are the losers, flattering every result. But firms leave an index
#  for two opposite reasons — acquisition (deals close at a premium, so
#  those are missing WINNERS) and distress or demotion (missing LOSERS) —
#  so the direction has to be measured, not assumed.
#
#  This is a diagnostic, not a selection rule: nothing here changes
#  config/strategy.json.
#
#  1. WHY the missing firms left, from Wikipedia's index-change history
#  2. HOW MUCH member-time is unobservable, and the trades that implies
#  3. A BOUND on the backtest result under explicit assumptions
#  4. FRAGILITY: does the edge come from firms that were about to leave the
#     index anyway? If so the hole matters more, AND the strategy is
#     leaning on companies in trouble.
#
#    python research/survivorship_test.py
# ═══════════════════════════════════════════════════════════════════════

import re
import sqlite3
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from trader.backtest import load_market, run  # noqa: E402
from trader.config import load_config  # noqa: E402
from trader.data import RESEARCH_DB  # noqa: E402

PAGES = ["Historical_components_of_the_S%26P_500",
         "List_of_S%26P_400_companies",
         "List_of_S%26P_600_companies"]

HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (index research script)"})

# Excess return assumed for one unobservable trade, by why the firm left.
ASSUMED = {"acquired": 0.15, "demoted": -0.15, "failed": -0.50,
           "restructured": 0.0, "unknown": -0.15}

FAILED = re.compile(r"bankrupt|chapter 11|liquidat|receivership|delisted|ceased", re.I)
ACQUIRED = re.compile(r"acquir|merg|bought|purchas|takeover|acquisition|taken private", re.I)
DEMOTED = re.compile(r"market cap|no longer representative|not representative|ranked|"
                     r"midcap|smallcap|s&p 400|s&p 600|moved to|removed from", re.I)
RESTRUCTURED = re.compile(r"spun off|spin-?off|split into|separat", re.I)


def classify(reason):
    r = str(reason)
    for label, pattern in (("failed", FAILED), ("acquired", ACQUIRED),
                           ("demoted", DEMOTED), ("restructured", RESTRUCTURED)):
        if pattern.search(r):
            return label
    return "unknown"


def removal_reasons():
    """Every index removal Wikipedia records a reason for, across the 500/400/600."""
    rows = []
    for page in PAGES:
        html = HTTP.get(f"https://en.wikipedia.org/wiki/{page}", timeout=30).text
        for table in pd.read_html(StringIO(html)):
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c)
                    for c in table.columns]
            if not (any("Removed" in c for c in cols) and any("Reason" in c for c in cols)):
                continue
            table.columns = cols
            tick = next(c for c in cols if "Removed" in c and "Ticker" in c)
            why = next(c for c in cols if "Reason" in c)
            sub = table[[tick, why]].dropna(subset=[tick])
            sub.columns = ["symbol", "reason"]
            rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["symbol"] = (df.symbol.astype(str).str.strip().str.upper()
                    .str.replace(".", "-", regex=False)
                    .str.replace(r"\[.*\]", "", regex=True))
    df["class"] = df.reason.map(classify)
    return df.drop_duplicates("symbol", keep="first").set_index("symbol")


def clustered_p(group):
    """Same date-clustered test the gates use."""
    m = group.groupby("entry_date").alpha.mean()
    if len(m) < 5:
        return float("nan")
    return float(stats.ttest_1samp(m, 0).pvalue)


def main():
    cfg = load_config()
    reasons = removal_reasons()
    print(f"Index-change history: {len(reasons):,} removals with a stated reason\n")

    conn = sqlite3.connect(f"file:{RESEARCH_DB}?mode=ro", uri=True)
    cov = pd.read_sql_query("SELECT * FROM departed_coverage", conn)
    uni = pd.read_sql_query("SELECT as_of, symbol FROM universe_history", conn)
    conn.close()

    missing = cov[cov.status != "kept"].copy()
    missing["class"] = missing.symbol.map(reasons["class"]).fillna("unknown")

    print("── 1. Why the missing S&P 500 firms left the index ────────────")
    counts = missing["class"].value_counts()
    for label, n in counts.items():
        example = ", ".join(missing.symbol[missing["class"] == label].head(6))
        print(f"   {label:<13}{n:>4}  ({n / len(missing) * 100:>4.1f}%)   e.g. {example}")
    acquired = counts.get("acquired", 0) / len(missing)
    print(f"\n   {acquired * 100:.0f}% were acquired — deals close at a premium, so those are "
          f"missing WINNERS,\n   which pushes the bias the opposite way from the usual worry.")

    have = set(cov.symbol[cov.status == "kept"]) | (set(uni.symbol) - set(cov.symbol))
    uni["missing"] = ~uni.symbol.isin(have)
    by_year = uni.groupby(uni.as_of.str[:4]).missing.mean()
    overall = uni.missing.mean()
    print("\n── 2. Share of index member-quarters with no data ─────────────")
    print("   " + "  ".join(f"{y}:{v * 100:.0f}%" for y, v in by_year.items()))
    print(f"   overall {overall * 100:.1f}% of member-time is unobservable")

    print("\n── 3. Bounding the effect on the backtest ─────────────────────")
    print("   running the live configuration over the repaired universe …", flush=True)
    market = load_market(cfg)
    res = run(market, cfg)
    trades = res.trades
    observed, n_obs = trades.alpha.mean(), len(trades)
    years = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
    n_missing = n_obs * overall / (1 - overall)
    mix = float((missing["class"].value_counts(normalize=True)
                 * pd.Series(ASSUMED)).dropna().sum())

    print(f"   {n_obs:,} observed trades over {years:.1f} years; the unobservable firms "
          f"imply ~{n_missing:,.0f} more")
    # One position is roughly `weight` of equity, so a change in mean alpha per
    # trade costs (missing trades per year) x weight at the portfolio level.
    weight = float((res.equity.stocks / res.equity.equity).mean() /
                   max(res.equity.n_positions.mean(), 1))
    per_year = n_missing / years
    print(f"   each position is ~{weight * 100:.1f}% of equity; "
          f"~{per_year:.0f} unobservable trades a year")
    # The most defensible estimate is empirical: firms that DID leave the index
    # but kept trading are the closest analogue to the ones that vanished.
    last_seen = uni.groupby("symbol").as_of.max()
    leaving = trades[pd.to_datetime(trades.symbol.map(last_seen))
                     < trades.entry_date + pd.Timedelta(days=365)]
    analogue = float(leaving.alpha.mean())

    print(f"   {'assumption':<38}{'mean alpha/trade':>20}{'portfolio drag':>16}")
    for label, assumed in ((f"like firms that did leave ({analogue*100:+.2f}pp)", analogue),
                           ("mix by stated reason", mix),
                           ("every missing trade -25%", -0.25),
                           ("every missing trade -50%", -0.50)):
        blended = (observed * n_obs + assumed * n_missing) / (n_obs + n_missing)
        drag = (assumed - observed) * per_year * weight
        print(f"   {label:<38}{observed * 100:+8.2f} -> {blended * 100:+.2f}pp"
              f"{drag * 100:>14.2f}pp/yr")

    print("\n── 4. Does the edge come from firms about to leave the index? ──")
    last_seen = uni.groupby("symbol").as_of.max()
    trades = trades.copy()
    trades["last_member"] = pd.to_datetime(trades.symbol.map(last_seen))
    trades["leaving"] = trades.last_member < trades.entry_date + pd.Timedelta(days=365)
    for label, group in (("left the index within a year", trades[trades.leaving]),
                         ("stayed in the index", trades[~trades.leaving])):
        if len(group) >= 5:
            print(f"   {label:<30} n={len(group):>5,}  "
                  f"mean alpha {group.alpha.mean() * 100:+.2f}pp  p={clustered_p(group):.3f}")
    if trades.leaving.any() and (~trades.leaving).any():
        a, b = trades.alpha[trades.leaving], trades.alpha[~trades.leaving]
        _, p = stats.ttest_ind(a, b, equal_var=False)
        print(f"   difference {(a.mean() - b.mean()) * 100:+.2f}pp (p={p:.3f}); "
              f"{trades.leaving.mean() * 100:.1f}% of trades were in firms on their way out")

    closes = market.closes
    vol = np.log(closes).diff().rolling(60, min_periods=30).std()
    col = {s: i for i, s in enumerate(closes.columns)}
    rows = market.cal.searchsorted(trades.entry_date.to_numpy())
    cols = trades.symbol.map(col).to_numpy()
    trades["vol"] = vol.to_numpy()[rows, cols]
    trades["price"] = closes.to_numpy()[rows, cols]
    print()
    for name, field in (("volatility at entry", "vol"), ("share price at entry", "price")):
        print(f"   alpha by {name}:")
        band = pd.qcut(trades[field], 3, labels=["low", "mid", "high"])
        for level, group in trades.groupby(band, observed=True):
            print(f"      {str(level):<5} n={len(group):>5,}  "
                  f"{group.alpha.mean() * 100:+.2f}pp  p={clustered_p(group):.3f}")

    # ── 5. Is the volatile-stock edge the signal, or just beta? ───────
    # Excess return is measured against SPY, so high-beta names beat it in a
    # rising market whether or not they beat earnings estimates. The control
    # is other events in equally volatile stocks.
    print()
    print("── 5. Signal, or just high-beta stocks in a bull market? ───────")
    ev = market.events
    ev = ev[ev.pit & ev.conviction.notna()].copy()
    ev = ev[ev.entry_idx + cfg.hold_days < len(market.cal)]
    ff = closes.ffill().to_numpy()
    bench = col[cfg.benchmark]
    e = ev.entry_idx.to_numpy()
    x = e + cfg.hold_days
    c = ev.symbol.map(col).to_numpy()
    ev["excess"] = (ff[x, c] / ff[e, c] - 1) - (ff[x, bench] / ff[e, bench] - 1)
    ev["vol"] = vol.to_numpy()[e, c]
    ev["entry_date"] = market.cal[e]
    ev = ev[np.isfinite(ev.excess) & np.isfinite(ev.vol)]
    ev["band"] = pd.qcut(ev.vol, 3, labels=["low", "mid", "high"])
    ev["signal"] = ev.conviction >= cfg.cutoff
    print(f"   {'volatility':<12}{'top 5% events':>26}{'all other events':>26}{'difference':>14}")
    for level, group in ev.groupby("band", observed=True):
        hit, miss = group[group.signal], group[~group.signal]
        diff = hit.excess.mean() - miss.excess.mean()
        _, p = stats.ttest_ind(hit.excess, miss.excess, equal_var=False)
        print(f"   {str(level):<12}{hit.excess.mean() * 100:+10.2f}pp (n={len(hit):>4,})"
              f"{miss.excess.mean() * 100:+12.2f}pp (n={len(miss):>6,})"
              f"{diff * 100:+10.2f}pp (p={p:.3f})")
    print("   the middle column is the beta effect; the difference is what the signal adds")


if __name__ == "__main__":
    main()
