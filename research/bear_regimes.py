# ======================================================================
#  Is volatility targeting a bear-market tool you can actually switch on?
#
#  POST-HOC ON SPENT DATA. This answers a question; it is not a test.
#
#  Targeting clearly earns its keep in bear markets -- the one surviving
#  pass is ~70% the Great Depression. But "bear market" is obvious only in
#  hindsight. The useful question is whether a signal knowable AT THE TIME
#  identifies the stretches where targeting pays, early enough to matter.
#
#  Three definitions, compared on the same 61 series:
#    hindsight  every drawdown deeper than 20%, from peak to trough.
#               Uses the future (you only know the trough afterwards).
#    below MA   price below its 200-day average at yesterday's close.
#    in 20% DD  already down 20% from the running peak at yesterday's close.
#
#  And the rule the question implies: hold buy-and-hold normally, switch
#  to the volatility-targeted book only while the real-time signal says
#  "bear".
#
#    .venv\Scripts\python.exe research\bear_regimes.py
# ======================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from mechanics import performance  # noqa: E402
from voltarget_synthesis import pair, px, samples  # noqa: E402


def flags(bh):
    eq = (1 + bh).cumprod()
    dd = eq / eq.cummax() - 1
    hind = pd.Series(False, index=bh.index)
    under = dd < 0
    seg = (~under).cumsum()
    episodes = []
    for _, g in dd[under].groupby(seg[under]):
        if g.min() < -0.20:
            start, trough = g.index[0], g.idxmin()
            hind.loc[start:trough] = True
            episodes.append((start, trough))
    below_ma = (eq < eq.rolling(200).mean()).shift(1, fill_value=False)
    in_dd = (dd < -0.20).shift(1, fill_value=False)
    return eq, hind, below_ma.astype(bool), in_dd.astype(bool), episodes


def late_by(eq, flag, episodes):
    """For each real bear market: how much of the eventual fall had already
    happened by the day the real-time signal first switched on."""
    out = []
    for start, trough in episodes:
        peak_val = eq.loc[:start].iloc[-2] if len(eq.loc[:start]) > 1 else eq.loc[start]
        total = eq.loc[trough] / peak_val - 1
        on = flag.loc[start:trough]
        if not on.any() or total >= 0:
            out.append(1.0)                 # never switched on before the bottom
            continue
        first = on.idxmax()
        out.append((eq.loc[first] / peak_val - 1) / total)
    return out


def main():
    rows, lateness_ma, lateness_dd = [], [], []
    for test, verdict, sym, name, lo, hi, q in samples():
        bh, tg, rf = pair(px(sym), lo, hi, q)
        if len(bh) < 500:
            continue
        eq, hind, ma, dd20, eps = flags(bh)
        diff = tg - bh
        sw_ma = pd.Series(np.where(ma, tg, bh), index=bh.index)
        sw_dd = pd.Series(np.where(dd20, tg, bh), index=bh.index)
        B, T = performance(bh, rf), performance(tg, rf)
        S1, S2 = performance(sw_ma, rf), performance(sw_dd, rf)
        rec = lambda f: (f & hind).sum() / max(hind.sum(), 1)
        prec = lambda f: (f & hind).sum() / max(f.sum(), 1)
        rows.append(dict(
            test=test,
            edge_hind=diff[hind].mean() * 252 if hind.any() else np.nan,
            edge_ma=diff[ma].mean() * 252 if ma.any() else np.nan,
            edge_dd=diff[dd20].mean() * 252 if dd20.any() else np.nan,
            edge_bull=diff[~hind].mean() * 252,
            rec_ma=rec(ma), prec_ma=prec(ma), rec_dd=rec(dd20), prec_dd=prec(dd20),
            sh_t=T["sharpe"] - B["sharpe"], sh_ma=S1["sharpe"] - B["sharpe"],
            sh_dd=S2["sharpe"] - B["sharpe"],
            dd_b=B["max_dd"], dd_t=T["max_dd"], dd_ma=S1["max_dd"], dd_dd=S2["max_dd"],
            r_b=B["ret"], r_t=T["ret"], r_ma=S1["ret"], r_dd=S2["ret"]))
        lateness_ma += late_by(eq, ma, eps)
        lateness_dd += late_by(eq, dd20, eps)
    R = pd.DataFrame(rows)
    med = R.median(numeric_only=True)

    print(f"{len(R)} series. POST-HOC on spent data.\n")
    print("-- 1. Where does targeting earn its money? (annualised, targeted minus buy-and-hold) --")
    print(f"   during HINDSIGHT bear markets (peak to trough)   {med.edge_hind*100:+7.1f}%/yr")
    print(f"   when price was below its 200-day average         {med.edge_ma*100:+7.1f}%/yr")
    print(f"   when already down 20% from the peak              {med.edge_dd*100:+7.1f}%/yr")
    print(f"   everything outside hindsight bear markets        {med.edge_bull*100:+7.1f}%/yr")

    print("\n-- 2. How late do real-time signals see a bear market? --")
    print(f"   share of the eventual fall ALREADY GONE when the signal switched on (median):")
    print(f"      below 200-day average   {np.median(lateness_ma)*100:5.0f}%   "
          f"({len(lateness_ma)} bear markets)")
    print(f"      down 20% from peak      {np.median(lateness_dd)*100:5.0f}%")
    print(f"   of the days each signal calls 'bear', the share that really were falling:")
    print(f"      below 200-day average   {med.prec_ma*100:5.0f}%   "
          f"(and it catches {med.rec_ma*100:.0f}% of the falling days)")
    print(f"      down 20% from peak      {med.prec_dd*100:5.0f}%   "
          f"(and it catches {med.rec_dd*100:.0f}% of the falling days)")

    print("\n-- 3. The rule the question implies: buy-and-hold, switch to targeting in a bear --")
    print(f"   {'':<40}{'return':>9}{'dExSharpe':>11}{'max DD':>9}")
    print(f"   {'buy and hold':<40}{med.r_b*100:>8.2f}%{0:>+11.3f}{med.dd_b*100:>8.0f}%")
    print(f"   {'always targeted':<40}{med.r_t*100:>8.2f}%{med.sh_t:>+11.3f}{med.dd_t*100:>8.0f}%")
    print(f"   {'targeted only below 200-day average':<40}{med.r_ma*100:>8.2f}%"
          f"{med.sh_ma:>+11.3f}{med.dd_ma*100:>8.0f}%")
    print(f"   {'targeted only when down 20%':<40}{med.r_dd*100:>8.2f}%"
          f"{med.sh_dd:>+11.3f}{med.dd_dd*100:>8.0f}%")
    print(f"\n   (medians across series; each column's median is taken separately)")


if __name__ == "__main__":
    main()
