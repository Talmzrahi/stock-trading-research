# ======================================================================
#  What happens if you "try as many things as possible until one clicks"?
#
#  Feed the project's own test strategies that contain NO information and
#  count what it lets through. Two experiments:
#
#  A. Is the one surviving pass real timing? The S&P 500 1928-1992 result
#     (+0.186 excess Sharpe) is compared against the same volatility-
#     targeting weights SHUFFLED across months: identical exposures, the
#     timing destroyed. If shuffled books match it often, it was luck.
#
#  B. A search. 1,000 coin-flip timing rules (each month: hold the market
#     or cash, at random), then 780 "interactions" (every pair of 40 coin
#     flips, held only when both agree). Each is run through a T2-style
#     test: monthly excess return regressed on the market's, Newey-West,
#     pass = positive alpha with p < 0.05. Every pass is a false positive
#     by construction, because nothing here knows anything.
#
#    .venv\Scripts\python.exe research\noise_search.py
# ======================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
from mechanics import hold_from_month_end, month_end_mask, performance  # noqa: E402
from voltarget_synthesis import CAP, COST, LOOK, MARGIN, RF, TARGET, px  # noqa: E402

RNG = np.random.default_rng(20260924)


def blocks(index):
    """Id of the holding block each day belongs to: a weight set at a month
    end applies from the next trading day until the following month end."""
    me = pd.Series(month_end_mask(index), index=index)
    return me.shift(1, fill_value=False).cumsum().to_numpy()


def book(eq, rf, w):
    w = np.asarray(w, dtype=float)
    rate = np.where(w <= 1.0, rf, rf + MARGIN / 252)
    turn = np.abs(np.diff(w, prepend=w[0]))
    return w * eq + (1 - w) * rate - turn * COST


def experiment_a():
    s = px("GSPC").loc["1928":"1992"]
    eq = (s.pct_change().fillna(0.0) + 0.04 / 252)
    rf = RF.reindex(eq.index).ffill().fillna(0.0)
    raw = (TARGET / (eq.rolling(LOOK).std() * np.sqrt(252))).clip(0, CAP)
    w = hold_from_month_end(raw, eq.index)
    ok = w.notna().to_numpy()
    eq, rf, w = eq[ok], rf[ok], w[ok]
    bid = blocks(eq.index)
    per_block = pd.Series(w.to_numpy()).groupby(bid).first().to_numpy()
    base = performance(eq, rf)["sharpe"]
    actual = performance(pd.Series(book(eq.to_numpy(), rf.to_numpy(), w), index=eq.index),
                         rf)["sharpe"] - base
    null = []
    for _ in range(1000):
        shuffled = RNG.permutation(per_block)[bid - bid.min()]
        r = pd.Series(book(eq.to_numpy(), rf.to_numpy(), shuffled), index=eq.index)
        null.append(performance(r, rf)["sharpe"] - base)
    null = np.array(null)
    print("-- A. Is the S&P 1928-1992 pass real timing, or the right exposure at a lucky time? --")
    print(f"   actual volatility targeting     dExSharpe {actual:+.3f}")
    print(f"   same weights, shuffled months   mean {null.mean():+.3f}   "
          f"95th pct {np.percentile(null, 95):+.3f}   best {null.max():+.3f}")
    print(f"   share of shuffled books that match or beat it: {(null >= actual).mean():.1%}")
    return actual, null


def monthly(index, *series):
    m = pd.DatetimeIndex(index).to_period("M")
    return [(1 + pd.Series(x, index=index)).groupby(m).prod() - 1 for x in series]


def t2(port_m, mkt_m, rf_m):
    y = (port_m - rf_m).to_numpy()
    x = sm.add_constant((mkt_m - rf_m).to_numpy())
    f = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return f.params[0] * 12, f.pvalues[0]


def experiment_b():
    s = px("GSPC").loc["1928":]
    eq = (s.pct_change().fillna(0.0) + 0.04 / 252).to_numpy()
    idx = s.index
    rf = RF.reindex(idx).ffill().fillna(0.0).to_numpy()
    bid = blocks(idx)
    nb = bid.max() - bid.min() + 1
    mkt_m, rf_m = monthly(idx, eq, rf)

    def run(on_block):
        h = on_block[bid - bid.min()].astype(float)
        return t2(monthly(idx, h * eq + (1 - h) * rf)[0], mkt_m, rf_m)

    singles = [run(RNG.random(nb) < 0.5) for _ in range(1000)]
    base = [RNG.random(nb) < 0.5 for _ in range(40)]
    pairs = [run(base[i] & base[j]) for i in range(40) for j in range(i + 1, 40)]

    print("\n-- B. 1,000 coin-flip strategies, then 780 'interactions' between 40 of them --")
    print(f"   {len(nb * [0]) // 12:,} years of S&P 500 data. Every rule below is pure noise.\n")
    for lab, res in (("single coin flips", singles), ("pairwise interactions", pairs)):
        a = np.array([x[0] for x in res])
        p = np.array([x[1] for x in res])
        passed = (a > 0) & (p < 0.05)
        print(f"   {lab:<24} {len(res):>5} tested   {passed.sum():>3} PASS the test "
              f"({passed.mean():.1%})")
        best = np.argmax(np.where(a > 0, -np.log(p), -np.inf))
        print(f"   {'':<24} best one: alpha {a[best]*100:+.2f}%/yr, p = {p[best]:.4f}")
    a = np.array([x[0] for x in singles])
    p = np.array([x[1] for x in singles])
    print("\n   the best result found, as a function of how many noise rules were tried:")
    for n in (1, 10, 100, 1000):
        sub_p = np.where(a[:n] > 0, p[:n], 1.0)
        k = np.argmin(sub_p)
        print(f"      tried {n:>5}:  best alpha {a[k]*100:+.2f}%/yr at p = {sub_p[k]:.4f}")


def main():
    print("Feeding the gate strategies that know nothing.\n")
    experiment_a()
    experiment_b()


if __name__ == "__main__":
    main()
