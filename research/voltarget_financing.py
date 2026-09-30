# =====================================================================
#  What actually limits the diversified book: the cost of leverage.
#
#  CONSTRUCTION on data already seen. Both one-shot samples are spent.
#
#  The book's Sharpe advantage over SPY cannot be converted into return,
#  because borrowing at T-bill + 1.5% costs more than the excess return
#  the extra exposure earns. Measured here: moving to futures-style
#  financing (rf + 0.3%) instead of retail margin buys +1.7%/yr and
#  +0.09 Sharpe at a 20% target. That is the largest single improvement
#  found, and it is an execution choice rather than a research one.
#
#  ALSO TESTED AND REJECTED (2026-09-23): a trend filter, the standard
#  complement to volatility targeting. It made the book WORSE at every
#  window -- Sharpe 0.72 with no filter, 0.57 at 126d, 0.54 at 200d,
#  0.51 at 252d -- and did not fix the weak 2021-2026 era either
#  (+4.4%/yr against +7.3% without it). Trend and volatility targeting
#  both cut exposure, and stacking them compounds the drag.
#
#    .venv\Scripts\python.exe research\voltarget_financing.py
# =====================================================================
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"research"))
from voltarget_multi import BUCKETS, LOOK, COST, load, stats, month_ends

px,tb=load(); idx=px["SPY"].index
rets=pd.DataFrame({s:v.pct_change() for s,v in px.items()}).reindex(idx)
B=pd.DataFrame({n:rets[s].mean(axis=1,skipna=True) for n,s in BUCKETS.items()})
B=B[B.notna().any(axis=1)]; i=B.index; me=month_ends(i); rf=tb.reindex(i).ffill().fillna(0.0)
bv=B.rolling(LOOK).std()*np.sqrt(252)
inv=(1.0/bv).where(B.notna()); w=inv.div(inv.sum(axis=1),axis=0)
keep=pd.DataFrame(np.repeat(me[:,None],w.shape[1],axis=1),index=w.index,columns=w.columns)
w=w.where(keep).shift(1).ffill(); rp=(w*B).sum(axis=1,min_count=1)
pv=rp.rolling(LOOK).std()*np.sqrt(252)

def book(target,cap,spread):
    lev=(target/pv).clip(lower=0,upper=cap).where(me).shift(1).ffill()
    rate=np.where(lev.to_numpy()<=1.0, rf.to_numpy()/252,(rf.to_numpy()+spread)/252)
    tn=w.diff().abs().sum(axis=1).fillna(0.0)+lev.diff().abs().fillna(0.0)
    return (lev*rp+(1-lev)*rate-tn*COST).dropna().loc['2005':]

spy=rets.SPY.fillna(0.0).loc['2005':]
a,v,s,d,g=stats(spy)
print("2005+. How much does the cost of leverage matter?\n")
print(f"   {'':<40}{'return':>9}{'vol':>8}{'Sharpe':>9}{'maxDD':>8}")
print(f"   {'SPY buy & hold':<40}{a*100:>8.2f}%{v*100:>8.1f}%{s:>9.2f}{d*100:>7.0f}%")
for label,spread in [("retail margin, rf+1.5%",0.015),
                     ("broker margin, rf+0.75%",0.0075),
                     ("futures financing, rf+0.3%",0.003),
                     ("free money, rf+0.0% (bound)",0.0)]:
    print(f"\n   {label}")
    for tgt,cp in [(0.10,1.5),(0.15,2.0),(0.20,2.5)]:
        a,v,s,d,g=stats(book(tgt,cp,spread))
        print(f"      {'target %d%%'%(tgt*100):<37}{a*100:>8.2f}%{v*100:>8.1f}%"
              f"{s:>9.2f}{d*100:>7.0f}%")
