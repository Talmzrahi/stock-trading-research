# ═══════════════════════════════════════════════════════════════════════
#  If the project allowed shorting, where would it be best and what would
#  it earn? Grid over signal x cutoff x holding period, dollar-neutral.
#
#  DEVELOPMENT DATA, AND THIS IS A FITTED SEARCH. Picking the best of 27
#  configurations is parameter fitting, not evidence. The winner is
#  reported with its era split and its cost curve precisely so the fitting
#  is visible.
#
#  Nothing sees the future: the reader is refitted each year on prior
#  years only (expanding window), and its percentile rank is trailing.
#  That costs sample -- scores start 2014 -- and is worth it; three
#  results in this project died once look-ahead was removed.
#
#  RESULT (2026-09-22). Best of the grid: PEAD surprise, top/bottom 10%,
#  20-session hold -- +3.57%/yr, vol 10.0%, Sharpe 0.36, p=0.207, alpha
#  +3.04%/yr (p=0.322), beta +0.04. It is -2.60%/yr across 2014-2019 and
#  +8.83%/yr across 2020-2026, and it is a lone peak in the grid, not a
#  plateau. At 20bps a side it earns +0.67%/yr; at 35bps it loses money.
#  Nothing in the grid reaches p<0.05. Every config is beta ~0, which is
#  the one thing dollar-neutrality delivers reliably.
#
#  COSTS: `side` multiplies the RETURN, never applied after the cost
#  subtraction -- that inversion turns short-side costs into gains and it
#  was live in v3_longshort_probe.py until 2026-09-22, worth ~+0.9pp/yr at
#  a 60-session hold and ~+2.5pp/yr at 20 sessions.
#
#  Funding reality unchanged: shorting needs a $2,000 US margin minimum
#  and whole shares, so this is a research question at a ~$100 stake.
#
#    .venv\Scripts\python.exe research\longshort_sim.py
# ═══════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"research"))
from trader.backtest import load_market
from trader.config import load_config
from trader.signals.sue import trailing_percentile
from inference_check import market_alpha
from v3_readers_eval import ridge
from v3_signal_probe import BASE, load
BORROW,MIN_DAYS=0.01,5
cfg=load_config(); d=load(cfg)
tcols=[c for c in d.columns if any(c.startswith(p) for p in
       ("distilroberta_fin","finbert","minilm","spread","lm_tone"))]
feats=BASE+tcols
d=d[d.reaction.notna()].copy(); d["entry_date"]=pd.to_datetime(d.gap_entry_date)
d["year"]=d.entry_date.dt.year; d=d.sort_values("entry_date").reset_index(drop=True)
X,y=d[feats].to_numpy(float),d.reaction.to_numpy(); d["escore"]=np.nan
for Y in sorted(d.year.unique()):
    tr,te=(d.year<Y-1).to_numpy(),(d.year==Y).to_numpy()
    if tr.sum()<2000: continue
    d.loc[te,"escore"]=ridge(X[tr],y[tr],10.0)(X[te])
d=d[d.escore.notna()].copy()
d["tx"]=trailing_percentile(d.entry_date,d.escore)
d=d[d.tx.notna()&d.conviction.notna()].copy()
d["combo"]=(d.tx.rank(pct=True)+d.conviction.rank(pct=True))/2
market=load_market(cfg); closes=market.closes.ffill()
col={s:i for i,s in enumerate(closes.columns)}
rets=closes.pct_change().fillna(0.0).to_numpy(); nd=len(market.cal)
cal=pd.DatetimeIndex(market.cal)

def side(p,sgn,hold,cost):
    """sgn is applied to the RETURN; cost is always a subtraction."""
    tot,cnt=np.zeros(nd),np.zeros(nd)
    for e,sym in zip(p.gap_entry_idx.to_numpy(),p.symbol):
        c=col.get(sym)
        if c is None: continue
        w=slice(e+1,e+hold+1)
        tot[w]+=sgn*rets[w,c]; cnt[w]+=1
        tot[e+1]-=cost; tot[min(e+hold,nd-1)]-=cost
    o=np.zeros(nd); m=cnt>0; o[m]=tot[m]/cnt[m]; return o,cnt

def series(sig,cut,hold,cost):
    L,ln=side(d[d[sig]>=1-cut],+1,hold,cost); S,sn=side(d[d[sig]<=cut],-1,hold,cost)
    live=(ln>0)&(sn>0)
    if live.sum()<250: return None
    idx=cal[live]
    ser=pd.Series((L+S)[live]/2 - np.where(sn[live]>0,BORROW/252,0)/2,index=idx)
    per=idx.to_period("M")
    mo=pd.DataFrame({"port":(1+ser).groupby(per).prod()-1})
    mo["bench"]=((1+pd.Series(rets[live,col[cfg.benchmark]],index=idx)).groupby(per).prod()-1)
    mo=mo[ser.groupby(per).size()>=MIN_DAYS]
    return mo if len(mo)>=36 else None

print("COSTS FIXED: sign applied to returns, cost always subtracted.\n")
print(f"{'signal':<12}{'cut':>5}{'hold':>6}{'ret/yr':>9}{'vol':>7}{'Sharpe':>8}{'p':>7}"
      f"{'alpha':>9}{'p':>7}{'beta':>7}{'14-19':>9}{'20-26':>9}")
rows=[]
for sig,name in [("conviction","PEAD"),("tx","TEXT"),("combo","BOTH")]:
    for cut in [0.10,0.20,0.30]:
        for hold in [20,40,60]:
            mo=series(sig,cut,hold,10/1e4)
            if mo is None: continue
            a,ap,beta,_=market_alpha(mo)
            ann=mo.port.mean()*12; vol=mo.port.std()*np.sqrt(12)
            p=stats.ttest_1samp(mo.port,0).pvalue
            h1=mo[mo.index.year<=2019].port.mean()*12
            h2=mo[mo.index.year>=2020].port.mean()*12
            rows.append((name,cut,hold,ann,vol,ann/vol,p,a,ap,beta,h1,h2))
R=pd.DataFrame(rows,columns=["s","cut","hold","ann","vol","sh","p","a","ap","b","h1","h2"])
for _,r in R.sort_values("sh",ascending=False).head(10).iterrows():
    print(f"{r.s:<12}{r.cut:>5.0%}{int(r.hold):>6}{r.ann*100:>+8.2f}%{r.vol*100:>6.1f}%"
          f"{r.sh:>8.2f}{r.p:>7.3f}{r.a*100:>+8.2f}%{r.ap:>7.3f}{r.b:>+7.2f}"
          f"{r.h1*100:>+8.2f}%{r.h2*100:>+8.2f}%")
print("\n── Cost sensitivity, now that it is wired correctly ──")
print(f"   {'config':<20}{'5bps':>9}{'10bps':>9}{'20bps':>9}{'35bps':>9}")
for sig,cut,hold,lab in [("conviction",0.10,20,"PEAD 10%/20d"),
                         ("tx",0.20,60,"TEXT 20%/60d"),("combo",0.20,60,"BOTH 20%/60d")]:
    line=f"   {lab:<20}"
    for bps in [5,10,20,35]:
        mo=series(sig,cut,hold,bps/1e4)
        line+=f"{mo.port.mean()*12*100:>+8.2f}%"
    print(line)
