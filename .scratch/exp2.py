import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
exec(open('.scratch/exp.py').read().split('def rf()')[0])
# drift-adjusted anchor: OD recent level + network-wide recent anomaly
d['anchor']=d.roll6.fillna(d.lag1)
d['anchor_adj']=d.anchor+(d.net_lag1-d.net_roll12).fillna(0)
def rf(): return RandomForestRegressor(300,max_depth=14,min_samples_leaf=4,random_state=0,n_jobs=-1)
def hgb(): return HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0)
rows=[]
for s in ['2023-01','2023-07','2024-01','2024-07','2025-01','2025-07']:
    sp=pd.Period(s,'M'); tr=d[d.period<sp]; te=d[(d.period>=sp)&(d.period<sp+6)]
    if len(te)<200: continue
    y=te[T]
    res={'base_roll6':te.anchor,'base_roll12':te.roll12.fillna(te.lag1),'base_adj':te.anchor_adj}
    Xa,Xb=tr[RICH].fillna(-1),te[RICH].fillna(-1)
    for an in ('anchor','anchor_adj'):
        a,b=tr[an],te[an]
        pr=b+rf().fit(Xa,tr[T]-a).predict(Xb); res['RF_'+an]=pr
        ph=b+hgb().fit(Xa,tr[T]-a).predict(Xb); res['HGB_'+an]=ph
        res['BLEND_'+an]=(pr+ph)/2
    for k,v in res.items():
        v=np.clip(v,0,None)
        rows.append({'fold':s,'model':k,'MAE':mae(y,v),'R2':r2_score(y,v),'bias':(v-y).mean()})
    print('done',s,flush=True)
r=pd.DataFrame(rows)
print(r.pivot(index='model',columns='fold',values='MAE').round(3))
print(r.groupby('model')[['MAE','R2','bias']].mean().round(3).sort_values('MAE'))
