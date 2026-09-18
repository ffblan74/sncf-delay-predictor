import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error as mae, r2_score
exec(open('.scratch/exp.py').read().split('def rf()')[0])
print('dup keys:', df.duplicated(['date','gare_depart','gare_arrivee']).sum(), flush=True)
d['anchor']=d.roll6.fillna(d.lag1)
enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
cats=['service','gare_depart','gare_arrivee']
C=pd.DataFrame(enc.fit_transform(d[cats]),columns=enc.get_feature_names_out(cats),index=d.index)
dd=pd.concat([d,C],axis=1)
SRV=[c for c in C.columns if c.startswith('service')]
STA=[c for c in C.columns if not c.startswith('service')]
def hgb(): return HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0)
rows=[]
for s in ['2023-01','2023-07','2024-01','2024-07','2025-01','2025-07']:
    sp=pd.Period(s,'M'); tr=dd[dd.period<sp]; te=dd[(dd.period>=sp)&(dd.period<sp+6)]
    y=te[T]; a,b=tr.anchor,te.anchor
    for nm,F in [('rich',RICH),('rich+srv',RICH+SRV),('rich+srv+sta',RICH+SRV+STA)]:
        p=np.clip(b+hgb().fit(tr[F].fillna(-1),tr[T]-a).predict(te[F].fillna(-1)),0,None)
        rows.append({'fold':s,'m':nm,'MAE':mae(y,p),'R2':r2_score(y,p)})
    print('done',s,flush=True)
r=pd.DataFrame(rows)
print(r.pivot(index='m',columns='fold',values='MAE').round(3))
print(r.groupby('m')[['MAE','R2']].mean().round(4))
