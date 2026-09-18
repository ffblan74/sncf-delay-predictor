import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
T='retard_moyen_tous_trains_arrivee'
df=pd.read_csv('data/regularite-mensuelle-tgv-aqst.csv',sep=';')
df['period']=pd.PeriodIndex(df.date,freq='M'); df=df[df.nb_train_prevu>0]
df=df[(df[T]>-1)&(df[T]<120)]; df[T]=df[T].clip(lower=0)
df['od']=df.gare_depart+' > '+df.gare_arrivee; df=df.sort_values(['od','period'])
df['cr']=df.nb_annulation/df.nb_train_prevu
df['p15']=df.nb_train_retard_sup_15/df.nb_train_prevu
df['p30']=df.nb_train_retard_sup_30/df.nb_train_prevu
df['plate']=df.nb_train_retard_arrivee/df.nb_train_prevu
g=df.groupby('od')
for l in (1,2,3,12): df[f'lag{l}']=g[T].shift(l)
for w in (3,6,12): df[f'roll{w}']=g[T].shift(1).rolling(w,min_periods=max(2,w//2)).mean().values
LAG=['cr','p15','p30','plate','retard_moyen_arrivee','retard_moyen_tous_trains_depart',
     'prct_cause_infra','prct_cause_externe','prct_cause_gestion_trafic','prct_cause_materiel_roulant']
for c in LAG:
    df[c+'_l1']=g[c].shift(1); df[c+'_r6']=g[c].shift(1).rolling(6,min_periods=3).mean().values
net=df.groupby('period')[T].mean()
df=df.join(net.shift(1).rename('net_lag1'),on='period')
df=df.join(net.shift(1).rolling(3).mean().rename('net_roll3'),on='period')
df=df.join(net.shift(1).rolling(12).mean().rename('net_roll12'),on='period')
df['net_anom']=df.net_roll3-df.net_roll12
dep=df.groupby(['gare_depart','period'])[T].mean()
df=df.join(dep.groupby('gare_depart').shift(1).rename('dep_lag1'),on=['gare_depart','period'])
df['month']=df.period.dt.month
df['trains_lag1']=g['nb_train_prevu'].shift(1); df['traffic_ratio']=df.nb_train_prevu/df.trains_lag1
BASE=['lag1','lag2','lag3','lag12','roll3','roll6','roll12','net_lag1','net_roll3','net_anom',
      'dep_lag1','duree_moyenne','nb_train_prevu','traffic_ratio','month']
RICH=BASE+[c+s for c in LAG for s in ('_l1','_r6')]
d=df.dropna(subset=['lag1','roll3','net_lag1']).copy()
def rf(): return RandomForestRegressor(300,max_depth=14,min_samples_leaf=4,random_state=0,n_jobs=-1)
def hgb(): return HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=0)
for s in ['2024-01','2025-01']:
    sp=pd.Period(s,'M'); tr=d[d.period<sp]; te=d[(d.period>=sp)&(d.period<sp+12)]
    bb=te.roll6.fillna(te.lag1); ba=tr.roll6.fillna(tr.lag1)
    print('== test',s,'n_tr',len(tr),'n_te',len(te),flush=True)
    print('   base_roll6 MAE %.3f R2 %+.3f'%(mae(te[T],bb),r2_score(te[T],bb)),flush=True)
    print('   base_roll12 MAE %.3f R2 %+.3f'%(mae(te[T],te.roll12.fillna(te.lag1)),r2_score(te[T],te.roll12.fillna(te.lag1))),flush=True)
    for nm,F in [('BASE',BASE),('RICH',RICH)]:
        Xa,Xb=tr[F].fillna(-1),te[F].fillna(-1)
        for mn,mk in [('RF',rf),('HGB',hgb)]:
            p=bb+mk().fit(Xa,tr[T]-ba).predict(Xb)
            print('   %-4s %-3s resid  MAE %.3f R2 %+.3f'%(nm,mn,mae(te[T],p),r2_score(te[T],p)),flush=True)
            p=mk().fit(Xa,tr[T]).predict(Xb)
            print('   %-4s %-3s direct MAE %.3f R2 %+.3f'%(nm,mn,mae(te[T],p),r2_score(te[T],p)),flush=True)
