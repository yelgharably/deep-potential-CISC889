import json,glob,numpy as np,pandas as pd
R=[]
for f in glob.glob('*.json'):
    d=json.load(open(f));c=d['config']
    R.append(dict(group=c['group'],arm=c['arm'],lr=float(c['lr']),clip=float(c['clip']),steps=c['steps'],seed=c['seed'],status=d['status'],
      Me=d['M_err'],phi=d['phi_rms'],bMe=d['best_M_err'],bphi=d['best_phi_rms'],bstep=d['best_step'],wall=d['wall_s']))
df=pd.DataFrame(R);print(len(df),df.status.value_counts().to_dict(),'wall max/med',df.wall.max(),df.wall.median())
pd.set_option('display.width',200)
df['col']=df.Me>0.9
g=df.groupby(['group','arm','lr','clip','steps'])
t=g.agg(n=('seed','size'),collapse=('col','mean'),phi_med=('phi','median'),bphi_med=('bphi','median'),Me_med=('Me','median'),bMe_med=('bMe','median')).round(3)
print(t.to_string())
# paired diffs vs baseline, best-ckpt & final, per setting
rng=np.random.default_rng(0)
def ci(x):
    x=np.array(x);b=[rng.choice(x,len(x)).mean() for _ in range(2000)];return x.mean(),np.percentile(b,2.5),np.percentile(b,97.5)
print('\nPaired (arm - baseline) phi_rms, mean [95%CI]; negative = flux better')
for (grp,lr,cl,st),s in df.groupby(['group','lr','clip','steps']):
    base=s[s.arm=='baseline'].set_index('seed')
    for arm in sorted(set(s.arm)-{'baseline'}):
        a=s[s.arm==arm].set_index('seed');k=a.index.intersection(base.index)
        if len(k)<3:continue
        f=ci(a.phi[k]-base.phi[k]);b=ci(a.bphi[k]-base.bphi[k])
        print(f'{grp} lr={lr:g} cl={cl:g} st={st} {arm:10s} final {f[0]:+.4f} [{f[1]:+.4f},{f[2]:+.4f}]  best {b[0]:+.4f} [{b[1]:+.4f},{b[2]:+.4f}]')
