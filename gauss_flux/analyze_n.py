import json,glob,numpy as np,pandas as pd
R=[]
for f in glob.glob('../expanse_pull/results_n/*.json'):
    d=json.load(open(f));c=d['config']
    R.append(dict(g=c['group'],arm=c['arm'],lr=float(c['lr']),noise=float(c['noise']),nd=str(c['n_data']),seed=int(c['seed']),st=d['status'],Me=d['M_err'],bMe=d['best_M_err'],phi=d['phi_rms'],bphi=d['best_phi_rms']))
df=pd.DataFrame(R);print(len(df),df.st.value_counts().to_dict());pd.set_option('display.width',200)
rng=np.random.default_rng(0)
def ci(x):
    b=[rng.choice(x,len(x)).mean() for _ in range(2000)];return x.mean(),*np.percentile(b,[2.5,97.5])
for key in ['bphi','bMe']:
    print('\npaired (arm-baseline)',key,'mean [95%CI]; <0 flux better')
    for (lr,nz,nd),s in df.groupby(['lr','noise','nd']):
        b=s[s.arm=='baseline'].set_index('seed')[key];out=f'lr={lr:g} noise={nz:g} N={nd:5s} base={b.median():.3f} |'
        for a in ['geom','lower','lohi']:
            x=s[s.arm==a].set_index('seed')[key]-b;m,l,h=ci(x.dropna().values);out+=f' {a} {m:+.3f}[{l:+.3f},{h:+.3f}]'
        print(out)
