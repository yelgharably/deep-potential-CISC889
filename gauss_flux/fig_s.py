import json,glob,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt,os,sys
src=sys.argv[1] if len(sys.argv)>1 else '../expanse_pull/results_s'
R=[]
for f in glob.glob(src+'/*.json'):
    d=json.load(open(f));c=d['config']
    R.append(dict(group=c['group'],arm=c['arm'],lr=float(c['lr']),steps=int(c['steps']),seed=int(c['seed']),Me=d['M_err'],phi=d['phi_rms'],bphi=d['best_phi_rms']))
df=pd.DataFrame(R);os.makedirs('figures',exist_ok=True)
col=dict(baseline='#444',geom='#e69f00',lower='#d55e00',lohi='#0072b2')
s1=df[df.group=='S1_lr'];fig,ax=plt.subplots(1,3,figsize=(15,4))
for a,g in s1.groupby('arm'):
    lrs=sorted(g.lr.unique())
    ax[0].plot(lrs,[(g[g.lr==l].Me>0.9).mean() for l in lrs],'o-',c=col[a],label=a)
    ax[1].plot(lrs,[g[g.lr==l].phi.median() for l in lrs],'o-',c=col[a],label=a)
b=df[df.group=='S4_orig_lr_bestckpt']
for a in col:
    x=b[b.arm==a]
    if len(x): ax[2].bar([list(col).index(a)-.2,list(col).index(a)+.2],[x.phi.median(),x.bphi.median()],.4,color=[col[a],col[a]],alpha=1)
ax[2].set_xticks(range(4));ax[2].set_xticklabels(list(col));ax[2].set_title('lr=0.05: final (left) vs best ckpt (right), median')
for i,t in enumerate(['collapse rate (M_err>0.9) vs lr','median Φ RMS vs lr (final)']):ax[i].set_title(t);ax[i].set_xscale('log')
ax[1].set_yscale('log');ax[0].legend();ax[0].set_xlabel('lr');ax[1].set_xlabel('lr');plt.tight_layout();plt.savefig('figures/stable_overview.png',dpi=130)
# paired diff
rng=np.random.default_rng(0);fig,ax=plt.subplots(figsize=(8,4.5))
lrs=[1e-3,3e-3,1e-2,2e-2];w=.25
for j,a in enumerate(['geom','lower','lohi']):
    m,lo,hi=[],[],[]
    for l in lrs:
        s=s1[s1.lr==l];x=s[s.arm==a].set_index('seed').bphi-s[s.arm=='baseline'].set_index('seed').bphi
        bs=[rng.choice(x,len(x)).mean() for _ in range(2000)];m.append(x.mean());lo.append(x.mean()-np.percentile(bs,2.5));hi.append(np.percentile(bs,97.5)-x.mean())
    ax.errorbar(np.arange(4)+(j-1)*w,m,yerr=[lo,hi],fmt='o',c=col[a],label=a,capsize=3)
ax.axhline(0,c='k',lw=.8);ax.set_xticks(range(4));ax.set_xticklabels([f'lr={l:g}' for l in lrs]);ax.set_ylabel('Δ Φ RMS vs baseline (best ckpt)\n<0 = flux better');ax.legend();plt.tight_layout();plt.savefig('figures/stable_paired.png',dpi=130)
