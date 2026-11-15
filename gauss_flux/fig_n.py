import json,glob,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
R=[]
for f in glob.glob('../expanse_pull/results_n/*.json')+glob.glob('../expanse_pull/results_s/*.json'):
    d=json.load(open(f));c=d['config']
    if c['group']=='S1_lr' and float(c['lr']) in (3e-3,1e-2) or c['group'].startswith('N'):
        R.append(dict(arm=c['arm'],lr=float(c['lr']),noise=float(c['noise']),nd=str(c['n_data']),seed=int(c['seed']),bphi=d['best_phi_rms'],bMe=d['best_M_err']))
df=pd.DataFrame(R)
col=dict(baseline='#444',geom='#e69f00',lower='#d55e00',lohi='#0072b2')
rng=np.random.default_rng(0)
def band(x):
    b=[rng.choice(x,len(x)).mean() for _ in range(1000)];return np.percentile(b,[2.5,97.5])
def curve(ax,sub,xs,key,filt,xl):
    for a in col:
        m,lo,hi=[],[],[]
        for x in xs:
            v=sub[(sub.arm==a)&filt(sub,x)][key].values;m.append(v.mean());l,h=band(v);lo.append(l);hi.append(h)
        ax.plot(range(len(xs)),m,'o-',c=col[a],label=a);ax.fill_between(range(len(xs)),lo,hi,color=col[a],alpha=.15)
    ax.set_xticks(range(len(xs)));ax.set_xticklabels(xl);
fig,ax=plt.subplots(2,3,figsize=(15,8))
for i,lr in enumerate([3e-3,1e-2]):
    s=df[df.lr==lr];nz=[0,.004,.016,.05];nd=['8192','4096','2048','1024']
    sn=s[s.nd=='all'];curve(ax[i,0],sn,nz,'bphi',lambda d,x:d.noise==x,[str(x) for x in nz]);ax[i,0].set_yscale('log')
    curve(ax[i,1],sn,nz,'bMe',lambda d,x:d.noise==x,[str(x) for x in nz])
    sd=s[s.noise==0];xs=['all']+nd;curve(ax[i,2],sd,xs,'bphi',lambda d,x:d.nd==x,xs)
    ax[i,0].set_title(f'lr={lr:g}: Φ RMS vs noise σ');ax[i,1].set_title('mass error vs noise σ');ax[i,2].set_title('Φ RMS vs training points (clean)')
    ax[i,0].set_xlabel('σ');ax[i,1].set_xlabel('σ');ax[i,2].set_xlabel('N points')
ax[0,0].legend();plt.tight_layout();plt.savefig('figures/noise_ndata.png',dpi=130)
# paired
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for k,key in enumerate(['bphi','bMe']):
    for j,a in enumerate(['geom','lower','lohi']):
        m,lo,hi=[],[],[]
        for nz in [0,.004,.016,.05]:
            s=df[(df.lr==3e-3)&(df.noise==nz)&(df.nd=='all')]
            x=s[s.arm==a].set_index('seed')[key]-s[s.arm=='baseline'].set_index('seed')[key]
            l,h=band(x.values);m.append(x.mean());lo.append(x.mean()-l);hi.append(h-x.mean())
        ax[k].errorbar(np.arange(4)+(j-1)*.2,m,yerr=[lo,hi],fmt='o',c=col[a],label=a,capsize=3)
    ax[k].axhline(0,c='k',lw=.8);ax[k].set_xticks(range(4));ax[k].set_xticklabels(['0','0.004','0.016','0.05']);ax[k].set_xlabel('noise σ')
    ax[k].set_ylabel(['Δ Φ RMS vs baseline','Δ mass error vs baseline'][k]+'\n<0 = flux better (lr=3e-3, best ckpt)')
ax[0].legend();plt.tight_layout();plt.savefig('figures/noise_paired.png',dpi=130)
