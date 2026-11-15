import json,glob,numpy as np,collections,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
G=collections.defaultdict(list)
for f in glob.glob('phi_results/*.json'):
    d=json.load(open(f));c=d['cfg']
    key=c['arm']+(f" w={c['w']:g}" if c['arm']=='lower' else (f" ≤{c['hi_scale']:g}M_bar" if c['arm']=='lohi' else ''));G[key].append(d['best'])
order=[k for k in ['baseline','monopos','lower w=0.1','lower w=0.3','lower w=1','lohi ≤1.6M_bar','lohi ≤1.3M_bar'] if k in G]
col=dict(zip(order,['#444','#e69f00','#56b4e9','#0072b2','#003f7f','#cc79a7','#d55e00']))
fig,ax=plt.subplots(1,3,figsize=(16,4.5)); r=G[order[0]][0]['r']
for k in order:
    v=np.array([d['M_ratio'] for d in G[k]]);m,s=v.mean(0),v.std(0);ax[0].errorbar(r,m,yerr=s,fmt='o-',c=col[k],label=f'{k} (n={len(v)})',capsize=3)
ax[0].axhline(1,c='k',ls='--',lw=.8);ax[0].set_xlabel('sphere radius around Sun [kpc]');ax[0].set_ylabel('M_enc / M_bar  (physical ≥ 1)');ax[0].legend(fontsize=7)
for a_,key,lab,ref in ((ax[1],'rho_sun','ρ at Sun [M☉/pc³]',0.10),(ax[2],'Sigma_z1p0','Σ(<1 kpc) at R0 [M☉/pc²]',70)):
    for i,k in enumerate(order):
        x=np.array([d[key] for d in G[k]]);a_.bar(i,x.mean(),yerr=x.std(),color=col[k],capsize=3)
    a_.axhline(ref,c='k',ls='--',lw=1,label=f'literature ≈ {ref:g}');a_.set_xticks(range(len(order)));a_.set_xticklabels(order,rotation=40,ha='right',fontsize=8);a_.set_ylabel(lab);a_.legend()
plt.tight_layout();plt.savefig('../gauss_flux/figures/gaia_physical.png',dpi=130)
