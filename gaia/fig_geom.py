import json,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
C=json.load(open('cone_diag.json'));Y=json.load(open('cyl_diag.json'))
names=[tuple(n) for n in C['names']];base=C['base'];exp=[(b+d)/b for b,d in base]
col={'baseline':'#444','lohi <=1.3':'#d55e00','lohi <=1.6':'#cc79a7','lower w=0.3':'#0072b2'}
fig,ax=plt.subplots(1,2,figsize=(17,5),gridspec_kw=dict(width_ratios=[2.3,1]))
w=.2;x=np.arange(len(names))
for j,(k,c) in enumerate(col.items()):
    v=np.array(C[k]);ax[0].bar(x+(j-1.5)*w,v.mean(0),w,yerr=v.std(0),color=c,label=k,capsize=2,error_kw=dict(lw=.8))
ax[0].scatter(x,exp,marker='_',s=900,c='k',lw=2,zorder=5,label='expected (M_bar + ρ_DM·V)/M_bar')
ax[0].axhline(1,c='gray',ls=':',lw=.8);ax[0].set_ylim(0,5);ax[0].set_xticks(x);ax[0].set_xticklabels([f'{a}\nr₂={r:g}' for a,r in names],fontsize=7,rotation=0)
ax[0].set_ylabel('M_cone / M_bar,cone');ax[0].set_title('30° cones from the Sun (clipped at 5; `lower` up/down reach 5–8)');ax[0].legend(fontsize=8,loc='upper right')
cyl=[f'Rc={r:g}\nH={h:g}' for r,h in Y['cyl']];xe=np.arange(4);Mb=np.array(Y['M_bar'])
expc=[(Mb[i]+0.01e9*np.pi*r*r*2*h)/Mb[i] for i,(r,h) in enumerate(Y['cyl'])]
for j,(k,c) in enumerate({'baseline':'#444','lohi ≤1.3':'#d55e00','lohi ≤1.6':'#cc79a7','lower w=0.3':'#0072b2'}.items()):
    v=np.array(Y['ratios'][k]);ax[1].bar(xe+(j-1.5)*w,v.mean(0),w,yerr=v.std(0),color=c,capsize=2,error_kw=dict(lw=.8))
ax[1].scatter(xe,expc,marker='_',s=900,c='k',lw=2,zorder=5);ax[1].axhline(1,c='gray',ls=':',lw=.8)
ax[1].set_xticks(xe);ax[1].set_xticklabels(cyl,fontsize=8);ax[1].set_title('cylinders on the Sun\'s vertical axis');ax[1].set_ylabel('M_cyl / M_bar,cyl')
plt.tight_layout();plt.savefig('../gauss_flux/figures/gaia_cone_cylinder.png',dpi=130)
