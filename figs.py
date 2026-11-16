import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
O=json.load(open('results.json')); T=O['tok']; S=json.load(open('stats.json'))
LBL={'YTU COSMOS BERT (raw)':'YTU COSMOS (raw input)','YTU COSMOS BERT (lowercased)':'YTU COSMOS (lowercased)','Gemma 2':'Gemma 2 / Turkish-Gemma','Qwen3':'Qwen3 / Trendyol-LLM-8B','Llama 3':'Llama 3 / Turkish-Llama','Toraman morph-BPE':'Toraman morph-BPE*','Trendyol-LLM 7B v1.0':'Trendyol-LLM 7B v1.0'}
L=lambda n:LBL.get(n,n)
GC={'Turkish PLM':'#2b5c8a','Multilingual PLM':'#d08a2e','Turkish LLM':'#3d7f5c','Multilingual LLM':'#9c4a4a'}
U=lambda n:T[n]['UD (news/mixed)']
names=sorted(T,key=lambda n:U(n)['fertility'])
# Fig 1 fertility
fig,ax=plt.subplots(figsize=(6.6,4.6))
y=np.arange(len(names))
for i,n in enumerate(names):
    D=U(n); lo,hi=D['fert_ci']
    ax.barh(i,D['fertility'],color=GC[T[n]['group']],height=0.7)
    ax.errorbar(D['fertility'],i,xerr=[[D['fertility']-lo],[hi-D['fertility']]],color='k',lw=0.8,capsize=2)
    ax.text(D['fertility']+0.06,i,f"{D['fertility']:.3f}",va='center',fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([L(n) for n in names],fontsize=7.5)
ax.axvline(1,ls=':',color='grey',lw=0.8); ax.text(1.02,len(names)-0.4,'theoretical minimum (1.0)',fontsize=6.5,color='grey')
ax.set_xlabel('Fertility (subword tokens per word), UD test sets, 95% bootstrap CI')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=g) for g,c in GC.items()],loc='lower right',fontsize=7,frameon=False)
ax.set_xlim(0,6.8); plt.tight_layout(); plt.savefig('figs/fig1_fertility.pdf'); plt.savefig('figs/fig1_fertility.png',dpi=300)
# Fig 2 PR plane
fig,ax=plt.subplots(figsize=(5.6,4.6))
r=np.linspace(0.01,1,300)
for f in [0.2,0.3,0.4,0.5,0.6]:
    p=f*r/(2*r-f); m=(p>0)&(p<=1); ax.plot(r[m],p[m],color='lightgrey',lw=0.8,zorder=0)
    k=np.argmin(abs(r-0.68))
    if m[k] and p[k]<0.69: ax.text(0.68,p[k]+0.006,f'F1={f}',fontsize=6.5,color='grey',ha='right')
off={'YTU COSMOS BERT (raw)':(-60,6),'YTU COSMOS BERT (lowercased)':(-78,-10),'TURNA':(6,6),'BERTurk':(6,-2),'Kumru':(6,2),'XLM-R':(6,2),'mT5':(6,-6),'TurkishBERTweet':(6,-4),'Gemma 2':(6,4),'Gemma 3':(-46,6),'GPT-4o (o200k)':(6,-4),'Llama 3':(-112,3),'Qwen3':(6,-1),'Trendyol-LLM 7B v1.0':(-95,-12),'GPT-4 (cl100k)':(5,-9),'Mistral v0.3':(5,5),'Commencis-LLM':(6,-8),'Toraman morph-BPE':(-80,6)}
for n in names:
    D=U(n); mk='o' if n!='Toraman morph-BPE' else 'x'
    ax.scatter(D['R'],D['P'],color=GC[T[n]['group']],s=28,marker=mk,zorder=3)
    if n=='Commencis-LLM': continue
    lab='Mistral v0.3 / Commencis' if n=='Mistral v0.3' else L(n)
    ax.annotate(lab,(D['R'],D['P']),xytext=off.get(n,(5,3)),textcoords='offset points',fontsize=6.3,color=GC[T[n]['group']])
ax.set_xlabel('Recall (morpheme boundaries)'); ax.set_ylabel('Precision (morpheme boundaries)')
ax.set_xlim(0.1,0.7); ax.set_ylim(0.1,0.7); plt.tight_layout(); plt.savefig('figs/fig2_pr.pdf'); plt.savefig('figs/fig2_pr.png',dpi=300)
# Fig 3 morph bins
sel=['YTU COSMOS BERT (lowercased)','BERTurk','TURNA','Kumru','XLM-R','mT5','Gemma 3','GPT-4o (o200k)','Llama 3','Qwen3']
fig,ax=plt.subplots(figsize=(6.4,3.8))
for n in sel:
    b=U(n)['bins']; ls='-' if 'Turkish' in T[n]['group'] else '--'
    ax.plot([1,2,3,4],[b[k] for k in '1234'],ls,marker='o',ms=3.5,color=GC[T[n]['group']],lw=1.2,label=L(n))
ax.set_xticks([1,2,3,4]); ax.set_xticklabels(['1','2','3','4+']); ax.set_xlim(0.8,4.2); ax.legend(fontsize=6.5,frameon=False,loc='upper left',bbox_to_anchor=(1.0,1.0))
ax.set_xlabel('Number of surface morphemes in the word'); ax.set_ylabel('Fertility (tokens per word)')
plt.tight_layout(); plt.savefig('figs/fig3_morph_bins.pdf'); plt.savefig('figs/fig3_morph_bins.png',dpi=300)
# Fig 4 distribution
fig,ax=plt.subplots(figsize=(6.6,3.6))
order=sorted(T,key=lambda n:-U(n)['single'])
for i,n in enumerate(order):
    D=U(n); s=D['single']*100; u=D['unk']*D['fertility']*100 if n=='YTU COSMOS BERT (raw)' else 0
    ax.bar(i,s,color='#3d7f5c'); ax.bar(i,100-s-u,bottom=s,color='#b8c4cc'); 
    if u: ax.bar(i,u,bottom=100-u,color='#a33')
ax.set_xticks(range(len(order))); ax.set_xticklabels([L(n) for n in order],rotation=55,ha='right',fontsize=6.5)
ax.set_ylabel('Share of word tokens (%)')
ax.legend(handles=[Patch(color='#3d7f5c',label='single token'),Patch(color='#b8c4cc',label='multiple tokens'),Patch(color='#a33',label='contains UNK')],fontsize=7,frameon=False,loc='upper right',ncol=3,bbox_to_anchor=(1,1.08))
plt.tight_layout(); plt.savefig('figs/fig4_distribution.pdf'); plt.savefig('figs/fig4_distribution.png',dpi=300)
# Fig 5 premium & cost (LLM tokenizers)
llm=[n for n in T if T[n]['group'] in ('Turkish LLM','Multilingual LLM')]
llm=sorted(llm,key=lambda n:T[n]['parallel']['premium'])
fig,ax=plt.subplots(1,2,figsize=(7.0,3.3))
for i,n in enumerate(llm):
    ax[0].barh(i,T[n]['parallel']['premium'],color=GC[T[n]['group']])
    ax[0].text(T[n]['parallel']['premium']+0.03,i,f"{T[n]['parallel']['premium']:.2f}",va='center',fontsize=6.5)
    c=U(n)['fertility']*1e6/1e6
    ax[1].barh(i,U(n)['fertility']*1000,color=GC[T[n]['group']])
    ax[1].text(U(n)['fertility']*1000+40,i,f"{U(n)['fertility']*1000:.0f}",va='center',fontsize=6.5)
ax[0].axvline(1,color='grey',ls=':',lw=0.8)
ax[0].set_yticks(range(len(llm))); ax[0].set_yticklabels([L(n) for n in llm],fontsize=7); ax[1].set_yticks([])
ax[0].set_xlabel('Turkish/English token ratio (WMT16 newstest)'); ax[1].set_xlabel('Tokens per 1,000 Turkish words (UD)')
ax[0].set_xlim(0,2.4); ax[1].set_xlim(0,4300)
plt.tight_layout(); plt.savefig('figs/fig5_cost.pdf'); plt.savefig('figs/fig5_cost.png',dpi=300)
# Fig 6 cross-domain
dn=['UD (news/mixed)','Tweets','Product reviews']; dl=['UD (news/mixed)','Tweets','Product reviews']
sel2=[n for n in names if n!='Toraman morph-BPE']
fig,ax=plt.subplots(figsize=(7.0,3.4)); w=0.27
for j,(d,c) in enumerate(zip(dn,['#4b6a88','#d0a040','#7aa874'])):
    ax.bar(np.arange(len(sel2))+(j-1)*w,[T[n][d]['fertility'] for n in sel2],w,color=c,label=dl[j])
ax.set_xticks(range(len(sel2))); ax.set_xticklabels([L(n) for n in sel2],rotation=55,ha='right',fontsize=6.5)
ax.set_ylabel('Fertility (tokens per word)'); ax.legend(frameon=False,fontsize=7)
plt.tight_layout(); plt.savefig('figs/fig6_domains.pdf'); plt.savefig('figs/fig6_domains.png',dpi=300)
print('ok')
