# Tokenizer-level Spearman correlations and exploratory Cetvel link -> stats.json
import json, itertools, time
from scipy.stats import spearmanr
O=json.load(open('results.json')); T=O['tok']
excl={'Toraman morph-BPE','YTU COSMOS BERT (raw)'}
names=[n for n in T if n not in excl]; U=lambda n:T[n]['UD (news/mixed)']
sp=lambda x,y:[float(spearmanr(x,y).statistic),float(spearmanr(x,y).pvalue)]
f=[U(n)['fertility'] for n in names]; S={}
for k in ['P','R','F1','renyi','single']: S['fert_vs_'+k]=sp(f,[U(n)[k] for n in names])
S['fert_vs_premium']=sp(f,[T[n]['parallel']['premium'] for n in names])
S['fert_vs_vocab']=sp(f,[T[n]['vocab'] for n in names])
for d in ['Tweets','Product reviews']: S['ud_vs_'+d]=sp(f,[T[n][d]['fertility'] for n in names])
S['n_models']=len(names)
# Cetvel (Er et al., 2025, Table 2): Avg, MT, GEC of 7-8B base models
cet={'Llama 3':('Llama-3.1-8B',29.5,15.6,35.3),'Qwen3':('Qwen2.5-7B',28.2,5.5,22.3),'Mistral v0.3':('Mistral-v0.3-7B',23.2,3.8,23.5),
     'Trendyol-LLM 7B v1.0':('Trendyol-v1.0-7B-Base',17.3,1.0,0.1),'Commencis-LLM':('Commencis-7B',21.9,4.7,23.7)}
def exact(x,y):
    r0=spearmanr(x,y).statistic; perms=list(itertools.permutations(y))
    return [float(r0),sum(abs(spearmanr(x,p).statistic)>=abs(r0)-1e-12 for p in perms)/len(perms)]
C={'models':cet}; cn=list(cet)
for k,i in [('avg',1),('mt',2),('gec',3)]:
    y=[cet[n][i] for n in cn]
    C['fertility_vs_'+k]=exact([U(n)['fertility'] for n in cn],y); C['F1_vs_'+k]=exact([U(n)['F1'] for n in cn],y)
    C['premium_vs_'+k]=exact([T[n]['parallel']['premium'] for n in cn],y); C['renyi_vs_'+k]=exact([U(n)['renyi'] for n in cn],y)
S['cetvel']=C
from zemberek import TurkishMorphology
t0=time.time(); M=TurkishMorphology.create_with_defaults(); S['zemberek_load']=time.time()-t0
w=[l.split('\t')[1] for l in open('tr_imst-ud-test.conllu',encoding='utf8') if l[0].isdigit()][:4000]
t0=time.time(); [M.analyze(x) for x in w]; S['zemberek_speed_fresh']=len(w)/(time.time()-t0)
json.dump(S,open('stats.json','w'),indent=1)
