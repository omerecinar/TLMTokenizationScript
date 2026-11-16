# Reproducible input-representation evaluation of Turkish tokenizers (CPU only, no model weights)
import json, time, random, math, re, collections, sys
import numpy as np
from transformers import AutoTokenizer
import tiktoken
from zemberek import TurkishMorphology
rng=np.random.default_rng(42); random.seed(42)
OUT={}
def trlower(s): return s.replace('I','ı').replace('İ','i').lower()
LET=re.compile(r'[A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû]')
# ---------- data ----------
def read_conllu(fn):
    sents=[]; cur=[]
    for line in open(fn,encoding='utf8'):
        line=line.rstrip('\n')
        if not line: 
            if cur: sents.append(cur); cur=[]
            continue
        if line.startswith('#'): continue
        c=line.split('\t')
        if '-' in c[0] or '.' in c[0]: continue
        cur.append((c[1],c[2]))
    if cur: sents.append(cur)
    return sents
ud=read_conllu('tr_boun-ud-test.conllu')+read_conllu('tr_imst-ud-test.conllu')
def build(sent_words):
    # returns text, list of (start,end,word,is_letter)
    text='';spans=[]
    for w in sent_words:
        if text: text+=' '
        s=len(text); text+=w; spans.append((s,len(text),w,bool(LET.search(w))))
    return text,spans
UD=[]; lemmas={}
for s in ud:
    words=[w for w,_ in s]
    t,sp=build(words); UD.append((t,sp))
    for (w,l) in s:
        if LET.search(w): lemmas.setdefault(w,[]).append(l)
nw=sum(1 for t,sp in UD for x in sp if x[3]); types=set(x[2] for t,sp in UD for x in sp if x[3])
OUT['ud']=dict(sentences=len(UD),word_tokens=nw,word_types=len(types))
print('UD',OUT['ud'],flush=True)
from datasets import load_dataset
ds=load_dataset('winvoker/turkish-sentiment-analysis-dataset')
allrows=[r for sp in ['train','test'] for r in ds[sp]]
def sample(src,n):
    rows=[r['text'].strip() for r in allrows if r['dataset']==src and 3<=len(r['text'].split())<=60]
    random.Random(7).shuffle(rows); return rows[:n]
TOK=re.compile(r"\w+(?:['’]\w+)*|[^\w\s]",re.U)
def corp(texts):
    C=[]
    for x in texts:
        ws=TOK.findall(x)
        if ws: C.append(build(ws))
    return C
DOM={'UD (news/mixed)':UD,'Tweets':corp(sample('tweet-pn',2000)),'Product reviews':corp(sample('urun_yorumlari',2000))}
for k,v in DOM.items(): print(k,len(v),sum(1 for t,sp in v for x in sp if x[3]),flush=True)
OUT['domains']={k:dict(sentences=len(v),word_tokens=sum(1 for t,sp in v for x in sp if x[3])) for k,v in DOM.items()}
wmt=load_dataset('wmt/wmt16','tr-en',split='test')
PAR=[(r['translation']['tr'],r['translation']['en']) for r in wmt]
# ---------- zemberek ----------
t0=time.time(); M=TurkishMorphology.create_with_defaults(); zload=time.time()-t0
GOLD={}  # word -> list of surface morphemes (first analysis), if concatenation==word
alltypes=set()
for C in DOM.values():
    for t,sp in C:
        for x in sp:
            if x[3]: alltypes.add(x[2])
zstats={}
ana_n=0; amb=[]; lemma_hit=0; t0=time.time(); analysed_tokens=0; tok_total=0
cache={}
def analyze(w):
    if w not in cache:
        try: cache[w]=list(M.analyze(w))
        except Exception: cache[w]=[]
    return cache[w]
for C,name in [(UD,'UD')]:
    for t,sp in C:
        for x in sp:
            if not x[3]: continue
            tok_total+=1; a=analyze(x[2])
            a=[r for r in a if not r.is_unknown()] if a else []
            if a:
                analysed_tokens+=1; amb.append(len(a))
gold_l=0;hit_l=0
for w,ls in lemmas.items():
    a=[r for r in analyze(w) if not r.is_unknown()]
    if not a: continue
    cands=set()
    for r in a:
        try: cands.add(trlower(r.get_stem()))
        except Exception: pass
        try: cands.add(trlower(r.item.lemma))
        except Exception: pass
    for l in ls:
        gold_l+=1; hit_l+= trlower(l) in cands
zt=time.time()-t0
# speed on fresh morphology over a sample of UD tokens
samp=[x[2] for t,sp in UD for x in sp if x[3]][:3000]
M2=M; t0=time.time()
for w in samp: M2.analyze(w)
zspeed=len(samp)/(time.time()-t0)
OUT['zemberek']=dict(analysis_rate=analysed_tokens/tok_total, ambiguity=float(np.mean(amb)), lemma_oracle=hit_l/gold_l, speed_wps=zspeed, load_s=zload)
def gold_seg(w):
    a=[r for r in analyze(w) if not r.is_unknown()]
    if not a: return None
    segs=[md.surface for md in a[0].morpheme_data_list if md.surface]
    if trlower(''.join(segs))!=trlower(w): return 'MISALIGN'
    return segs
for w in alltypes: GOLD[w]=gold_seg(w)
udtypes=[w for w in types]
an=[w for w in udtypes if GOLD[w] is not None]; al=[w for w in an if GOLD[w]!='MISALIGN']
OUT['zemberek'].update(analysed_types=len(an), aligned_types=len(al))
print('zemberek',OUT['zemberek'],flush=True)
# ---------- tokenizers ----------
HF={
 'BERTurk':'dbmdz/bert-base-turkish-cased',
 'YTU COSMOS BERT (raw)':'ytu-ce-cosmos/turkish-base-bert-uncased',
 'YTU COSMOS BERT (lowercased)':'ytu-ce-cosmos/turkish-base-bert-uncased',
 'TURNA':'boun-tabi-LMG/TURNA',
 'TurkishBERTweet':'VRLLab/TurkishBERTweet',
 'Toraman morph-BPE':'ctoraman/RoBERTa-TR-medium-morph-44k',
 'XLM-R':'FacebookAI/xlm-roberta-base',
 'mT5':'google/mt5-base',
 'Kumru':'vngrs/Kumru-2B',
 'Trendyol-LLM 7B v1.0':'Trendyol/Trendyol-LLM-7b-base-v1.0',
 'Commencis-LLM':'Commencis/Commencis-LLM',
 'Llama 3':'NousResearch/Meta-Llama-3.1-8B',
 'Qwen3':'Qwen/Qwen3-8B',
 'Gemma 2':'ytu-ce-cosmos/Turkish-Gemma-9b-T1',
 'Gemma 3':'unsloth/gemma-3-4b-it',
 'Mistral v0.3':'unsloth/mistral-7b-v0.3',
}
TT={'GPT-4o (o200k)':'o200k_base','GPT-4 (cl100k)':'cl100k_base'}
GROUP={}
for k in ['BERTurk','YTU COSMOS BERT (raw)','YTU COSMOS BERT (lowercased)','TURNA','TurkishBERTweet','Toraman morph-BPE']: GROUP[k]='Turkish PLM'
for k in ['XLM-R','mT5']: GROUP[k]='Multilingual PLM'
for k in ['Kumru','Trendyol-LLM 7B v1.0','Commencis-LLM']: GROUP[k]='Turkish LLM'
for k in ['Llama 3','Qwen3','Gemma 2','Gemma 3','Mistral v0.3','GPT-4o (o200k)','GPT-4 (cl100k)']: GROUP[k]='Multilingual LLM'
class Tk:
    def __init__(s,name):
        s.name=name
        if name in TT:
            s.e=tiktoken.get_encoding(TT[name]); s.V=s.e.n_vocab; s.unk=None; s.kind='tt'
        else:
            s.t=AutoTokenizer.from_pretrained(HF[name]); s.V=len(s.t); s.unk=s.t.unk_token_id; s.kind='hf'
        s.pre=trlower if 'lowercased' in name else (lambda x:x)
    def enc(s,text):
        text=s.pre(text)
        if s.kind=='tt':
            ids=s.e.encode(text); _,offs=s.e.decode_with_offsets(ids)
            spans=[(offs[i], offs[i+1] if i+1<len(offs) else len(text)) for i in range(len(ids))]
            return ids,spans
        o=s.t(text,return_offsets_mapping=True,add_special_tokens=False)
        return o['input_ids'],[tuple(x) for x in o['offset_mapping']]
    def count(s,text): return len(s.enc(text)[0])
def per_word(tk,text,spans):
    ids,offs=tk.enc(text)
    starts=[sp[0] for sp in spans]
    cnt=[0]*len(spans); bnd=[set() for _ in spans]; unk=[0]*len(spans)
    import bisect
    for i,(a,b) in enumerate(offs):
        # first non-space char
        seg=text[a:b] if b>a else ''
        k=0
        while k<len(seg) and seg[k].isspace(): k+=1
        if b<=a: pos=a
        elif k==len(seg): pos=b  # whitespace-only token -> next word
        else: pos=a+k
        j=bisect.bisect_right(starts,pos)-1
        if j<0: j=0
        if pos>=spans[j][1] and j+1<len(spans): j=j+1
        cnt[j]+=1
        if tk.unk is not None and ids[i]==tk.unk: unk[j]+=1
        ws,we=spans[j][0],spans[j][1]
        for p in (pos,b):
            if ws<p<we: bnd[j].add(p-ws)
    return ids,cnt,bnd,unk
def renyi_eff(counter,V,alpha=2.5):
    tot=sum(counter.values()); p=np.array(list(counter.values()),dtype=float)/tot
    H=1/(1-alpha)*math.log((p**alpha).sum())
    return H/math.log(V)
RES={}
for name in list(HF)+list(TT):
    t0=time.time()
    try: tk=Tk(name)
    except Exception as e: print('FAIL',name,e,flush=True); continue
    R=dict(group=GROUP[name],vocab=tk.V)
    for dname,C in DOM.items():
        wc=[];wm=[];wtype={};unk_tok=0;tot_tok=0;unigram=collections.Counter();t1=time.time()
        for text,spans in C:
            ids,cnt,bnd,unk=per_word(tk,text,spans)
            unigram.update(ids); tot_tok+=len(ids)
            for j,x in enumerate(spans):
                if not x[3]: continue
                wc.append(cnt[j]); unk_tok+=unk[j]; wm.append(x[2])
                if x[2] not in wtype: wtype[x[2]]=bnd[j]
        dt=time.time()-t1
        wc=np.array(wc)
        D=dict(fertility=float(wc.mean()), single=float((wc==1).mean()), unk=unk_tok/max(1,wc.sum()),
               renyi=renyi_eff(unigram,tk.V), tokens_per_sec=tot_tok/dt)
        # bootstrap fertility
        bs=[wc[rng.integers(0,len(wc),len(wc))].mean() for _ in range(1000)]
        D['fert_ci']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
        # morph boundary metrics over aligned types
        tp=[];fp=[];fn=[];per_type=[]
        for w,b in wtype.items():
            g=GOLD.get(w)
            if not g or g=='MISALIGN': continue
            gb=set(np.cumsum([len(x) for x in g])[:-1].tolist())
            per_type.append((len(b&gb),len(b-gb),len(gb-b)))
        A=np.array(per_type) if per_type else np.zeros((0,3))
        def prf(A):
            T,F,N=A.sum(0); P=T/(T+F) if T+F else 0; Rr=T/(T+N) if T+N else 0
            return P,Rr,(2*P*Rr/(P+Rr) if P+Rr else 0)
        P,Rr,F1=prf(A); D.update(P=P,R=Rr,F1=F1,aligned=len(per_type))
        bb=[prf(A[rng.integers(0,len(A),len(A))]) for _ in range(1000)]
        bb=np.array(bb); D['P_ci']=np.percentile(bb[:,0],[2.5,97.5]).tolist(); D['R_ci']=np.percentile(bb[:,1],[2.5,97.5]).tolist(); D['F1_ci']=np.percentile(bb[:,2],[2.5,97.5]).tolist()
        if dname.startswith('UD'):
            # morph-count bins and spearman (word tokens with aligned gold)
            mc=[];tc=[]
            for c,w in zip(wc,wm):
                g=GOLD.get(w)
                if g and g!='MISALIGN': mc.append(len(g)); tc.append(int(c))
            mc=np.array(mc);tc=np.array(tc)
            from scipy.stats import spearmanr
            rho=spearmanr(mc,tc)
            D['spearman_morph_tokens']=[float(rho.statistic),float(rho.pvalue),len(mc)]
            D['bins']={str(k):float(tc[np.minimum(mc,4)==k].mean()) for k in [1,2,3,4]}
            D['bin_sizes']={str(k):int((np.minimum(mc,4)==k).sum()) for k in [1,2,3,4]}
            D['wc']=wc.tolist()
        R[dname]=D
    # parallel premium
    tr=sum(tk.count(a) for a,b in PAR); en=sum(tk.count(b) for a,b in PAR)
    trw=sum(len(a.split()) for a,b in PAR); enw=sum(len(b.split()) for a,b in PAR)
    R['parallel']=dict(tr_tokens=tr,en_tokens=en,premium=tr/en,tr_tok_per_word=tr/trw,en_tok_per_word=en/enw)
    RES[name]=R
    print(name,'V=',tk.V,'fert=%.3f'%R['UD (news/mixed)']['fertility'],'F1=%.3f'%R['UD (news/mixed)']['F1'],'prem=%.2f'%R['parallel']['premium'],'%.0fs'%(time.time()-t0),flush=True)
OUT['tok']=RES
# Toraman pre-segmented condition
tk=Tk('Toraman morph-BPE'); cnts=[]
for text,spans in UD:
    for x in spans:
        if not x[3]: continue
        g=GOLD.get(x[2]); inp=' '.join(g) if g and g!='MISALIGN' else x[2]
        cnts.append(len(tk.t(inp,add_special_tokens=False)['input_ids']) - 0)
cnts=np.array(cnts); bs=[cnts[rng.integers(0,len(cnts),len(cnts))].mean() for _ in range(1000)]
OUT['toraman_preseg']=dict(fertility=float(cnts.mean()),ci=np.percentile(bs,[2.5,97.5]).tolist())
OUT['toraman_char_types']=None
# paired fertility differences (UD)
names=[n for n in RES]; W={n:np.array(RES[n]['UD (news/mixed)']['wc']) for n in names}
N=len(W[names[0]]); idx=[rng.integers(0,N,N) for _ in range(1000)]
pairs=[];excl=0
for i in range(len(names)):
    for j in range(i+1,len(names)):
        d=W[names[i]]-W[names[j]]; bs=np.array([d[k].mean() for k in idx]); lo,hi=np.percentile(bs,[2.5,97.5])
        pairs.append((names[i],names[j],float(d.mean()),float(lo),float(hi))); excl+= (lo>0 or hi<0)
OUT['paired']=dict(n_pairs=len(pairs),n_excluding_zero=int(excl),pairs=pairs)
for n in names: RES[n]['UD (news/mixed)'].pop('wc')
json.dump(OUT,open('results.json','w'),indent=1,default=float)
print('DONE',flush=True)
