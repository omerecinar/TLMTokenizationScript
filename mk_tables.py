import json
RN="R\\'enyi"
O=json.load(open('results.json')); T=O['tok']; S=json.load(open('stats.json')); E=json.load(open('examples.json'))
U=lambda n:T[n]['UD (news/mixed)']
LBL={'YTU COSMOS BERT (raw)':'YTU COSMOS (raw input)','YTU COSMOS BERT (lowercased)':'YTU COSMOS (lowercased)','Gemma 2':'Gemma 2$^{a}$','Qwen3':'Qwen3$^{b}$','Llama 3':'Llama 3$^{c}$','Toraman morph-BPE':'Toraman morph-BPE$^{\\dagger}$','GPT-4o (o200k)':'GPT-4o (o200k)','GPT-4 (cl100k)':'GPT-4 (cl100k)'}
L=lambda n:LBL.get(n,n)
order=['BERTurk','YTU COSMOS BERT (raw)','YTU COSMOS BERT (lowercased)','TURNA','TurkishBERTweet','Toraman morph-BPE','XLM-R','mT5','Kumru','Trendyol-LLM 7B v1.0','Commencis-LLM','Llama 3','Qwen3','Gemma 2','Gemma 3','Mistral v0.3','GPT-4o (o200k)','GPT-4 (cl100k)']
grp_order=['Turkish PLM','Multilingual PLM','Turkish LLM','Multilingual LLM']
GN={'Turkish PLM':'Turkish pre-trained encoders / encoder--decoders','Multilingual PLM':'Multilingual pre-trained encoders / encoder--decoders','Turkish LLM':'Turkish-oriented LLMs','Multilingual LLM':'Multilingual LLMs'}
fmt=lambda x:f'{x:,.0f}'.replace(',','{,}')
def pct(x,d=1): return f'{100*x:.{d}f}'
TAB={}
foot_shared="$^{a}$Shared by Turkish-Gemma-9b-T1. $^{b}$Shared by Trendyol-LLM-8B-T1. $^{c}$Shared by Turkish-Llama-8b. $^{\\dagger}$Reusability failure (Section~\\ref{sec:practical}); boundary scores are not comparable."
# Table 1: tools
info={'BERTurk':('Turkish','WordPiece','\\citet{schweter2020berturk}'),'YTU COSMOS BERT (raw)':('Turkish','WordPiece (uncased)','\\citet{cosmosbert}'),'TURNA':('Turkish','SentencePiece Unigram','\\citet{uludogan2024turna}'),'TurkishBERTweet':('Turkish (social media)','byte-level BPE','\\citet{najafi2024turkishbertweet}'),'Toraman morph-BPE':('Turkish','morphology-based BPE','\\citet{toraman2023impact}'),'XLM-R':('100 languages','SentencePiece Unigram','\\citet{conneau2020xlmr}'),'mT5':('101 languages','SentencePiece Unigram','\\citet{xue2021mt5}'),'Kumru':('Turkish','byte-level BPE (from scratch)','\\citet{kumru2025}'),'Trendyol-LLM 7B v1.0':('Turkish','extended Mistral SentencePiece','\\citet{trendyol}'),'Commencis-LLM':('Turkish','Mistral SentencePiece','\\citet{commencis}'),'Llama 3':('multilingual','byte-level BPE','\\citet{grattafiori2024llama3}'),'Qwen3':('multilingual','byte-level BPE','\\citet{yang2025qwen3}'),'Gemma 2':('multilingual','SentencePiece','\\citet{gemma2024gemma2}'),'Gemma 3':('multilingual','SentencePiece','\\citet{gemma2025gemma3}'),'Mistral v0.3':('multilingual','SentencePiece (byte fallback)','\\citet{jiang2023mistral}'),'GPT-4o (o200k)':('multilingual','byte-level BPE (tiktoken)','\\citet{tiktoken}'),'GPT-4 (cl100k)':('multilingual','byte-level BPE (tiktoken)','\\citet{tiktoken}')}
rows=[]
for g in grp_order:
    rows.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{GN[g]}}}}}\\\\')
    for n in order:
        if n=='YTU COSMOS BERT (lowercased)' or T[n]['group']!=g: continue
        lab=L(n).replace(' (raw input)','')
        a,b,c=info[n]; rows.append(f'{lab} & {a} & {b} & {fmt(T[n]["vocab"])} & {c}\\\\')
TAB['t1']=('\\begin{tabular}{lllrl}\n\\toprule\nTokenizer & Language focus & Algorithm & Vocabulary & Source\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t1_caption']='Tokenizers evaluated in this study. Zemberek \\citep{akin2007zemberek} serves as the rule-based reference; the ITU Turkish NLP pipeline \\citep{eryigit2014itu} is discussed qualitatively only because it requires a personal access key.'
TAB['t1_foot']=foot_shared.split(' $^{\\dagger}')[0]
# Table 2 zemberek
Z=O['zemberek']
TAB['t2']=('\\begin{tabular}{lr}\n\\toprule\nMetric & Value\\\\\n\\midrule\n'+
 f'Analysis rate (word tokens) & {pct(Z["analysis_rate"])}\\%\\\\\nMean ambiguity (analyses per analysed word) & {Z["ambiguity"]:.2f}\\\\\nLemma coverage (oracle upper bound) & {pct(Z["lemma_oracle"])}\\%\\\\\n'+
 f'Throughput (words/s, single CPU core) & {fmt(S["zemberek_speed_fresh"])}\\\\\nModel loading time (s) & {S["zemberek_load"]:.1f}\\\\\nTypes with a segmentation that concatenates to the word & {fmt(Z["aligned_types"])} / {fmt(Z["analysed_types"])}\\\\\n'+'\\bottomrule\n\\end{tabular}')
TAB['t2_caption']='Rule-based morphological analysis with Zemberek on the UD test sets (18{,}140 word tokens, 9{,}406 word types).'
# Table 3 main
rows=[]
for g in grp_order:
    rows.append(f'\\multicolumn{{10}}{{l}}{{\\textit{{{GN[g]}}}}}\\\\')
    for n in order:
        if T[n]['group']!=g: continue
        D=U(n)
        rows.append(f'{L(n)} & {D["fertility"]:.3f} & {pct(D["single"])} & {pct(D["unk"],2)} & {D["P"]:.3f} & {D["R"]:.3f} & {D["F1"]:.3f} & {D["renyi"]:.3f} & {D["spearman_morph_tokens"][0]:.3f} & {T[n]["parallel"]["premium"]:.2f}\\\\')
TAB['t3']=('\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nTokenizer & Fert. & Single (\\%) & UNK (\\%) & P & R & F1 & '+RN+' & $\\rho_{m}$ & TR/EN\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t3_caption']='Tokenization efficiency and morpheme-boundary alignment on the UD test sets. Fert.\\ = fertility; Single = share of words kept as one token; P/R/F1 = morpheme-boundary precision, recall and F1 over 8{,}846 aligned word types; '+RN+' = '+RN+' efficiency ($\\alpha=2.5$); $\\rho_{m}$ = word-level Spearman correlation between morpheme count and token count; TR/EN = Turkish-to-English token ratio on WMT16 newstest2016.'
TAB['t3_foot']=foot_shared
# Table 4 CIs
rows=[]
for n in order:
    D=U(n)
    c=lambda k:f'[{D[k][0]:.3f}, {D[k][1]:.3f}]'
    rows.append(f'{L(n)} & {D["fertility"]:.3f} {c("fert_ci")} & {D["P"]:.3f} {c("P_ci")} & {D["R"]:.3f} {c("R_ci")} & {D["F1"]:.3f} {c("F1_ci")}\\\\')
TAB['t4']=('\\begin{tabular}{lllll}\n\\toprule\nTokenizer & Fertility & Precision & Recall & F1\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t4_caption']='Point estimates with 95\\% percentile bootstrap confidence intervals (1{,}000 resamples; word tokens for fertility, aligned word types for boundary scores).'
# Table 5 bins
sel=[n for n in order if n!='Toraman morph-BPE']
rows=[]
for n in sel:
    b=U(n)['bins']; inc=(b['4']/b['1']-1)*100
    rows.append(f'{L(n)} & {b["1"]:.3f} & {b["2"]:.3f} & {b["3"]:.3f} & {b["4"]:.3f} & {inc:.0f}\\%\\\\')
bs=U('BERTurk')['bin_sizes']
TAB['t5']=('\\begin{tabular}{lrrrrr}\n\\toprule\nTokenizer & 1 morph. & 2 morph. & 3 morph. & 4+ morph. & Increase\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t5_caption']=f'Fertility by number of surface morphemes and relative increase from one to four or more morphemes (group sizes: {fmt(bs["1"])}, {fmt(bs["2"])}, {fmt(bs["3"])} and {fmt(bs["4"])} word tokens).'
# Table 6 domains
rows=[]
for n in sel:
    r=[]
    for d in ['UD (news/mixed)','Tweets','Product reviews']:
        r+= [f'{T[n][d]["fertility"]:.3f}', f'{T[n][d]["F1"]:.3f}']
    rows.append(f'{L(n)} & '+' & '.join(r)+'\\\\')
dd=O['domains']
TAB['t6']=('\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{UD (news/mixed)} & \\multicolumn{2}{c}{Tweets} & \\multicolumn{2}{c}{Product reviews}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nTokenizer & Fert. & F1 & Fert. & F1 & Fert. & F1\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t6_caption']=f'Cross-domain robustness: fertility and boundary F1 on the UD test sets ({fmt(dd["UD (news/mixed)"]["word_tokens"])} words), tweets ({fmt(dd["Tweets"]["word_tokens"])} words) and product reviews ({fmt(dd["Product reviews"]["word_tokens"])} words).'
# Table 7 cost
llm=[n for n in order if T[n]['group'] in ('Turkish LLM','Multilingual LLM')]
llm=sorted(llm,key=lambda n:U(n)['fertility'])
base=U('GPT-4o (o200k)')['fertility']
rows=[]
for n in llm:
    f=U(n)['fertility']; p=T[n]['parallel']
    rows.append(f'{L(n)} & {f*1000:,.0f} & {p["tr_tok_per_word"]:.2f} & {p["en_tok_per_word"]:.2f} & {p["premium"]:.2f} & {(p["premium"]-1)*100:+.0f}\\% & {f:.2f} & {(f/base-1)*100:+.0f}\\%\\\\'.replace(',','{,}'))
TAB['t7']=('\\begin{tabular}{lrrrrrrr}\n\\toprule\n & Tokens per & \\multicolumn{2}{c}{Tokens per word (WMT16)} & & Turkish & Cost per $10^{6}$ & vs.\\\\\n\\cmidrule(lr){3-4}\nTokenizer & 1{,}000 words & TR & EN & TR/EN & surcharge & words (USD) & GPT-4o\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t7_caption']='Converting fertility into inference cost for LLM tokenizers. Cost is computed with Equation~(\\ref{eq:cost}) at a normalised price of $P=1$~USD per million tokens, so it scales linearly with any actual tariff. The Turkish surcharge is the relative increase in tokens for the same content compared with English.'
TAB['t7_foot']=foot_shared.split(' $^{\\dagger}')[0]
# Table 8 spearman model-level
lab={'fert_vs_single':'Fertility vs. single-token share','fert_vs_P':'Fertility vs. boundary precision','fert_vs_R':'Fertility vs. boundary recall','fert_vs_F1':'Fertility vs. boundary F1','fert_vs_renyi':'Fertility vs. '+RN+' efficiency','fert_vs_premium':'Fertility vs. TR/EN token ratio','fert_vs_vocab':'Fertility vs. vocabulary size','ud_vs_Tweets':'Fertility rank: UD vs. tweets','ud_vs_Product reviews':'Fertility rank: UD vs. product reviews'}
def pfmt(p): return '$<$0.001' if p<0.001 else f'{p:.3f}'
rows=[f'{lab[k]} & {S[k][0]:.3f} & {pfmt(S[k][1])}\\\\' for k in lab]
TAB['t8']=('\\begin{tabular}{lrr}\n\\toprule\nPair of quantities & Spearman $\\rho$ & $p$\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t8_caption']=f'Tokenizer-level Spearman correlations across the {S["n_models"]} tokenizers with a functional standard interface (Toraman morph-BPE and the raw-input COSMOS condition excluded).'
# Table 9 cetvel
C=S['cetvel']
rows=[]
for n,(m,avg,mt,gec) in C['models'].items():
    rows.append(f'{L(n)} & {m} & {U(n)["fertility"]:.3f} & {U(n)["F1"]:.3f} & {avg:.1f} & {mt:.1f} & {gec:.1f}\\\\')
rows.append('\\midrule')
for k,kl in [('fertility','Fertility'),('F1','Boundary F1'),('premium','TR/EN ratio'),('renyi',RN+' eff.')]:
    rows.append(f'\\multicolumn{{4}}{{l}}{{$\\rho$ (exact $p$): {kl}}} & '+' & '.join(f'{C[k+"_vs_"+t][0]:+.2f} ({C[k+"_vs_"+t][1]:.2f})' for t in ['avg','mt','gec'])+'\\\\')
TAB['t9']=('\\begin{tabular}{llrrrrr}\n\\toprule\nTokenizer & Cetvel model & Fert. & F1 & Avg. & MT & GEC\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t9_caption']='Exploratory link to downstream performance. Cetvel scores of 7--8B base models are taken from \\citet{er2025cetvel} (Table~2 therein); one model per tokenizer. Spearman $\\rho$ with exact permutation $p$-values ($n=5$).'
# Table 10 examples
rows=[]
for r in E['examples']:
    cell=lambda k:(' $|$ '.join(r[k]) if r[k] else '---').replace("'","'")
    rows.append(f'{r["word"]} & {cell("BERTurk")} & {cell("mT5")} & {cell("Kumru")} & {cell("GPT-4o")} & {cell("Zemberek")}\\\\')
TAB['t10']=('\\begin{tabular}{llllll}\n\\toprule\nWord & BERTurk & mT5 & Kumru & GPT-4o (o200k) & Zemberek\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}')
TAB['t10_caption']='Segmentation of a word family with increasing inflection density. Pieces are separated by $|$; ``\\_'' denotes a stand-alone word-boundary piece. Zemberek outputs lower-cased morphemes and drops the apostrophe, so \\textit{T\\"urkiye\'nin} cannot be aligned.'
# Table 11 Toraman
tp=O['toraman_preseg']; d=U('Toraman morph-BPE')
TAB['t11']=('\\begin{tabular}{lrr}\n\\toprule\nInput condition & Fertility & 95\\% CI\\\\\n\\midrule\n'+
 f'Raw word input & {d["fertility"]:.3f} & [{d["fert_ci"][0]:.3f}, {d["fert_ci"][1]:.3f}]\\\\\nPre-segmented with Zemberek & {tp["fertility"]:.3f} & [{tp["ci"][0]:.3f}, {tp["ci"][1]:.3f}]\\\\\n'+'\\bottomrule\n\\end{tabular}')
TAB['t11_caption']='Fertility of the morphology-based BPE tokenizer under two preprocessing conditions.'
json.dump(TAB,open('shared/tables.json','w'),indent=1,ensure_ascii=False)
for k in ['t3','t7','t10']: print(TAB[k][:1500]); print()
