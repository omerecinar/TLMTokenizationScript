# Reproducibility package

Evaluation of Turkish tokenizers at the input-representation level (CPU only, no model weights).

    pip install transformers tokenizers sentencepiece tiktoken zemberek-python datasets scipy matplotlib
    # download UD test files tr_boun-ud-test.conllu and tr_imst-ud-test.conllu into this folder
    python run_all.py            # -> results.json
    python stats.py python stats_and_figures.py  # -> stats.json and figures/python stats_and_figures.py  # -> stats.json and figures/ python figs.py  # -> stats.json and figs/
    python mk_tables.py          # -> LaTeX table bodies

To add a tokenizer, add one entry to the HF (Hugging Face) or TT (tiktoken) dictionary in run_all.py.
