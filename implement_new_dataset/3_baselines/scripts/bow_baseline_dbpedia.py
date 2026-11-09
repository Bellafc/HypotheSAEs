"""Bag-of-words baseline on the DBpedia holdout, mirroring kogan_baseline_10k.py (post hoc; does not change any
HypotheSAEs result or setting).

Linear SVM (LinearSVC, the classification analogue of Kogan et al.'s linear SVR) on bag-of-words features
(log1p TF or TF-IDF; unigrams or unigrams+bigrams). Fit on 12k train, choose C on val AUC, score the holdout once
per configuration. Reference rows: logistic regression on text-embedding-3-small and on the 50 LASSO-selected SAE
neurons (M=256, K=8) with sklearn's default C=1, the same setting as every logistic fit in the repo, and the HypotheSAEs headline holdout AUC
(50 hypotheses annotated by gpt-4o-mini; in-sample logit on holdout, as in the paper).

Usage: python implement_new_dataset/3_baselines/scripts/bow_baseline_dbpedia.py
Output: implement_new_dataset/3_baselines/results/3.1_plain_models/bow_baseline_dbpedia_holdout.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.embedding import load_embedding_cache  # noqa: E402
from hypothesaes.sae import load_model  # noqa: E402

S = ["train", "val", "holdout"]
DATA = os.path.join(ROOT, "data/dbpedia/processed/12k")
HSAE = os.path.join(ROOT, "implement_new_dataset/2_pipeline/results/dbpedia")  # HypotheSAEs results read here
dfs = {s: pd.read_json(os.path.join(DATA, f"{s}.json"), lines=True) for s in S}
texts = {s: d["text"].tolist() for s, d in dfs.items()}
y = {s: d["label"].values for s, d in dfs.items()}
rows = []


def fit_default(make, X, name):
    """one fit at sklearn's default C=1 (no tuning), as in run_pipeline.fit_metric"""
    m = make(1.0).fit(X["train"], y["train"])
    rows.append({"model": name, "C": 1.0, "val_auc": roc_auc_score(y["val"], m.decision_function(X["val"])),
                 "holdout_auc": roc_auc_score(y["holdout"], m.decision_function(X["holdout"]))})
    print(rows[-1], flush=True)


def fit_select(make, X, name):
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        m = make(C).fit(X["train"], y["train"])
        v = roc_auc_score(y["val"], m.decision_function(X["val"]))
        if best is None or v > best[0]:
            best = (v, C, m)
    v, C, m = best
    rows.append({"model": name, "C": C, "val_auc": v, "holdout_auc": roc_auc_score(y["holdout"], m.decision_function(X["holdout"]))})
    print(rows[-1], flush=True)


svm = lambda C: LinearSVC(C=C, max_iter=20000, random_state=0)
logit = lambda C: LogisticRegression(C=C, max_iter=5000)
for ngram in [(1, 1), (1, 2)]:
    cv = CountVectorizer(ngram_range=ngram, min_df=5, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    Cn = {s: cv.fit_transform(texts[s]) if s == "train" else cv.transform(texts[s]) for s in S}
    print(f"ngram {ngram}: vocab {len(cv.vocabulary_)}", flush=True)
    tf = {s: normalize(Cn[s].astype(float).log1p()) for s in S}
    tt = TfidfTransformer(sublinear_tf=True).fit(Cn["train"])
    tfidf = {s: tt.transform(Cn[s]) for s in S}
    g = "unigram" if ngram == (1, 1) else "uni+bigram"
    for fname, F in [("log1p TF", tf), ("TF-IDF", tfidf)]:
        fit_select(svm, F, f"linear SVM, {fname}, {g}")

t2e = load_embedding_cache("dbpedia_text-embedding-3-small")
E = {s: np.stack([t2e[t] for t in texts[s]]).astype(np.float32) for s in S}
del t2e
fit_default(logit, E, "logistic on text-embedding-3-small")
met = json.load(open(os.path.join(HSAE, "metrics_M256_K8_lasso_n50.json")))
sae = load_model(os.path.join(ROOT, "checkpoints/dbpedia_text-embedding-3-small_12k/SAE_M=256_K=8.pt"))
A = {s: sae.get_activations(E[s])[:, met["selected_neurons"]] for s in S}
fit_default(logit, A, "logistic on 50 LASSO-selected SAE neurons (HypotheSAEs features)")
auc = json.load(open(os.path.join(HSAE, "holdout_metrics.json")))["metrics"]["auroc"]
rows.append({"model": "HypotheSAEs 50 hypotheses (gpt-4o-mini annotations, in-sample logit on holdout)",
             "holdout_auc": auc})
res = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_colwidth", 80)
print(res.round(4).to_string(index=False))
res.to_csv(os.path.join(ROOT, "implement_new_dataset/3_baselines/results/3.1_plain_models/bow_baseline_dbpedia_holdout.csv"), index=False)
