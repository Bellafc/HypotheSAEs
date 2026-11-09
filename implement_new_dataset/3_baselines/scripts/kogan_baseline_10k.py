"""Kogan et al. (2009) style baseline on the 10-K holdout (post-hoc comparison; does not change any
HypotheSAEs result or setting).

Kogan et al.: linear-kernel support vector regression (epsilon-insensitive loss) on bag-of-words features of the
MD&A (TF = log(1+count) or TF-IDF, unigrams / unigrams+bigrams), optionally plus the log volatility of the
preceding 12 months ("history"). Here: fit on train chunks, choose C on val (paper used default SVMlight
settings), predict the holdout once per configuration. Out-of-sample R^2 on holdout.

For reference, the same out-of-sample protocol for: prevol only; ridge on text-embedding-3-small; OLS on the 20
LASSO-selected SAE neurons (M=512,K=8) of the HypotheSAEs run.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, RidgeCV
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import run_pipeline as P  # noqa: E402
from hypothesaes.embedding import load_embedding_cache  # noqa: E402

r2 = lambda y, p: 1 - np.mean((y - p) ** 2) / np.mean((y - np.mean(y)) ** 2)
cfg = P.load_config("10k")
dfs, texts, y = P.load_data(cfg)
S = ["train", "val", "holdout"]
pv = {s: dfs[s].prevol_c.values for s in S}
rows = []


def svr_fit(Xtr, Xva, Xho, name):
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        m = LinearSVR(C=C, epsilon=0.1, loss="epsilon_insensitive", max_iter=20000, random_state=0, dual=True).fit(Xtr, y["train"])
        v = r2(y["val"], m.predict(Xva))
        if best is None or v > best[0]: best = (v, C, m)
    v, C, m = best
    rows.append({"model": name, "C": C, "val_r2": v, "holdout_r2": r2(y["holdout"], m.predict(Xho))})
    print(rows[-1], flush=True)


for ngram in [(1, 1), (1, 2)]:
    cv = CountVectorizer(ngram_range=ngram, min_df=5, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    C = {s: cv.fit_transform(texts[s]) if s == "train" else cv.transform(texts[s]) for s in S}
    print(f"ngram {ngram}: vocab {len(cv.vocabulary_)}", flush=True)
    tf = {s: normalize(C[s].astype(float).log1p()) for s in S}
    tt = TfidfTransformer(sublinear_tf=True).fit(C["train"]); tfidf = {s: tt.transform(C[s]) for s in S}
    g = "unigram" if ngram == (1, 1) else "uni+bigram"
    for fname, F in [("log1p TF", tf), ("TF-IDF", tfidf)]:
        svr_fit(F["train"], F["val"], F["holdout"], f"Kogan SVR, {fname}, {g}")
        H = {s: hstack([F[s], csr_matrix(pv[s][:, None])]).tocsr() for s in S}
        svr_fit(H["train"], H["val"], H["holdout"], f"Kogan SVR, {fname}, {g} + history")

# reference models, same out-of-sample protocol
m = LinearRegression().fit(pv["train"][:, None], y["train"])
rows.append({"model": "history (prevol) only, OLS", "val_r2": r2(y["val"], m.predict(pv["val"][:, None])),
             "holdout_r2": r2(y["holdout"], m.predict(pv["holdout"][:, None]))})
t2e = load_embedding_cache(cfg["cache_name"])
E = {s: np.stack([t2e[t] for t in texts[s]]).astype(np.float32) for s in S}
rd = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(E["train"], y["train"])
rows.append({"model": "ridge on text-embedding-3-small", "val_r2": r2(y["val"], rd.predict(E["val"])),
             "holdout_r2": r2(y["holdout"], rd.predict(E["holdout"]))})
ck = json.load(open(os.path.join(cfg["output_dir"], "pre_eval_hypotheses.json")))
sae = P.get_sae(cfg, {s: E[s] for s in ["train", "val"]}, ck["M"], ck["K"])
A = {s: sae.get_activations(E[s])[:, ck["selected"]] for s in S}
ls = LinearRegression().fit(A["train"], y["train"])
rows.append({"model": "OLS on 20 LASSO-selected SAE neurons (HypotheSAEs features)", "val_r2": r2(y["val"], ls.predict(A["val"])),
             "holdout_r2": r2(y["holdout"], ls.predict(A["holdout"]))})
res = pd.DataFrame(rows); pd.set_option("display.width", 200, "display.max_colwidth", 70)
print(res.round(4).to_string(index=False))
res.to_csv(os.path.join(ROOT, "implement_new_dataset/3_baselines/results/3.1_plain_models/kogan_baseline_10k_holdout.csv"), index=False)
