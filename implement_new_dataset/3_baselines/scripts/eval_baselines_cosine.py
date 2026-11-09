"""Preliminary, (almost) free comparison of HypotheSAEs vs BERTopic / HypoGeniC on the FULL validation sets.
Annotation proxy (task 5a): cosine(text-embedding-3-small(text), embedding(hypothesis)), z-scored; multivariate
logit (DBpedia) / OLS (10-K) fit on val as in score_hypotheses; Bonferroni 0.05/H. Only hypotheses are embedded
(~$0.001). Not a substitute for the gpt-4o-mini evaluation.
Output: implement_new_dataset/3_baselines/results/eval2_cosine_val/summary.csv. Usage (repo root): OPENAI_KEY_SAE=... python implement_new_dataset/3_baselines/scripts/eval_baselines_cosine.py"""
import json, os, sys, numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.embedding import get_openai_embeddings, load_embedding_cache
B = "implement_new_dataset/3_baselines/results"
H = {"dbpedia": {
        "HypotheSAEs": pd.read_csv("implement_new_dataset/2_pipeline/results/dbpedia/holdout_hypotheses.csv").hypothesis.tolist(),
        "BERTopic": json.load(open(f"{B}/3.2_bertopic/dbpedia/hypotheses.json"))["hypotheses"],
        "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/dbpedia/hypotheses.json"))["hypotheses"]},
     "10k": {
        "HypotheSAEs": [h["interpretation"] for h in json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))["hypotheses"]],
        "BERTopic": json.load(open(f"{B}/3.2_bertopic/10k/hypotheses.json"))["hypotheses"],
        "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/10k/hypotheses.json"))["hypotheses"]}}
cfg = {"dbpedia": ("data/dbpedia/processed/12k/val.json", "label", "dbpedia_text-embedding-3-small"),
       "10k": ("data/10k/val.json", "y", "10k_text-embedding-3-small")}
rows = []
for ds, (f, lab, cache) in cfg.items():
    d = pd.read_json(f, lines=True); t2e = load_embedding_cache(cache)
    T = np.stack([t2e[t] for t in d.text]); T /= np.linalg.norm(T, axis=1, keepdims=True); y = d[lab].values.astype(float)
    for m, hs in H[ds].items():
        h2e = get_openai_embeddings(hs, model="text-embedding-3-small", cache_name="baseline_hyps_text-embedding-3-small", show_progress=False)
        Hm = np.stack([h2e[h] for h in hs]); Hm /= np.linalg.norm(Hm, axis=1, keepdims=True)
        S = T @ Hm.T; X = sm.add_constant((S - S.mean(0)) / S.std(0))
        if ds == "dbpedia":
            r = sm.Logit(y, X).fit(disp=0, maxiter=1000, method="lbfgs"); metric = roc_auc_score(y, r.predict(X)); mname = "AUC"
        else:
            r = sm.OLS(y, X).fit(); metric = r.rsquared; mname = "R2"
        nsig = int((np.asarray(r.pvalues)[1:] < 0.05 / len(hs)).sum())
        rows.append({"dataset": ds, "method": m, "n_hyp": len(hs), mname: round(metric, 3), "n_sig_0.05/H": nsig})
        print(rows[-1], flush=True)
os.makedirs("implement_new_dataset/3_baselines/results/eval2_cosine_val", exist_ok=True); pd.DataFrame(rows).to_csv("implement_new_dataset/3_baselines/results/eval2_cosine_val/summary.csv", index=False)
