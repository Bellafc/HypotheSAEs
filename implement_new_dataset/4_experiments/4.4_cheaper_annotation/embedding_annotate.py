"""Benchmark embedding-similarity "annotations" against LLM annotations on the DBpedia heldout set.

For each hypothesis h and heldout text x: score(x, h) = cos(emb(x), emb(h)).
Holdout text embeddings are reused from the cache; only the hypotheses are newly embedded.

Questions:
  1) How correlated are cosine scores with the LLM's yes/no annotations? (per-hypothesis AUC, point-biserial r,
     and agreement/kappa after binarizing each hypothesis at the LLM's prevalence)
  2) Same conclusions about which hypotheses are predictive? (separation scores, Bonferroni-significant sets,
     rank correlation, overall AUC), for both binarized and continuous cosine annotations.

Usage:
    python implement_new_dataset/4_experiments/4.4_cheaper_annotation/embedding_annotate.py --embedder text-embedding-3-small --backend openai
    python implement_new_dataset/4_experiments/4.4_cheaper_annotation/embedding_annotate.py --embedder Qwen/Qwen3-Embedding-0.6B --backend local
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.annotate import annotate, CACHE_DIR  # noqa: E402
from hypothesaes.embedding import load_embedding_cache  # noqa: E402
from hypothesaes.evaluation import score_hypotheses  # noqa: E402

DEFAULT_HYP = ("implement_new_dataset/2_pipeline/results/dbpedia/"
               "holdout_hypotheses.csv")
REF_CACHE = "dbpedia_text-embedding-3-small"  # gpt-4o-mini T=0 annotations
QWEN_INSTRUCT = ("Instruct: Given a description of a topic, retrieve Wikipedia article openings "
                 "that match the description\nQuery:")
DATASETS = {  # dbpedia = default (unchanged)
    "dbpedia": {"holdout": "data/dbpedia/processed/{size}/holdout.json", "hyp_csv": DEFAULT_HYP, "ref_cache": REF_CACHE,
                "emb_prefix": "dbpedia", "label": "label", "classification": True,
                "out": "implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/dbpedia/cosine", "instruct": QWEN_INSTRUCT},
    "10k": {"holdout": "data/10k/holdout.json", "hyp_csv": "implement_new_dataset/2_pipeline/results/10k/holdout_hypotheses.csv",
               "ref_cache": "10k_text-embedding-3-small", "emb_prefix": "10k", "label": "y", "classification": False,
               "out": "implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/10k/cosine",
               "instruct": ("Instruct: Given a description of a property of a text, retrieve excerpts from 10-K MD&A "
                            "sections that have this property\nQuery:")},
}


def embed_hypotheses(hyps, args):
    t = time.time()
    if args.backend == "openai":
        from hypothesaes.embedding import get_openai_embeddings
        t2e = get_openai_embeddings(hyps, model=args.embedder, cache_name=f"{args.cfg['emb_prefix']}_hyps_{args.embedder}",
                                    show_progress=False)
        H = np.stack([t2e[h] for h in hyps])
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(args.embedder, device="cuda")
        prompt = args.cfg["instruct"] + " " if args.qwen_instruct else None
        H = m.encode(hyps, prompt=prompt, batch_size=64)
    return np.asarray(H, dtype=np.float32), time.time() - t


def logit_eval(X, y, names, alpha=0.05, classification=True):  # paper: Bonferroni 0.05/H
    """multivariate logit/OLS like the repo's score_hypotheses: in-sample AUC (or R^2) + Bonferroni-significant set."""
    Xc = sm.add_constant(X)
    res = sm.Logit(y, Xc).fit(disp=0, maxiter=500) if classification else sm.OLS(y, Xc).fit()
    p = np.asarray(res.pvalues)[1:]
    sig = {n for n, pv in zip(names, p) if pv < alpha / len(names)}
    metric = roc_auc_score(y, res.predict(Xc)) if classification else float(res.rsquared)
    return metric, sig


def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="12k")
    ap.add_argument("--dataset", default="dbpedia", choices=list(DATASETS))
    ap.add_argument("--embedder", default="text-embedding-3-small")
    ap.add_argument("--backend", default="openai", choices=["openai", "local"])
    ap.add_argument("--hyp_csv", default=None)
    ap.add_argument("--no_qwen_instruct", dest="qwen_instruct", action="store_false",
                    help="embed hypotheses without the Qwen3 query instruction")
    args = ap.parse_args()
    args.cfg = c = {k: (v.format(size=args.size) if isinstance(v, str) else v) for k, v in DATASETS[args.dataset].items()}
    cls = c["classification"]
    tag = args.embedder.split("/")[-1] + ("" if args.backend == "openai" or args.qwen_instruct else "_noinstruct")

    hyps = pd.read_csv(os.path.join(ROOT, args.hyp_csv or c["hyp_csv"]))["hypothesis"].tolist()
    ho = pd.read_json(os.path.join(ROOT, c["holdout"]), lines=True)
    texts, y = ho["text"].tolist(), ho[c["label"]].values

    # LLM reference annotations (already cached, no API calls)
    ref = annotate(tasks=[(t, h) for t in texts for h in hyps], model="gpt-4o-mini", show_progress=False,
                   use_cache_only=True, cache_path=os.path.join(CACHE_DIR, f"{c['ref_cache']}_hypothesis-eval.json"))
    L = np.stack([np.array([ref[h][t] for t in texts]) for h in hyps], 1)  # n x H, 0/1

    # embedding scores
    emb_cache = f"{c['emb_prefix']}_{args.embedder.split('/')[-1]}"
    t2e = load_embedding_cache(emb_cache)
    if any(t not in t2e for t in texts):  # e.g. Qwen embeddings of 10-K holdout texts not computed yet
        if args.backend == "openai":
            from hypothesaes.embedding import get_openai_embeddings
            t2e = get_openai_embeddings(texts, model=args.embedder, cache_name=emb_cache)
        else:
            from hypothesaes.embedding import get_local_embeddings
            t2e = get_local_embeddings(texts, model=args.embedder, batch_size=32, cache_name=emb_cache)
    T = np.stack([t2e[t] for t in texts]).astype(np.float32)
    t0 = time.time()
    H, t_hyp = embed_hypotheses(hyps, args)
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True); Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    S = Tn @ Hn.T  # n x H cosine
    t_total = time.time() - t0

    # (1) agreement with LLM, per hypothesis
    rows = []
    B = np.zeros_like(L)
    for j, h in enumerate(hyps):
        s, l = S[:, j], L[:, j]
        prev = l.mean()
        k = max(int(round(prev * len(s))), 1)
        thr = np.sort(s)[-k]
        B[:, j] = (s >= thr).astype(int)  # binarize at the LLM's prevalence
        rows.append({"hypothesis": h, "llm_prevalence": prev,
                     "auc_cos_vs_llm": roc_auc_score(l, s) if 0 < prev < 1 else np.nan,
                     "r_pointbiserial": pearsonr(s, l)[0] if 0 < prev < 1 else np.nan,
                     "agree_binarized": (B[:, j] == l).mean(), "kappa_binarized": kappa(B[:, j], l),
                     "corr_cos_y": pearsonr(s, y)[0]})
    per = pd.DataFrame(rows)

    # (2) conclusions about predictiveness
    m_llm, d_llm = score_hypotheses({h: L[:, j] for j, h in enumerate(hyps)}, y, classification=cls, corrected_pval_threshold=0.05)
    m_bin, d_bin = score_hypotheses({h: B[:, j] for j, h in enumerate(hyps)}, y, classification=cls, corrected_pval_threshold=0.05)
    d_llm, d_bin = d_llm.set_index("hypothesis").loc[hyps], d_bin.set_index("hypothesis").loc[hyps]
    thr_llm, thr_bin = m_llm["Significant"][2], m_bin["Significant"][2]
    sig_llm = set(d_llm.index[d_llm.regression_pval < thr_llm])
    sig_bin = set(d_bin.index[d_bin.regression_pval < thr_bin])
    Sz = (S - S.mean(0)) / S.std(0)
    auc_cont, sig_cont = logit_eval(Sz, y, hyps, classification=cls)
    per["sep_llm"] = d_llm["separation_score"].values
    per["sep_emb_binarized"] = d_bin["separation_score"].values

    summary = {
        "embedder": args.embedder, "hypothesis_prompt": "qwen3 instruct" if args.backend == "local" and args.qwen_instruct else "none",
        "time_sec_embedding_scores": round(t_total, 2), "time_sec_embed_hypotheses": round(t_hyp, 2),
        "n_hypotheses": len(hyps), "n_texts": len(texts),
        # (1) agreement
        "mean_auc_cos_vs_llm": per["auc_cos_vs_llm"].mean(), "median_auc_cos_vs_llm": per["auc_cos_vs_llm"].median(),
        "mean_r_pointbiserial": per["r_pointbiserial"].mean(),
        "mean_kappa_binarized": per["kappa_binarized"].mean(),
        "pooled_agree_binarized": float((B == L).mean()),
        # (2) conclusions
        ("auc_llm" if cls else "r2_llm"): m_llm["auroc" if cls else "r2"],
        ("auc_emb_binarized" if cls else "r2_emb_binarized"): m_bin["auroc" if cls else "r2"],
        ("auc_emb_continuous" if cls else "r2_emb_continuous"): auc_cont,
        "n_sig_llm": len(sig_llm), "n_sig_emb_binarized": len(sig_bin), "n_sig_emb_continuous": len(sig_cont),
        "sig_overlap_llm_binarized": len(sig_llm & sig_bin), "sig_overlap_llm_continuous": len(sig_llm & sig_cont),
        "spearman_sep_llm_vs_emb_binarized": spearmanr(per.sep_llm, per.sep_emb_binarized, nan_policy="omit")[0],
        "spearman_sep_llm_vs_corr_cos_y": spearmanr(per.sep_llm, per.corr_cos_y, nan_policy="omit")[0],
        "sign_agreement_sep": float(np.mean(np.sign(per.sep_llm) == np.sign(per.corr_cos_y))),
    }
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(f"=== {tag} ===")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}, indent=2))
    print("\nWorst-agreeing hypotheses (AUC of cosine vs LLM labels):")
    print(per.sort_values("auc_cos_vs_llm")[["hypothesis", "llm_prevalence", "auc_cos_vs_llm", "kappa_binarized",
                                              "sep_llm", "sep_emb_binarized"]].head(6).round(3).to_string(index=False))
    print("\nBest-agreeing:")
    print(per.sort_values("auc_cos_vs_llm", ascending=False)[["hypothesis", "auc_cos_vs_llm", "kappa_binarized",
                                                              "sep_llm", "sep_emb_binarized"]].head(4).round(3).to_string(index=False))

    od = os.path.join(ROOT, c["out"])
    os.makedirs(od, exist_ok=True)
    per.to_csv(os.path.join(od, f"per_hypothesis_{tag}.csv"), index=False)
    json.dump(summary, open(os.path.join(od, f"summary_{tag}.json"), "w"), indent=2, default=float)
    np.save(os.path.join(od, f"cosine_{tag}.npy"), S)
    print("saved to", od)


if __name__ == "__main__":
    main()
