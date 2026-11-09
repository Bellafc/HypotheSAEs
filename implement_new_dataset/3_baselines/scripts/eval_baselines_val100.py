"""Evaluate HypotheSAEs, BERTopic and HypoGeniC hypotheses on the SAME 100 random validation texts per dataset,
with the same annotator as the holdout (gpt-4o-mini-2024-07-18, T=0, repo prompt, 256 words) and the repo's
score_hypotheses (logit for DBpedia, OLS for 10-K); significance at the paper's Bonferroni threshold 0.05/H.
Every finished annotation is appended to a JSONL log at once (durable), and merged into the cache on restart.
DBpedia also gets a recovery-style F1: Hungarian matching (correlation) between hypotheses and the positive l3
classes PRESENT in the 100 texts; mean F1 of matched pairs (no surface-similarity LLM calls).
Output: implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/. Annotation cache name: valsub100.
Usage (repo root): OPENAI_KEY_SAE=... python implement_new_dataset/3_baselines/scripts/eval_baselines_val100.py"""
import json, os, sys, threading
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import hypothesaes.annotate as A
from hypothesaes.evaluation import score_hypotheses

B = "implement_new_dataset/3_baselines/results"
OUT = "implement_new_dataset/3_baselines/results/eval1_llm_labels_val100"; os.makedirs(OUT, exist_ok=True)
MODEL = "gpt-4o-mini-2024-07-18"
HYP = {"dbpedia": {
        "HypotheSAEs": pd.read_csv("implement_new_dataset/2_pipeline/results/dbpedia/holdout_hypotheses.csv").hypothesis.tolist(),
        "BERTopic": json.load(open(f"{B}/3.2_bertopic/dbpedia/hypotheses.json"))["hypotheses"],
        "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/dbpedia/hypotheses.json"))["hypotheses"]},
       "10k": {
        "HypotheSAEs": [h["interpretation"] for h in json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))["hypotheses"]],
        "BERTopic": json.load(open(f"{B}/3.2_bertopic/10k/hypotheses.json"))["hypotheses"],
        "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/10k/hypotheses.json"))["hypotheses"]}}
SRC = {"dbpedia": ("data/dbpedia/processed/12k/val.json", "label", True),
       "10k": ("data/10k/val.json", "y", False)}


def val100(ds, f):
    """the 100 validation texts: first a pool (DBpedia: 500 random val texts; 10-K: one random chunk per val report,
    427 texts), then 100 random texts from that pool. (Until 2026-11-09 the pool was saved as data/val500/; same texts.)"""
    v = pd.read_json(f, lines=True, dtype={"cik": str})
    pool = (v.sample(500, random_state=2026) if ds == "dbpedia" else v.groupby("key").sample(1, random_state=2026))
    return pool.reset_index(drop=True).sample(100, random_state=7).reset_index(drop=True)
CACHE = "valsub100"
cpath = os.path.join(A.CACHE_DIR, f"{CACHE}_hypothesis-eval.json"); lpath = cpath.replace(".json", ".log.jsonl")
if os.path.exists(lpath):
    cache = A.get_annotation_cache(cpath)
    for line in open(lpath):
        try: r = json.loads(line); cache[r["k"]] = r["v"]
        except Exception: pass
    A.save_annotation_cache(cpath, cache)
lock, logf = threading.Lock(), open(lpath, "a"); orig = A._store_annotation
def store(results, concept, text, annotation, cache=None):
    orig(results, concept, text, annotation, cache)
    if cache is not None and annotation is not None:
        with lock: logf.write(json.dumps({"k": A.generate_cache_key(concept, text), "v": annotation}) + "\n"); logf.flush()
A._store_annotation = store

rows = []
for ds, (f, lab, cls) in SRC.items():
    d = val100(ds, f)
    d.to_json(f"{OUT}/{ds}_val100.json", orient="records", lines=True)
    y = d[lab].values.astype(float); texts = d.text.tolist()
    for m, hs in HYP[ds].items():
        ann = A.annotate_texts_with_concepts(texts=texts, concepts=hs, model=MODEL, cache_name=CACHE, max_words_per_example=256,
                                             n_workers=64, show_progress=False, temperature=0.0)
        keep = {h: v for h, v in ann.items() if 0 < v.mean() < 1}  # constant columns cannot enter the regression
        met, det = score_hypotheses(keep, y, classification=cls, corrected_pval_threshold=0.05)
        det = det.assign(method=m, dataset=ds); det.to_csv(f"{OUT}/{ds}_{m}_hypotheses.csv", index=False)
        thr = 0.05 / len(hs)  # paper: Bonferroni over all H generated hypotheses
        row = {"dataset": ds, "method": m, "n_hyp": len(hs), "n_nonconstant": len(keep),
               ("AUC" if cls else "R2"): round(float(met["auroc" if cls else "r2"]), 3),
               "n_sig_0.05/H": int((det.regression_pval < thr).sum()),
               "mean_prevalence": round(float(np.mean([v.mean() for v in ann.values()])), 3)}
        if cls:  # recovery-style F1 vs positive l3 classes present in the sample
            classes = sorted(d.loc[d.label == 1, "l3"].unique())
            R = np.stack([(d.l3 == c).values.astype(float) for c in classes], 1)
            Hm = np.stack([ann[h] for h in hs], 1).astype(float)
            C = np.nan_to_num(np.corrcoef(np.concatenate([R, Hm], 1).T)[:len(classes), len(classes):])
            ri, ci = linear_sum_assignment(-C)
            f1 = []
            for i, j in zip(ri, ci):
                tp = (R[:, i] * Hm[:, j]).sum(); p = Hm[:, j].sum(); r_ = R[:, i].sum()
                f1.append(2 * tp / (p + r_) if p + r_ > 0 else 0.0)
            row.update({"classes_present": len(classes), "recovery_mean_F1": round(float(np.mean(f1)), 3),
                        "classes_F1>=0.5": int(np.sum(np.array(f1) >= 0.5))})
        rows.append(row); print(row, flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/summary.csv", index=False); print(res.to_string(index=False))
