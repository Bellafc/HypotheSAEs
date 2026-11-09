"""Does training-set fidelity (F1 of the interpretation vs. the neuron, gpt-4o-mini on 200 samples)
predict how good the hypothesis is on the holdout? Also: how well does the hypothesis annotation on the holdout track
its neuron (corr / F1 between neuron-active and LLM 'yes')? No API calls (all cached)."""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import hypothesaes.annotate as A
from hypothesaes.embedding import load_embedding_cache
from hypothesaes.quickstart import train_sae
OUT = "implement_new_dataset/4_experiments/4.3_fidelity/results"; os.makedirs(OUT, exist_ok=True); rows = []


def neuron_vs_annot(texts, hyps, neurons, cache_name, emb_cache, ckpt, M, K, train_texts):
    c = load_embedding_cache(emb_cache)
    Eh = np.stack([c[t] for t in texts]); Etr = np.stack([c[t] for t in train_texts[:2000]])
    sae = train_sae(embeddings=Etr, M=M, K=K, checkpoint_dir=ckpt, n_epochs=200, patience=5, show_progress=False)
    act = sae.get_activations(Eh)
    ann = A.annotate(tasks=[(t, h) for t in texts for h in hyps], model="gpt-4o-mini", use_cache_only=True, show_progress=False,
                     cache_path=os.path.join(A.CACHE_DIR, f"{cache_name}_hypothesis-eval.json"))
    out = []
    for h, n in zip(hyps, neurons):
        a = act[:, n] > 0; l = np.array([ann[h][t] for t in texts]) == 1
        corr = np.corrcoef(act[:, n], l)[0, 1] if l.std() > 0 and act[:, n].std() > 0 else np.nan
        f1 = 2 * (a & l).sum() / (a.sum() + l.sum()) if a.sum() + l.sum() else np.nan
        out.append({"holdout_corr_neuron_annot": corr, "holdout_F1_neuron_active_vs_yes": f1,
                    "neuron_active_rate": a.mean(), "annot_yes_rate": l.mean()})
    return pd.DataFrame(out)


# ---------- DBpedia (paper-style run, 50 hypotheses)
R = "implement_new_dataset/2_pipeline/results/dbpedia/"
h = pd.read_csv(R + "holdout_hypotheses.csv")
rec = pd.read_csv(R + "recovery_matches.csv")
db = h.merge(rec[["hypothesis", "f1_similarity", "surface_similarity"]], on="hypothesis", how="left")
ho = pd.read_json("data/dbpedia/processed/12k/holdout.json", lines=True); tr = pd.read_json("data/dbpedia/processed/12k/train.json", lines=True)
nv = neuron_vs_annot(ho.text.tolist(), db.hypothesis.tolist(), db.neuron_idx.tolist(), "dbpedia_text-embedding-3-small",
                     "dbpedia_text-embedding-3-small", "checkpoints/dbpedia_text-embedding-3-small_12k", 256, 8, tr.text.tolist())
db = pd.concat([db.reset_index(drop=True), nv], axis=1); db["abs_sep"] = db.separation_score.abs()
db.to_csv(f"{OUT}/dbpedia.csv", index=False)
for tgt in ["f1_similarity", "surface_similarity", "abs_sep", "holdout_F1_neuron_active_vs_yes"]:
    r = spearmanr(db.f1_fidelity, db[tgt], nan_policy="omit")
    rows.append({"dataset": "dbpedia", "x": "train fidelity F1", "y": tgt, "spearman": r[0], "p": r[1], "n": len(db)})
# ---------- 10-K (20 hypotheses)
pre = pd.DataFrame(json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))["hypotheses"])
ev = pd.read_csv("implement_new_dataset/2_pipeline/results/10k/holdout_hypotheses.csv")[["hypothesis", "separation_score", "regression_pval",
                                                         "feature_prevalence"]]
tk = pre.merge(ev, left_on="interpretation", right_on="hypothesis")
ho = pd.read_json("data/10k/holdout.json", lines=True); tr = pd.read_json("data/10k/train.json", lines=True)
ck = json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))
nv = neuron_vs_annot(ho.text.tolist(), tk.interpretation.tolist(), tk.neuron_idx.tolist(), "10k_text-embedding-3-small",
                     "10k_text-embedding-3-small", "checkpoints/10k_text-embedding-3-small", ck["M"], ck["K"], tr.text.tolist())
tk = pd.concat([tk.reset_index(drop=True), nv], axis=1); tk["abs_sep"] = tk.separation_score.abs(); tk["neglog_p"] = -np.log10(tk.regression_pval)
tk.to_csv(f"{OUT}/10k.csv", index=False)
for tgt in ["abs_sep", "neglog_p", "holdout_F1_neuron_active_vs_yes"]:
    r = spearmanr(tk.f1_fidelity, tk[tgt], nan_policy="omit")
    rows.append({"dataset": "10k", "x": "train fidelity F1", "y": tgt, "spearman": r[0], "p": r[1], "n": len(tk)})
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/summary.csv", index=False)
pd.set_option("display.width", 200); print(res.round(3).to_string(index=False))
for name, d, fcol in [("dbpedia", db, "f1_fidelity"), ("10k", tk, "f1_fidelity")]:
    print(name, "mean holdout F1(neuron active vs LLM yes) = %.3f, median corr = %.3f, train fidelity mean %.3f" %
          (d.holdout_F1_neuron_active_vs_yes.mean(), d.holdout_corr_neuron_annot.median(), d[fcol].mean()))
lo = db.nsmallest(5, "f1_fidelity")[["hypothesis", "f1_fidelity", "f1_similarity", "surface_similarity", "holdout_F1_neuron_active_vs_yes"]]
pd.set_option("display.max_colwidth", 70); print(lo.round(3).to_string(index=False))
