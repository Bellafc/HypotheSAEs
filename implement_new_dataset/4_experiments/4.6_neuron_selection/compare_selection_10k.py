"""Compare the same 9 neuron-selection methods as compare_selection.py (DBpedia) on 10-K, train/val only.

Methods (shared with compare_selection.py): {correlation, separation_score, lasso} x {as in the repo, positive-direction
only, 80% positive quota}, plus stability_lasso (LASSO on 30 random half-samples; stability selection, Meinshausen & Buehlmann 2010).
10-K is a regression task, so LASSO is the regression LASSO (classification=False) and the signed variants use a
binary search on the Lasso alpha.

Metrics on validation (neuron activation stands in for an LLM annotation of the neuron's hypothesis):
  val_r2_cont    OLS on the selected neurons' activations (train fit -> val R^2)
  val_r2_binary  same with activations binarized (> 0), closer to 0/1 LLM annotations
  val_r2_report  continuous-model predictions averaged per report (10-K reports have several chunks in val)
  n_pos_dir, mean prevalence, mean |pairwise correlation|
  overlap file: shared count / Jaccard for every pair of methods (same as compare_selection.py)

Usage:
    python implement_new_dataset/4_experiments/4.6_neuron_selection/compare_selection_10k.py          # uses implement_new_dataset/2_pipeline/results/10k/chosen_MK.json (M512 K8)
Outputs: implement_new_dataset/4_experiments/4.6_neuron_selection/results/10k/compare_selection_M{M}_K{K}.csv, compare_selection_overlap_M{M}_K{K}.csv,
         compare_selection_sets_M{M}_K{K}.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.embedding import load_embedding_cache  # noqa: E402
from hypothesaes.select_neurons import select_neurons  # noqa: E402
import run_pipeline as P  # noqa: E402  (config, data, SAE checkpoints)
from compare_selection import make_methods, overlap_table  # noqa: E402  (the same 9 methods as on DBpedia)

def stability_lasso(A, y, n, B=30, frac=0.5):
    """10th method (formerly the "altsel" stage): LASSO on 30 random half-samples, keep the n most often chosen."""
    rng = np.random.default_rng(0)
    cnt = np.zeros(A.shape[1])
    for _ in range(B):
        ii = rng.choice(len(y), int(frac * len(y)), replace=False)
        s, _ = select_neurons(activations=A[ii], target=y[ii], n_select=n, method="lasso", classification=False,
                              verbose=False)
        cnt[s] += 1
    return [int(i) for i in np.argsort(-cnt)[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_selects", default="20,50")
    args = ap.parse_args()
    cfg = P.load_config("10k")
    out = os.path.join(paths.exp_dir("4.6_neuron_selection"), "results/10k")
    os.makedirs(out, exist_ok=True)
    M, K = P.chosen_MK(cfg)
    dfs, _, y = P.load_data(cfg)
    dfs = {s: dfs[s] for s in ["train", "val"]}
    t2e = load_embedding_cache(cfg["cache_name"])
    E = {s: np.stack([t2e[t] for t in d["text"]]).astype(np.float32) for s, d in dfs.items()}
    del t2e
    sae = P.get_sae(cfg, E, M, K)  # loads the existing checkpoint
    A = {s: sae.get_activations(E[s]) for s in E}
    keys = dfs["val"]["key"].values if "key" in dfs["val"] else None
    corr = np.nan_to_num(np.array([np.corrcoef(A["train"][:, j], y["train"])[0, 1]
                                   if A["train"][:, j].std() > 0 else 0 for j in range(M)]))

    methods = make_methods(0.8, classification=False)
    methods["stability_lasso"] = stability_lasso
    rows, sets = [], {}
    for n in [int(x) for x in args.n_selects.split(",")]:
        for name, fn in methods.items():
            t = time.time()
            sel = fn(A["train"], y["train"], n)
            sets[(name, n)] = set(sel)
            m = LinearRegression().fit(A["train"][:, sel], y["train"])
            pv = m.predict(A["val"][:, sel])
            mb = LinearRegression().fit((A["train"][:, sel] > 0).astype(float), y["train"])
            pb = mb.predict((A["val"][:, sel] > 0).astype(float))
            r = {"method": name, "n_select": n, "train_r2_cont": P.r2(y["train"], m.predict(A["train"][:, sel])),
                 "val_r2_cont": P.r2(y["val"], pv), "val_r2_binary": P.r2(y["val"], pb)}
            if keys is not None:
                g = pd.DataFrame({"k": keys, "y": y["val"], "p": pv}).groupby("k").mean()
                r["val_r2_report"] = P.r2(g.y.values, g.p.values)
            C = np.corrcoef(A["train"][:, sel].T)
            iu = np.triu_indices(len(sel), 1)
            r.update({"n_pos_dir": int(np.sum(corr[sel] > 0)),
                      "mean_prevalence": float((A["train"][:, sel] > 0).mean()),
                      "mean_abs_pairwise_corr": float(np.nanmean(np.abs(C[iu]))),
                      "select_sec": round(time.time() - t, 1)})
            rows.append(r)
            print(f"done {name} n={n}: val_r2_cont={r['val_r2_cont']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\n=== 10-K / M={M} K={K} (validation) ===")
    print(res.round(4).to_string(index=False))
    print("\nOverlap of selected neurons (shared count / Jaccard):")
    ov = overlap_table(sets)
    print(ov.to_string(index=False))
    res.to_csv(os.path.join(out, f"compare_selection_M{M}_K{K}.csv"), index=False)
    ov.to_csv(os.path.join(out, f"compare_selection_overlap_M{M}_K{K}.csv"), index=False)
    json.dump({f"{k[0]}_n{k[1]}": sorted(v) for k, v in sets.items()},
              open(os.path.join(out, f"compare_selection_sets_M{M}_K{K}.json"), "w"))
    print("saved to", out)


if __name__ == "__main__":
    main()
