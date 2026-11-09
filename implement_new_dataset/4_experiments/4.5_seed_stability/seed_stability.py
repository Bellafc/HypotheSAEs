"""Stability of HypotheSAEs across SAE random seeds (no API calls).
For each dataset config (configs/dbpedia.yaml, configs/10k.yaml) the SAE is retrained with 5 seeds at the config's
(M, K) (10-K: the grid stage's choice in chosen_MK.json), neurons are selected as in the pipeline (LASSO, n_select of
the config) and evaluated on val with the pipeline's metric (AUC / R^2). Stability of the SELECTED features is measured
by activation similarity on val: for each selected neuron of seed a, the best Pearson correlation with any selected
neuron of seed b; a feature is 'reproduced' if r >= 0.7 (or 0.5).
Usage: python implement_new_dataset/4_experiments/4.5_seed_stability/seed_stability.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import run_pipeline as P  # noqa: E402

OUT = os.path.join(P.ROOT, "implement_new_dataset/4_experiments/4.5_seed_stability/results")
rows, pairs = [], []
for ds in ["dbpedia", "10k"]:
    cfg = P.load_config(ds)
    _, texts, y = P.load_data(cfg)
    E = P.embed(cfg, texts)
    M, K = P.chosen_MK(cfg)
    acts = {}
    for seed in range(5):
        torch.manual_seed(seed); np.random.seed(seed)
        cfg["checkpoint_dir"] = os.path.join(P.ROOT, "checkpoints", f"seedstab_{ds}_seed{seed}")
        sae = P.get_sae(cfg, E, M, K)
        A = {s: sae.get_activations(E[s]) for s in ["train", "val"]}
        sel, _ = P.select(cfg, A["train"], y["train"])
        metric = P.fit_metric(cfg, A["train"][:, sel], y["train"], A["val"][:, sel], y["val"])
        acts[seed] = A["val"][:, sel]
        rows.append({"dataset": ds, "seed": seed, "val_metric": metric, "dead": int((A["train"].max(0) == 0).sum())})
        print(rows[-1], flush=True)
    for a, b in itertools.permutations(range(5), 2):
        X, Y = acts[a], acts[b]
        C = np.nan_to_num(np.corrcoef(np.concatenate([X, Y], 1).T)[:X.shape[1], X.shape[1]:])
        best = C.max(1)
        pairs.append({"dataset": ds, "a": a, "b": b, "reproduced_r>=0.7": (best >= 0.7).mean(),
                      "reproduced_r>=0.5": (best >= 0.5).mean(), "median_best_r": np.median(best)})
R, Pr = pd.DataFrame(rows), pd.DataFrame(pairs)
os.makedirs(OUT, exist_ok=True)
R.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
Pr.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
print(R.groupby("dataset").val_metric.agg(["mean", "std", "min", "max"]).round(4))
print(Pr.groupby("dataset")[["reproduced_r>=0.7", "reproduced_r>=0.5", "median_best_r"]].mean().round(3))
