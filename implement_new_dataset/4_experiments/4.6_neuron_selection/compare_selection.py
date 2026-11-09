"""Compare neuron-selection methods on already-trained DBpedia SAEs, without any LLM calls.

Because DBpedia has ground-truth l3 labels, we can evaluate selected neurons directly
(neuron activation > 0 stands in for an LLM annotation of the neuron's hypothesis):
  - overlap between methods (Jaccard of selected sets)
  - val AUC of a logistic model on the selected neurons (continuous and binarized)
  - positive coverage: share of val y=1 hit by >=1 positive-direction selected neuron, and false-hit rate on y=0
  - neuron-level recovery (paper metric, with neurons in place of hypotheses): Hungarian matching on the
    (positive l3 class x neuron) correlation matrix on val, mean F1 of matched pairs, #pairs with F1>=0.5

Usage:
    python implement_new_dataset/4_experiments/4.6_neuron_selection/compare_selection.py    # DBpedia 12k, text-embedding-3-small, M256 K8 (configs/dbpedia.yaml)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.embedding import load_embedding_cache  # noqa: E402
from hypothesaes.sae import load_model  # noqa: E402
from hypothesaes.select_neurons import select_neurons  # noqa: E402

BASE_METHODS = {"correlation": {}, "separation_score": {}, "lasso": {}}  # lasso kwargs set in make_methods


def signed_scores(A, y, method):
    """Per-neuron signed score for correlation / separation_score (all neurons)."""
    sel, sc = select_neurons(activations=A, target=y, n_select=A.shape[1], method=method)
    out = np.zeros(A.shape[1])
    out[np.array(sel)] = sc
    return out


def lasso_signed(A, y, n, sign, classification=True, max_iter=1000):
    """L1 model (same solver/standardization as the repo; logistic for classification, Lasso for regression) with a
    binary search on alpha until exactly n coefficients of the requested sign are nonzero; returns them by |coef|."""
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(A)
    lo, hi = (1e-6, 1e4) if classification else (1e-6, 1e2)
    best = None
    for _ in range(30 if classification else 40):
        alpha = np.sqrt(lo * hi)
        if classification:
            coef = LogisticRegression(penalty="l1", solver="liblinear", C=1 / alpha, max_iter=max_iter).fit(X, y).coef_[0]
        else:
            coef = Lasso(alpha=alpha, max_iter=2000).fit(X, y).coef_
        k = int(np.sum(np.sign(coef) == sign))
        if k >= n and (best is None or k < best[0]):
            best = (k, coef)
        if k == n:
            break
        if k < n:
            hi = alpha
        else:
            lo = alpha
    if best is None:
        raise RuntimeError(f"could not find {n} neurons with sign {sign}")
    coef = best[1]
    idx = np.where(np.sign(coef) == sign)[0]
    return idx[np.argsort(-np.abs(coef[idx]))][:n].tolist()


def make_methods(pos_frac, classification=True):
    """The 9 methods: {correlation, separation_score, lasso} x {repo version, positive-only, pos_frac quota}.
    name -> fn(A_train, y_train, n) -> list of neuron indices. Shared with compare_selection_10k.py."""
    lasso_kw = {"classification": True} if classification else {"classification": False, "verbose": False}
    def base(name, kw):
        return lambda A, y, n: [int(i) for i in select_neurons(activations=A, target=y, n_select=n,
                                                                 method=name, **kw)[0]]

    def by_sign(name, n_pos_fn):
        def fn(A, y, n):
            s = signed_scores(A, y, name)
            n_pos = n_pos_fn(n)
            pos = [i for i in np.argsort(-s) if s[i] > 0][:n_pos]
            neg = [i for i in np.argsort(s) if s[i] < 0][:n - n_pos]
            return [int(i) for i in pos + neg]
        return fn

    def lasso_by_sign(n_pos_fn):
        def fn(A, y, n):
            n_pos = n_pos_fn(n)
            sel = lasso_signed(A, y, n_pos, +1, classification)
            if n - n_pos > 0:
                sel += lasso_signed(A, y, n - n_pos, -1, classification)
            return [int(i) for i in sel]
        return fn

    only = lambda n: n
    quota = lambda n: int(round(pos_frac * n))
    q = f"quota{int(pos_frac * 100)}"
    methods = {name: base(name, kw) for name, kw in BASE_METHODS.items()}
    methods["lasso"] = base("lasso", lasso_kw)
    for name in ["correlation", "separation_score"]:
        methods[f"{name}_posonly"] = by_sign(name, only)
        methods[f"{name}_{q}"] = by_sign(name, quota)
    methods["lasso_posonly"] = lasso_by_sign(only)
    methods[f"lasso_{q}"] = lasso_by_sign(quota)
    return methods


def overlap_table(sets):
    """Pairwise overlap of the selected neuron sets, for every pair of methods at every n_select.
    sets: {(method, n_select): set of neuron indices}. Shared by compare_selection.py and compare_selection_10k.py."""
    rows = []
    names = list(dict.fromkeys(k[0] for k in sets))
    for n in sorted({k[1] for k in sets}):
        for i, x in enumerate(names):
            for z in names[i + 1:]:
                a, b = sets[(x, n)], sets[(z, n)]
                rows.append({"n_select": n, "pair": f"{x} vs {z}", "shared": len(a & b),
                             "jaccard": round(len(a & b) / len(a | b), 3)})
    return pd.DataFrame(rows)


def corr_matrix(Y, X):
    """Pearson correlation between columns of Y (n x a) and X (n x b) -> (a x b)."""
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return Y.T @ X / len(Y)


def evaluate(sel, A, y, l3, pos_classes, sign):
    tr, ho = A["train"][:, sel], A["val"][:, sel]
    res = {}
    clf = LogisticRegression(max_iter=3000).fit(tr, y["train"])
    res["auc_cont"] = roc_auc_score(y["val"], clf.predict_proba(ho)[:, 1])
    trb, hob = (tr > 0).astype(float), (ho > 0).astype(float)
    clf = LogisticRegression(max_iter=3000).fit(trb, y["train"])
    res["auc_binary"] = roc_auc_score(y["val"], clf.predict_proba(hob)[:, 1])

    pos_dir = sign[sel] > 0
    res["n_pos_dir"] = int(pos_dir.sum())
    hit = (hob[:, pos_dir] > 0).any(1) if pos_dir.any() else np.zeros(len(hob), bool)
    res["pos_coverage"] = hit[y["val"] == 1].mean()
    res["false_hit_y0"] = hit[y["val"] == 0].mean()

    # neuron-level recovery vs positive l3 classes (paper: Hungarian on correlation, report F1 of matches)
    R = np.stack([(l3["val"] == c).astype(float) for c in pos_classes], 1)  # n x 50
    C = np.nan_to_num(corr_matrix(R, hob))
    ri, ci = linear_sum_assignment(-C)
    f1s = []
    for i, j in zip(ri, ci):
        a, r = hob[:, j] > 0, R[:, i] > 0
        f1s.append(2 * (a & r).sum() / max(a.sum() + r.sum(), 1))
    res["recov_mean_f1"] = float(np.mean(f1s))
    res["recov_n_f1_ge_0.5"] = int(np.sum(np.array(f1s) >= 0.5))
    # classes that any selected neuron captures with F1>=0.5 (allows one neuron -> one class each, no matching)
    F = np.array([[2 * ((hob[:, j] > 0) & (R[:, i] > 0)).sum() / max((hob[:, j] > 0).sum() + R[:, i].sum(), 1)
                   for j in range(hob.shape[1])] for i in range(R.shape[1])])
    res["n_classes_any_f1_ge_0.5"] = int((F.max(1) >= 0.5).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n_selects", default="20,50")
    ap.add_argument("--pos_frac", type=float, default=0.8, help="share of positive-direction neurons in quota methods")
    args = ap.parse_args()

    import run_pipeline as P  # data, embedding cache and SAE checkpoint of configs/dbpedia.yaml
    cfg = P.load_config("dbpedia")
    size, emb_tag, cache_name = cfg["vars"]["size"], cfg["embedding"]["model"].split("/")[-1], cfg["cache_name"]
    data_dir = cfg["data"]["dir"]
    out_dir = os.path.join(paths.exp_dir("4.6_neuron_selection"), "results/dbpedia")
    os.makedirs(out_dir, exist_ok=True)

    dfs = {s: pd.read_json(os.path.join(data_dir, f"{s}.json"), lines=True) for s in ["train", "val"]}
    y = {s: d["label"].values for s, d in dfs.items()}
    l3 = {s: d["l3"].values for s, d in dfs.items()}
    pos_classes = json.load(open(os.path.join(ROOT, "data/dbpedia/positive_l3_classes.json")))["positive_l3_classes"]

    t2e = load_embedding_cache(cache_name)
    sae = load_model(os.path.join(cfg["checkpoint_dir"], f"SAE_M={args.M}_K={args.K}.pt"))
    A = {s: sae.get_activations(np.stack([t2e[t] for t in d["text"]]).astype(np.float32)) for s, d in dfs.items()}
    # direction of each neuron = sign of its train correlation with y (same rule for every method)
    sign = np.sign(np.nan_to_num(corr_matrix(y["train"][:, None].astype(float), A["train"])[0]))

    METHODS = make_methods(args.pos_frac)
    rows, sets = [], {}
    for n in [int(x) for x in args.n_selects.split(",")]:
        for name, fn in METHODS.items():
            t = time.time()
            sel = fn(A["train"], y["train"], n)
            sets[(name, n)] = set(sel)
            r = {"method": name, "n_select": n, "select_sec": round(time.time() - t, 1)}
            r.update(evaluate(sel, A, y, l3, pos_classes, sign))
            rows.append(r)
            print(f"done {name} n={n}", flush=True)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"\n=== {size} / {emb_tag} / M={args.M} K={args.K} ===")
    print(res.round(3).to_string(index=False))

    print("\nOverlap of selected neurons (shared count / Jaccard):")
    ov = overlap_table(sets)
    print(ov.to_string(index=False))

    tag = f"M{args.M}_K{args.K}"
    res.to_csv(os.path.join(out_dir, f"compare_selection_{tag}.csv"), index=False)
    ov.to_csv(os.path.join(out_dir, f"compare_selection_overlap_{tag}.csv"), index=False)
    with open(os.path.join(out_dir, f"compare_selection_sets_{tag}.json"), "w") as f:
        json.dump({f"{k[0]}_n{k[1]}": sorted(v) for k, v in sets.items()}, f)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
