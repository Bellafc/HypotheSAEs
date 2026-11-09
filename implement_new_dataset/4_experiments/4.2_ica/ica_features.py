"""Replace the SAE with ICA (ICA Lens, Liu & Han 2026; ICA on embeddings, Yamagiwa et al. EMNLP 2023) and compare,
on our datasets and on the paper's datasets.

Datasets (--dataset)
  dbpedia, 10k   our data: OpenAI text-embedding-3-small cache, the SAE checkpoint of the main run (configs/<ds>.yaml)
  yelp, congress, headlines, wiki5, wiki15, bills
                 the paper's processed data (HuggingFace rmovva/HypotheSAEs, train + validation only, saved to
                 data/paper/), local Qwen/Qwen3-Embedding-0.6B embeddings (free; max 512 tokens), SAEs retrained
                 with the paper's settings (notebooks/experiment_reproduction.ipynb). Tasks as in the paper: yelp
                 regression on stars, congress classification on republican, headlines pairwise classification
                 (features of headline A minus headline B), wiki5 / wiki15 / bills classification on label_synthetic.

Feature sources (all fit on train only, then the SAME selection / interpretation / evaluation as HypotheSAEs):
  sae      the dataset's Top-K SAE
  ica_n    PCA to n dims -> FastICA (unit-variance whitening)                       n in {64, 128, 256}
  pca_n    PCA to n dims (decorrelation only, no non-Gaussianity)                   control
  raw      the embedding dimensions as they are                                    control
  rand_n   PCA-whitened n dims, then a random orthogonal rotation                   control (is ICA better than random?)
Every non-SAE component gets the same post-processing so it looks like SAE activations (the repo's sampling and
fidelity code assume non-negative, sparse activations): flip its sign so the long tail (positive skew) is positive,
then keep only the top of each component on train: z = max(s - tau, 0), tau = a train percentile of s
(90 / 97 / 99 on our data; 75 / 90 / 97 / 99 on the paper's real datasets, because some real targets are tied to
common features; 75 / 90 / 97 on the synthetic ones, because a feature kept on 1% of texts cannot match a topic that
covers 3-5% and LASSO on such sparse features took >10 min per fit).

Stages
  --stage screen   free. LASSO picks H features (dbpedia 50, 10k 20, the paper's H elsewhere) on train; validation
                   metrics without any LLM (feature activation > 0 stands in for an annotation): AUC or R^2 of the
                   continuous and of the 0/1 features, topic recovery for datasets with ground-truth topics (Hungarian
                   matching of positive topics to binarized features, F1), report-level R^2 on 10k.
                   -> screen_<ds>.csv; then summary.csv (best configuration of each kind per dataset) and
                   chosen.json (its name), picked by the dataset's criterion (0/1-feature metric, or recovery F1 on
                   the synthetic sets), since hypotheses are finally evaluated as 0/1 annotations.
  --stage llm      paid. For --method <kind> (its chosen configuration from screen): gpt-4o interpretations
                   (3 candidates, T=0.7), gpt-4o-mini fidelity scoring (200 examples), then gpt-4o-mini annotation of
                   100 validation texts: for dbpedia / 10k the SAME 100 used for the baselines
                   (implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/), for the other datasets 100 drawn once (seed 0) and saved
                   as <ds>_val100.json. A spend ledger enforces --budget across runs.
                   Sampling bins and task instructions: configs/<ds>.yaml for our data, the paper defaults otherwise.

Usage:
  python implement_new_dataset/4_experiments/4.2_ica/ica_features.py --stage screen --dataset dbpedia,10k,headlines
  OPENAI_KEY_SAE=... python implement_new_dataset/4_experiments/4.2_ica/ica_features.py --stage llm --method ica --dataset dbpedia --budget 11.5
Outputs: implement_new_dataset/4_experiments/4.2_ica/results/ (screen_<ds>.csv, summary.csv, chosen.json, hypotheses_<ds>_<method>.json,
         val100_summary.csv, usage_<ds>_<method>.csv, spend.json); paper embeddings in data/paper/<ds>/emb_*.npy;
         paper SAEs in checkpoints/paper_<ds>_Qwen3-Embedding-0.6B/
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import run_pipeline as P  # noqa: E402
from compare_selection import corr_matrix  # noqa: E402
from hypothesaes.embedding import load_embedding_cache  # noqa: E402
from hypothesaes.quickstart import train_sae  # noqa: E402
from hypothesaes.select_neurons import select_neurons  # noqa: E402

OUT = os.path.join(ROOT, "implement_new_dataset/4_experiments/4.2_ica/results")
NS = [64, 128, 256]
KINDS = ["sae", "ica", "pca", "rand", "raw"]
INTERP_MODEL, ANNOT_MODEL = "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"
# measured cost of one interpretation + scoring run + 100-text annotation; other datasets: rough guess until run once
EST_COST = {"dbpedia": 2.27 + 5000 * 0.000113, "10k": 2.03 + 2000 * 0.000139}
DEFAULT_EST_COST = 3.0

# our datasets: data, embeddings and SAE from configs/<ds>.yaml
OURS = {
    "dbpedia": dict(H=50, pcts=[90, 97, 99], topic="l3", select_by="val_auc_binary"),
    "10k": dict(H=20, pcts=[90, 97, 99], topic=None, select_by="val_r2_binary"),
}
# the paper's datasets
PAPER_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
PAPER_DATA = os.path.join(ROOT, "data/paper")
REAL, SYNTH = [75, 90, 97, 99], [75, 90, 97]
PAPER = {
    "yelp": dict(files=("yelp/train-200K.json", "yelp/val-10K.json"), text="text", label="stars", cls=False,
                 saes=[(1024, 32)], sae_kw={}, H=20, pcts=REAL, select_by="val_r2_binary"),
    "congress": dict(files=("congress/train-10sentence-114K.json", "congress/val-10sentence-16K.json"),
                     text="speech_text", label="republican", cls=True, saes=[(4096, 32)], sae_kw={}, H=20, pcts=REAL,
                     select_by="val_auc_binary"),
    "headlines": dict(files=("headlines/train-8.8K.json", "headlines/val-1.0K.json"), text=None,
                      label="label_pairwise", cls=True, saes=[(256, 8), (32, 4)], sae_kw={"n_epochs": 200, "patience": 5},
                      H=20, pcts=REAL, select_by="val_auc_binary"),
    # synthetic: label_synthetic = 1 for the K most common subtopics; the topics give a recovery metric
    "wiki5": dict(files=("wiki/train_K=5_N=7979.json", "wiki/validation_K=5_N=2000.json"), text="text", emb_dir="wiki",
                  label="label_synthetic", topic="subcategory", cls=True, saes=[(32, 4)], sae_kw={}, H=5, pcts=SYNTH,
                  select_by="val_recov_mean_f1"),
    "wiki15": dict(files=("wiki/train_K=15_N=7979.json", "wiki/validation_K=15_N=2000.json"), text="text",
                   emb_dir="wiki", label="label_synthetic", topic="subcategory", cls=True, saes=[(64, 4)], sae_kw={},
                   H=15, pcts=SYNTH, select_by="val_recov_mean_f1"),
    "bills": dict(files=("bills/train_K=5_N=16834.json", "bills/validation_K=5_N=2000.json"), text="summary",
                  label="label_synthetic", topic="subtopic", cls=True, saes=[(64, 4)], sae_kw={}, H=5, pcts=SYNTH,
                  select_by="val_recov_mean_f1"),
}
DATASETS = list(OURS) + list(PAPER)


# ------------------------------------------------------------------------------------------------ data
def load(ds):
    """One dict per dataset:
    E / texts      embeddings and texts of the units features are computed on (texts; headlines: unique headlines)
    df, y          the rows that are predicted (headlines: pairs, unit indices in pairs) and their labels
    topics         ground-truth topic per row (recovery), or None;  groups: val report ids (10k), or None
    sae            () -> SAE activations {train, val};  label / text: column names in df"""
    if ds in OURS:
        o, cfg = OURS[ds], P.load_config(ds)
        dfs, _, _ = P.load_data(cfg)
        df = {s: dfs[s] for s in ["train", "val"]}
        tc, lab = cfg["data"]["text_col"], cfg["data"]["label_col"]
        t2e = load_embedding_cache(cfg["cache_name"])
        E = {s: np.stack([t2e[t] for t in df[s][tc]]).astype(np.float32) for s in df}
        M, K = P.chosen_MK(cfg)
        return dict(ds=ds, cls=cfg["classification"], H=o["H"], pcts=o["pcts"], select_by=o["select_by"],
                    E=E, texts={s: df[s][tc].tolist() for s in df}, df=df, label=lab, text=tc, pairs=None,
                    y={s: df[s][lab].values for s in df},
                    topics={s: df[s][o["topic"]].values for s in df} if o["topic"] else None,
                    groups=df["val"][cfg["data"]["group_col"]].values if cfg["data"]["group_col"] else None,
                    sae=lambda: (lambda m: {s: m.get_activations(E[s]) for s in E})(P.get_sae(cfg, E, M, K)))
    c = PAPER[ds]
    from huggingface_hub import hf_hub_download
    df = dict(zip(["train", "val"], [pd.read_json(hf_hub_download("rmovva/HypotheSAEs", f, repo_type="dataset",
                                                                   local_dir=PAPER_DATA), lines=True)
                                     for f in c["files"]]))
    ed = os.path.join(PAPER_DATA, c.get("emb_dir", ds))
    os.makedirs(ed, exist_ok=True)
    if c["text"]:  # one text per row
        texts, pairs = {s: df[s][c["text"]].tolist() for s in df}, None
    else:  # headlines: features are fit on the unique headlines; a pair is represented by f(A) - f(B)
        texts = {s: sorted(set(d.headline_A) | set(d.headline_B)) for s, d in df.items()}
        idx = {s: {t: i for i, t in enumerate(texts[s])} for s in texts}
        pairs = {s: (np.array([idx[s][t] for t in d.headline_A]), np.array([idx[s][t] for t in d.headline_B]))
                 for s, d in df.items()}
    E = {s: embed_local(texts[s], os.path.join(ed, f"emb_{s}.npy")) for s in texts}
    ck = os.path.join(ROOT, "checkpoints", f"paper_{ds}_{PAPER_EMBEDDER.split('/')[-1]}")

    def sae():
        ms = [train_sae(embeddings=E["train"], M=M, K=K, checkpoint_dir=ck, val_embeddings=E["val"], **c["sae_kw"])
              for M, K in c["saes"]]
        return {s: np.concatenate([m.get_activations(E[s]) for m in ms], 1) for s in E}
    return dict(ds=ds, cls=c["cls"], H=c["H"], pcts=c["pcts"], select_by=c["select_by"], E=E, texts=texts, df=df,
                label=c["label"], text=c["text"], pairs=pairs, y={s: df[s][c["label"]].values for s in df},
                topics={s: df[s][c["topic"]].values for s in df} if "topic" in c else None, groups=None, sae=sae)


def embed_local(texts, path):
    if os.path.exists(path):
        return np.load(path)
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(PAPER_EMBEDDER, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    E = m.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    np.save(path, E)
    del m
    torch.cuda.empty_cache()
    return E


# ------------------------------------------------------------------------------------------------ features
def truncate(S_tr, S_va, pct):
    sign = np.sign(skew(S_tr, axis=0)); sign[sign == 0] = 1
    S_tr, S_va = S_tr * sign, S_va * sign
    tau = np.percentile(S_tr, pct, axis=0)
    return np.maximum(S_tr - tau, 0), np.maximum(S_va - tau, 0)


def components(kind, n, E, seed=0):
    """untruncated components for train and val, plus fit seconds"""
    t = time.time()
    if kind == "raw":
        mu = E["train"].mean(0)
        return E["train"] - mu, E["val"] - mu, time.time() - t
    pca = PCA(n_components=n, random_state=seed).fit(E["train"])
    Z = {s: pca.transform(E[s]) for s in ["train", "val"]}
    if kind == "ica":
        ica = FastICA(n_components=n, whiten="unit-variance", max_iter=2000, tol=1e-4, random_state=seed).fit(Z["train"])
        Z = {s: ica.transform(Z[s]) for s in Z}
    elif kind == "rand":
        sd = Z["train"].std(0)
        Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
        Z = {s: (Z[s] / sd) @ Q for s in Z}
    return Z["train"], Z["val"], time.time() - t


def name_of(kind, n, pct):
    return "sae" if kind == "sae" else (f"raw_p{pct}" if n is None else f"{kind}_{n}_p{pct}")


def parse(name):
    """'ica_256_p99' -> ('ica', 256, 99); 'raw_p90' -> ('raw', None, 90); 'sae' -> ('sae', None, None)"""
    p = name.split("_")
    if p[0] == "sae":
        return "sae", None, None
    return p[0], (int(p[1]) if len(p) == 3 else None), int(p[-1][1:])


def features(D, kind, n, pct):
    if kind == "sae":
        return D["sae"]()
    a, b, _ = truncate(*components(kind, n, D["E"])[:2], pct)
    return {"train": a, "val": b}


def rows(D, F, binary):
    """unit features -> row features (0/1 if binary); headlines: f(A) - f(B)"""
    X = {s: (F[s] > 0).astype(np.float32) if binary else F[s] for s in F}
    if D["pairs"] is None:
        return X
    return {s: X[s][D["pairs"][s][0]] - X[s][D["pairs"][s][1]] for s in X}


def select(D, F):
    kw = {"classification": True} if D["cls"] else {"classification": False, "verbose": False}
    sel, coefs = select_neurons(activations=rows(D, F, False)["train"], target=D["y"]["train"], n_select=D["H"],
                                method="lasso", **kw)
    return [int(i) for i in sel], [float(c) for c in coefs]


def recovery(topics_val, y_train, topics_train, B):
    """paper's synthetic metric with 0/1 columns B in place of hypotheses: Hungarian matching (on correlation) between
    the positive topics and B's columns; mean F1 of the matched pairs and #pairs with F1 >= 0.5"""
    pos = sorted(set(topics_train[y_train == 1]))
    R = np.stack([(topics_val == t).astype(float) for t in pos], 1)
    ri, ci = linear_sum_assignment(-np.nan_to_num(corr_matrix(R, B)))
    f1 = np.array([2 * (R[:, i] * B[:, j]).sum() / max(R[:, i].sum() + B[:, j].sum(), 1) for i, j in zip(ri, ci)])
    return float(f1.mean()), int((f1 >= 0.5).sum()), len(pos)


# ------------------------------------------------------------------------------------------------ screen
def evaluate(D, F):
    t = time.time()
    sel, _ = select(D, F)
    y = D["y"]
    r = {"select_sec": round(time.time() - t, 1), "mean_prevalence_selected": float((F["train"][:, sel] > 0).mean())}
    for tag, binary in [("cont", False), ("binary", True)]:
        X = rows(D, F, binary)
        a, b = X["train"][:, sel], X["val"][:, sel]
        if D["cls"]:
            r[f"val_auc_{tag}"] = roc_auc_score(y["val"], LogisticRegression(max_iter=3000).fit(a, y["train"])
                                                .predict_proba(b)[:, 1])
        else:
            p = LinearRegression().fit(a, y["train"]).predict(b)
            r[f"val_r2_{tag}"] = P.r2(y["val"], p)
            if D["groups"] is not None and not binary:  # 10k: chunk predictions averaged per report
                g = pd.DataFrame({"k": D["groups"], "y": y["val"], "p": p}).groupby("k").mean()
                r["val_r2_report"] = P.r2(g.y.values, g.p.values)
    if D["topics"] is not None:
        f1, n_ok, n_t = recovery(D["topics"]["val"], y["train"], D["topics"]["train"],
                                 (F["val"][:, sel] > 0).astype(float))
        r.update({"val_recov_mean_f1": f1, "val_recov_n_f1_ge_0.5": n_ok, "n_topics": n_t})
    return r


def screen(ds):
    D = load(ds)
    print(ds, {s: D["E"][s].shape for s in D["E"]}, flush=True)
    configs = [("sae", None, None)] + [("raw", None, p) for p in D["pcts"]] + \
              [(k, n, p) for k in ["ica", "pca", "rand"] for n in NS for p in D["pcts"]]
    out, comp = [], {}
    for kind, n, pct in configs:
        t = time.time()
        if kind == "sae":
            F, sec = D["sae"](), None
        else:
            if (kind, n) not in comp:
                comp = {(kind, n): components(kind, n, D["E"])}  # keep one fit in memory
            S_tr, S_va, sec = comp[(kind, n)]
            a, b = truncate(S_tr, S_va, pct)
            F = {"train": a, "val": b}
        r = {"dataset": ds, "features": name_of(kind, n, pct), "kind": kind, "n_features": F["train"].shape[1],
             "trunc_pct": pct, "fit_sec": None if sec is None else round(sec, 1)}
        r.update(evaluate(D, F))
        r["total_sec"] = round(time.time() - t, 1)
        out.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, f"screen_{ds}.csv"), index=False)


def summarize():
    """best configuration of each kind per dataset, by the dataset's criterion -> summary.csv, chosen.json"""
    summ, chosen = [], {}
    for ds in DATASETS:
        p = os.path.join(OUT, f"screen_{ds}.csv")
        if not os.path.exists(p):
            continue
        res, crit = pd.read_csv(p), (OURS.get(ds) or PAPER[ds])["select_by"]
        for k in KINDS:
            best = res[res.kind == k].sort_values(crit, ascending=False).iloc[0]
            summ.append(best.to_dict())
            if k != "sae":
                chosen.setdefault(ds, {})[k] = best.features
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    json.dump(chosen, open(os.path.join(OUT, "chosen.json"), "w"), indent=2)
    print(pd.DataFrame(summ).round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------ llm
def val100(D):
    """the 100 validation rows annotated in the llm stage"""
    if D["ds"] in OURS:  # shared with the baselines
        return pd.read_json(os.path.join(ROOT, f"implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/{D['ds']}_val100.json"),
                            lines=True)
    p = os.path.join(OUT, f"{D['ds']}_val100.json")
    if not os.path.exists(p):
        D["df"]["val"].sample(100, random_state=0).to_json(p, orient="records", lines=True)
    return pd.read_json(p, lines=True)


def downstream(Mx, y, cls, topics=None):
    """in-sample fit of the 100 rows on the hypothesis columns (non-constant ones): Logit AUC or OLS R^2 / adj. R^2,
    plus topic recovery of the 0/1 columns (same metrics as downstream() in batch_all.py)"""
    import statsmodels.api as sm
    keep = [j for j in range(Mx.shape[1]) if Mx[:, j].std() > 0]
    X = sm.add_constant(Mx[:, keep].astype(float))
    if not cls:
        r = sm.OLS(y, X).fit()
        return {"R2": r.rsquared, "adj_R2": r.rsquared_adj}
    r = sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=2000)
    res = {"AUC": roc_auc_score(y, r.predict(X))}
    if topics is not None:
        f1, n_ok, _ = recovery(topics, y, topics, Mx.astype(float))
        res.update({"recovery_F1": f1, "classes_F1>=0.5": n_ok})
    return res


def stage_llm(ds, method, budget):
    from hypothesaes.interpret_neurons import (NeuronInterpreter, InterpretConfig, ScoringConfig, SamplingConfig,
                                               LLMConfig, sample_percentile_bins)
    import hypothesaes.annotate as A
    from hypothesaes.evaluation import score_hypotheses

    ledger_p = os.path.join(OUT, "spend.json")
    ledger = json.load(open(ledger_p)) if os.path.exists(ledger_p) else {}
    key, est = f"{ds}_{method}", EST_COST.get(ds, DEFAULT_EST_COST)
    if key in ledger:
        print(f"{key} already done (ledger)"); return
    if sum(ledger.values()) + est > budget:
        raise SystemExit(f"budget guard: spent ${sum(ledger.values()):.2f} + estimated ${est:.2f} > ${budget:.2f}; "
                         f"not running {key}")
    chosen = json.load(open(os.path.join(OUT, "chosen.json")))[ds][method]
    D = load(ds)
    F = features(D, *parse(chosen))
    sel, coefs = select(D, F)
    pcfg = P.load_config(ds) if ds in OURS else P.PAPER_DEFAULTS  # sampling bins and task instructions
    pct, task = P.pct_kwargs(pcfg), pcfg["interpretation"]["task_instructions"]
    P.durable_annotation_log("valsub100")
    tracker = P.UsageTracker()
    interp = NeuronInterpreter(interpreter_model=INTERP_MODEL, annotator_model=ANNOT_MODEL,
                               n_workers_interpretation=10, n_workers_annotation=120,
                               cache_name=f"ica_features_{ds}_{chosen}")
    t0 = time.time()
    tracker.phase = "interpretation"
    cfg = InterpretConfig(sampling=SamplingConfig(function=sample_percentile_bins, n_examples=20, sampling_kwargs=pct),
                          llm=LLMConfig(temperature=0.7), n_candidates=3,
                          **({"task_specific_instructions": task} if task else {}))
    texts = D["texts"]["train"]
    interps = interp.interpret_neurons(texts=texts, activations=F["train"], neuron_indices=sel, config=cfg)
    tracker.phase = "scoring"
    fid = interp.score_interpretations(texts=texts, activations=F["train"], interpretations=interps,
                                       config=ScoringConfig(n_examples=200, sampling_function=sample_percentile_bins,
                                                            sampling_kwargs=pct), temperature=0.0)
    coef_of = dict(zip(sel, coefs))
    hyps = []
    for idx, f in fid.items():
        best = max(interps[idx], key=lambda s: f[s]["f1"] if s is not None else -1)
        hyps.append({"feature_idx": int(idx), "lasso_coef": coef_of[idx], "interpretation": best,
                     "f1_fidelity": float(f[best]["f1"])})
    json.dump({"features": chosen, "selected": sel, "hypotheses": hyps,
               "all_candidates": {str(k): v for k, v in interps.items()},
               "all_fidelity": {str(k): v for k, v in fid.items()}},
              open(os.path.join(OUT, f"hypotheses_{ds}_{method}.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # annotate 100 validation rows (same annotator, prompt and cache as the baselines)
    tracker.phase = "val100_eval"
    v, hs = val100(D), [h["interpretation"] for h in hyps]
    units = sorted(set(v.headline_A) | set(v.headline_B)) if D["pairs"] is not None else v[D["text"]].tolist()
    ann = A.annotate_texts_with_concepts(texts=units, concepts=hs, model=ANNOT_MODEL, cache_name="valsub100",
                                         max_words_per_example=256, n_workers=64, show_progress=False, temperature=0.0)
    U = np.stack([ann[h] for h in hs], 1)
    if D["pairs"] is None:
        Mx = U
    else:  # headlines: hypothesis(A) - hypothesis(B)
        ix = {t: i for i, t in enumerate(units)}
        Mx = U[[ix[t] for t in v.headline_A]] - U[[ix[t] for t in v.headline_B]]
    y = v[D["label"]].values.astype(float)
    keep = {h: Mx[:, j] for j, h in enumerate(hs) if Mx[:, j].std() > 0}
    _, det = score_hypotheses(keep, y, classification=D["cls"], corrected_pval_threshold=0.05)
    topic = (OURS.get(ds) or PAPER[ds]).get("topic")
    usage = tracker.report()
    cost = float(usage.cost_usd.iloc[-1]) if len(usage) else 0.0
    row = {"dataset": ds, "method": f"{method.upper()} ({chosen})", "n_hyp": len(hs),
           "mean_fidelity_f1": float(np.mean([h["f1_fidelity"] for h in hyps])),
           "prevalence": float(U.mean()), "n_sig_0.05/H": int((det.regression_pval < 0.05 / len(hs)).sum()),
           **downstream(Mx, y, D["cls"], v[topic].values if topic else None),
           "cost_usd": cost, "wall_min": (time.time() - t0) / 60}
    sp = os.path.join(OUT, "val100_summary.csv")
    res = pd.concat([pd.read_csv(sp), pd.DataFrame([row])]) if os.path.exists(sp) else pd.DataFrame([row])
    res.to_csv(sp, index=False)
    usage.to_csv(os.path.join(OUT, f"usage_{ds}_{method}.csv"), index=False)
    ledger[key] = cost
    json.dump(ledger, open(ledger_p, "w"), indent=2)
    print(row, flush=True)
    print(usage.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="ICA (and PCA / raw / random-rotation controls) in place of the SAE")
    ap.add_argument("--stage", choices=["screen", "llm"], required=True)
    ap.add_argument("--dataset", required=True, help=f"comma-separated for screen; one for llm. {', '.join(DATASETS)}")
    ap.add_argument("--method", choices=["ica", "pca", "rand", "raw"], default="ica", help="llm stage")
    ap.add_argument("--budget", type=float, default=11.5, help="max total $ across all llm runs (spend.json ledger)")
    args = ap.parse_args()
    dss = args.dataset.split(",")
    if bad := [d for d in dss if d not in DATASETS]:
        raise SystemExit(f"unknown dataset(s) {bad}; choose from {DATASETS}")
    os.makedirs(OUT, exist_ok=True)
    if args.stage == "screen":
        for ds in dss:
            screen(ds)
        summarize()
    else:
        if len(dss) != 1:
            raise SystemExit("the llm stage takes one dataset")
        stage_llm(dss[0], args.method, args.budget)


if __name__ == "__main__":
    main()
