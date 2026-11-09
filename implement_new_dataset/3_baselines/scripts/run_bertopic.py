"""BERTopic baseline (HypotheSAEs paper App. D): OpenAI embeddings (text-embedding-3-small, cached), default
UMAP/HDBSCAN/c-TF-IDF; min_cluster_size in {10,20,50,100,200,500} chosen by validation performance of a predictor
on H topics selected with an L1 path (exactly H nonzero). Topics -> hypotheses: gpt-4o-2024-11-20 labels each selected
topic from its keywords, 10 topic documents and 10 documents from other topics (paper's Figure-8 prompt is not
reproduced verbatim; this prompt follows its description).
Needs the BERTopic venv (see implement_new_dataset/README.md) and the embeddings exported by export_embeddings.py.
Output: implement_new_dataset/3_baselines/results/3.2_bertopic/<dataset>/{grid.csv,hypotheses.json}.
Usage: OPENAI_API_KEY=... <venv_bertopic>/bin/python implement_new_dataset/3_baselines/scripts/run_bertopic.py dbpedia 50   (or 10k 20)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # implement_new_dataset/
import paths as _paths  # noqa: E402
ROOT = _paths.ROOT  # repo root

import json, os, sys, time
import numpy as np, pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.linear_model import lasso_path, LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from openai import OpenAI

name, H = sys.argv[1], int(sys.argv[2])
F = {"dbpedia": (f"{ROOT}/data/dbpedia/processed/12k/{{}}.json", "label", True),
     "10k": (f"{ROOT}/data/10k/{{}}.json", "y", False)}[name]
d = {s: pd.read_json(F[0].format(s), lines=True) for s in ["train", "val"]}
E = {s: np.load(f"{ROOT}/data/baselines/emb/{name}_{s}.npy") for s in d}
y = {s: d[s][F[1]].values.astype(float) for s in d}
cls = F[2]
out = f"{ROOT}/implement_new_dataset/3_baselines/results/3.2_bertopic/{name}"; os.makedirs(out, exist_ok=True)


def onehot(t, ids):
    return np.stack([(t == i).astype(float) for i in ids], 1)


def select_H(X, yy):
    if X.shape[1] <= H: return list(range(X.shape[1]))
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    alphas, coefs, _ = lasso_path(Xs, yy - yy.mean(), n_alphas=300, eps=1e-4)
    nnz = (coefs != 0).sum(0)
    j = int(np.argmax(nnz >= H)) if (nnz >= H).any() else len(nnz) - 1
    c = np.abs(coefs[:, j]); return list(np.argsort(-c)[:H])


def score(Xtr, Xva):
    if cls:
        m = LogisticRegression(max_iter=2000).fit(Xtr, y["train"]); return roc_auc_score(y["val"], m.predict_proba(Xva)[:, 1])
    m = LinearRegression().fit(Xtr, y["train"]); p = m.predict(Xva)
    return 1 - np.mean((y["val"] - p) ** 2) / np.var(y["val"])


rows, best = [], None
for mcs in [10, 20, 50, 100, 200, 500]:
    t = time.time()
    tm = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
                  hdbscan_model=HDBSCAN(min_cluster_size=mcs, metric="euclidean", cluster_selection_method="eom", prediction_data=True),
                  calculate_probabilities=False, verbose=False)
    ttr, _ = tm.fit_transform(d["train"].text.tolist(), E["train"])
    tva, _ = tm.transform(d["val"].text.tolist(), E["val"])
    ttr, tva = np.array(ttr), np.array(tva)
    ids = sorted(i for i in set(ttr) if i != -1)
    Xtr, Xva = onehot(ttr, ids), onehot(tva, ids)
    sel = select_H(Xtr, y["train"])
    m = score(Xtr[:, sel], Xva[:, sel])
    rows.append({"min_cluster_size": mcs, "n_topics": len(ids), "outlier_frac_train": float((ttr == -1).mean()),
                 "n_selected": len(sel), "val_metric": m, "sec": round(time.time() - t)})
    print(rows[-1], flush=True)
    if best is None or m > best[0]: best = (m, mcs, tm, [ids[i] for i in sel], ttr)
pd.DataFrame(rows).to_csv(f"{out}/grid.csv", index=False)
m, mcs, tm, topics, ttr = best
print("chosen min_cluster_size", mcs, "val", m, flush=True)

client = OpenAI(); rng = np.random.default_rng(0); usage = [0, 0]
texts = d["train"].text.values
PROMPT = """Below are documents from one cluster ("topic") of a text dataset, the topic's top keywords, and, for contrast, documents from other clusters.
Write ONE short natural-language feature that describes what the documents in this topic have in common and that distinguishes them from the other documents. The feature must be checkable on a single new document (e.g. "mentions ..." / "discusses ..." / "is about ...").

Topic keywords: {kw}

Documents in the topic:
{inside}

Documents from other topics:
{outside}

Answer with the feature only, as one sentence fragment starting with a verb (e.g. "discusses ...")."""
hyps = []
for tid in topics:
    kw = ", ".join(w for w, _ in tm.get_topic(tid)[:10])
    ins = rng.choice(np.where(ttr == tid)[0], min(10, (ttr == tid).sum()), replace=False)
    outs = rng.choice(np.where(ttr != tid)[0], 10, replace=False)
    fmt = lambda ix: "\n".join(f"- {' '.join(texts[i].split()[:200])}" for i in ix)
    r = client.chat.completions.create(model="gpt-4o-2024-11-20", temperature=0.0, max_tokens=80,
                                       messages=[{"role": "user", "content": PROMPT.format(kw=kw, inside=fmt(ins), outside=fmt(outs))}])
    usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens
    hyps.append({"topic": int(tid), "keywords": kw, "size": int((ttr == tid).sum()), "hypothesis": r.choices[0].message.content.strip().strip('"')})
cost = usage[0] * 2.5e-6 + usage[1] * 10e-6
json.dump({"min_cluster_size": mcs, "val_metric": m, "hypotheses": [h["hypothesis"] for h in hyps], "topics": hyps,
           "labeling_cost_usd": cost}, open(f"{out}/hypotheses.json", "w"), indent=1)
print(f"labeling cost ${cost:.3f}"); print("\n".join(f"- [{h['size']}] {h['hypothesis']}" for h in hyps))
