"""Improved batched annotation. Same 100 texts / HypotheSAEs hypotheses / gpt-4o-mini as
batch_all.py, reference = single-call annotations. Variants:
  g5      : groups of 5 hypotheses per call, strict JSON schema, 0/1 only
  g10     : groups of 10 per call
  g10_ev  : groups of 10, each answer preceded by a one-sentence evidence string (schema: {"h1": {"evidence":..., "answer": 0|1}, ...})
  all_ev  : all K in one call, with evidence (DBpedia 50 / 10-K 20)"""
import json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from batch_all import client, MODEL, INSTR, HYP, downstream, kappa, SINGLE_COST
import hypothesaes.annotate as A
from hypothesaes.utils import truncate_text
lock = threading.Lock()


def schema(K, ev):
    item = ({"type": "object", "properties": {"evidence": {"type": "string"}, "answer": {"type": "integer", "enum": [0, 1]}},
             "required": ["evidence", "answer"], "additionalProperties": False} if ev else {"type": "integer", "enum": [0, 1]})
    return {"type": "json_schema", "json_schema": {"name": "a", "strict": True, "schema": {"type": "object",
            "properties": {f"h{i}": item for i in range(1, K + 1)}, "required": [f"h{i}" for i in range(1, K + 1)],
            "additionalProperties": False}}}


def ask(text, hs, ev, u):
    K = len(hs); props = "\n".join(f"{i}. {h}" for i, h in enumerate(hs, 1))
    fmt = (f"Return a JSON object with keys h1 ... h{K}; for each, first give 'evidence' (one short sentence quoting or citing the TEXT, "
           "or saying it is absent), then 'answer' (1 if the TEXT satisfies the property, else 0)." if ev else
           f"Return a JSON object with exactly the keys h1 ... h{K}; each value is 1 if the TEXT satisfies that property, else 0.")
    r = client.chat.completions.create(model=MODEL, temperature=0, response_format=schema(K, ev),
                                       messages=[{"role": "user", "content": INSTR.format(K=K, props=props, text=truncate_text(text, 256)) + "\n" + fmt}])
    with lock: u[0] += r.usage.prompt_tokens; u[1] += r.usage.completion_tokens; u[2] += 1
    j = json.loads(r.choices[0].message.content)
    return [(j[f"h{i}"]["answer"] if ev else j[f"h{i}"]) for i in range(1, K + 1)]


def annotate(text, hs, g, ev, u):
    out = []
    for s in range(0, len(hs), g): out += ask(text, hs[s:s + g], ev, u)
    return out


if __name__ == "__main__":
    rows = []
    for ds, hs in HYP.items():
        d = pd.read_json(f"implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/{ds}_val100.json", lines=True); texts = d.text.tolist(); K = len(hs)
        ref = A.annotate(tasks=[(t, h) for t in texts for h in hs], model=MODEL, use_cache_only=True, show_progress=False,
                         cache_path=os.path.join(A.CACHE_DIR, "valsub100_hypothesis-eval.json"))
        S = np.array([[ref[h][t] for h in hs] for t in texts])
        rows.append({"dataset": ds, "variant": "single (reference)", "prevalence": S.mean(), **downstream(S, d, ds, hs),
                     "calls": len(texts) * K, "cost_usd": len(texts) * K * SINGLE_COST[ds]})
        for name, g, ev in [("g5", 5, False), ("g10", 10, False), ("g10_ev", 10, True), ("all_ev", K, True)]:
            u = [0, 0, 0]
            with ThreadPoolExecutor(32) as ex:
                Bm = np.array(list(ex.map(lambda t: annotate(t, hs, g, ev, u), texts)))
            rows.append({"dataset": ds, "variant": name, "prevalence": Bm.mean(), "agreement": (S == Bm).mean(),
                         "kappa_pooled": kappa(S.ravel(), Bm.ravel()), "recall_single_yes": (Bm[S == 1] == 1).mean(),
                         "precision_vs_single": (S[Bm == 1] == 1).mean() if Bm.sum() else np.nan,
                         **downstream(Bm, d, ds, hs), "calls": u[2], "cost_usd": u[0] * 0.15e-6 + u[1] * 0.6e-6})
            print(rows[-1], flush=True)
    prev = pd.read_csv("implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/batch_all/summary.csv")
    res = pd.concat([pd.DataFrame(rows), prev[prev.variant.str.startswith("batched")]], ignore_index=True)
    os.makedirs("implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/batch_groups", exist_ok=True); res.to_csv("implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/batch_groups/summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["dataset", "variant", "prevalence", "kappa_pooled", "recall_single_yes", "precision_vs_single", "recovery_F1",
               "classes_F1>=0.5", "adj_R2", "calls", "cost_usd"]].sort_values(["dataset", "cost_usd"]).round(3).to_string(index=False))
