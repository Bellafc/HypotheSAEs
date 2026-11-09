"""Test: annotate ALL K hypotheses of a text in ONE call (K binary outputs) vs. the repo's one-call-per-(text, hypothesis)
annotation. Same 100 validation texts per dataset as implement_new_dataset/3_baselines/results/eval1_llm_labels_val100 (HypotheSAEs hypotheses: DBpedia K=50, 10-K K=20),
same model (gpt-4o-mini-2024-07-18, T=0) and 256-word truncation. Reference = cached single-call annotations (no new cost).
Variants:
  json   : structured outputs (strict JSON schema with keys h1..hK, each 0 or 1) -> format guaranteed by the API
  lines  : plain text, instructed to output exactly K lines "i: 0" / "i: 1" -> tests format compliance without schema
Output: implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/batch_all/. Usage (repo root): OPENAI_KEY_SAE=... python implement_new_dataset/4_experiments/4.4_cheaper_annotation/batch_all.py"""
import json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from openai import OpenAI
from scipy.optimize import linear_sum_assignment
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import hypothesaes.annotate as A
from hypothesaes.utils import truncate_text

MODEL = "gpt-4o-mini-2024-07-18"; OUT = "implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/batch_all"; os.makedirs(OUT, exist_ok=True)
client = OpenAI(api_key=os.environ["OPENAI_KEY_SAE"]); lock = threading.Lock(); usage = {}
HYP = {"dbpedia": pd.read_csv("implement_new_dataset/2_pipeline/results/dbpedia/holdout_hypotheses.csv").hypothesis.tolist(),
       "10k": [h["interpretation"] for h in json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))["hypotheses"]]}
LAB = {"dbpedia": ("label", True), "10k": ("y", False)}
SINGLE_COST = {"dbpedia": 11.266974 / 100000, "10k": 4.4532498 / 32020}  # measured $/annotation of the single-call runs

INSTR = """Check whether the TEXT satisfies each of the {K} PROPERTIES listed below. Judge every property independently, based only on evidence in the TEXT. When uncertain, answer 0 (No). A property whose meaning is unclear or unrelated to the text gets 0.

Examples of how to judge a single property:
- PROPERTY "mentions a natural scene." / TEXT "I love the way the sun sets in the evening." -> 1 ("sun sets" is a natural scene)
- PROPERTY "mentions that the breakfast is good on the airline." / TEXT "The airline staff was really nice! Enjoyable flight." -> 0 (breakfast is not mentioned)
- PROPERTY "appreciates the writing style of the author." / TEXT "The paper absolutely sucks ... However, the presentation of the paper is clear and the use of language is really impressive." -> 1
- PROPERTY "refers to historical dates." / TEXT "A member of the Democratic Party, he was the first African-American president of the United States." -> 0

PROPERTIES:
{props}

TEXT: "{text}"
"""
FMT = {"json": "Return a JSON object with exactly the keys h1 ... h{K}; each value is 1 if the TEXT satisfies that property, else 0.",
       "lines": "Output EXACTLY {K} lines and nothing else. Line i must be \"i: 1\" if the TEXT satisfies property i, or \"i: 0\" otherwise, for i = 1 ... {K}, in order. No explanations, no extra text."}


def schema(K):
    return {"type": "json_schema", "json_schema": {"name": "annotations", "strict": True, "schema": {
        "type": "object", "properties": {f"h{i}": {"type": "integer", "enum": [0, 1]} for i in range(1, K + 1)},
        "required": [f"h{i}" for i in range(1, K + 1)], "additionalProperties": False}}}


def ask(text, hs, variant):
    K = len(hs); props = "\n".join(f"{i}. {h}" for i, h in enumerate(hs, 1))
    prompt = INSTR.format(K=K, props=props, text=truncate_text(text, 256)) + "\n" + FMT[variant].format(K=K)
    kw = {"response_format": schema(K)} if variant == "json" else {}
    r = client.chat.completions.create(model=MODEL, temperature=0, messages=[{"role": "user", "content": prompt}], **kw)
    with lock:
        u = usage.setdefault(variant, [0, 0, 0]); u[0] += r.usage.prompt_tokens; u[1] += r.usage.completion_tokens; u[2] += 1
    out = r.choices[0].message.content
    if variant == "json":
        try:
            j = json.loads(out); v = [j[f"h{i}"] for i in range(1, K + 1)]
            return v if all(x in (0, 1) for x in v) else None, out
        except Exception:
            return None, out
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    ok = len(lines) == K and all(re.fullmatch(rf"{i}:\s*[01]", l) for i, l in enumerate(lines, 1))
    return ([int(l[-1]) for l in lines] if ok else None), out


def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def downstream(M, d, ds, hs):
    lab, cls = LAB[ds]; y = d[lab].values.astype(float)
    keep = [j for j in range(M.shape[1]) if 0 < M[:, j].mean() < 1]; X = sm.add_constant(M[:, keep].astype(float))
    res = {}
    if cls:
        r = sm.Logit(y, X).fit(disp=0, method="lbfgs", maxiter=2000); res["AUC"] = roc_auc_score(y, r.predict(X))
        classes = sorted(d.loc[d.label == 1, "l3"].unique())
        R = np.stack([(d.l3 == c).values.astype(float) for c in classes], 1)
        C = np.nan_to_num(np.corrcoef(np.concatenate([R, M], 1).T)[:len(classes), len(classes):])
        ri, ci = linear_sum_assignment(-C)
        f1 = [2 * (R[:, i] * M[:, j]).sum() / (M[:, j].sum() + R[:, i].sum()) if M[:, j].sum() + R[:, i].sum() else 0 for i, j in zip(ri, ci)]
        res["recovery_F1"] = float(np.mean(f1)); res["classes_F1>=0.5"] = int((np.array(f1) >= 0.5).sum())
    else:
        r = sm.OLS(y, X).fit(); res["R2"] = r.rsquared; res["adj_R2"] = r.rsquared_adj
    return res


if __name__ == "__main__":
    rows, cmp = [], []
    for ds, hs in HYP.items():
        d = pd.read_json(f"implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/{ds}_val100.json", lines=True); texts = d.text.tolist(); K = len(hs)
        ref = A.annotate(tasks=[(t, h) for t in texts for h in hs], model=MODEL, use_cache_only=True, show_progress=False,
                         cache_path=os.path.join(A.CACHE_DIR, "valsub100_hypothesis-eval.json"))
        S = np.array([[ref[h][t] for h in hs] for t in texts])
        rows.append({"dataset": ds, "variant": "single (reference)", "parse_ok": 1.0, "prevalence": S.mean(), **downstream(S, d, ds, hs),
                     "calls": len(texts) * K, "cost_usd": len(texts) * K * SINGLE_COST[ds]})
        for v in ["json", "lines"]:
            usage[v] = [0, 0, 0]
            with ThreadPoolExecutor(32) as ex:
                outs = list(ex.map(lambda t: ask(t, hs, v), texts))
            ok = np.array([o[0] is not None for o in outs])
            json.dump([{"text_idx": i, "raw": o[1]} for i, o in enumerate(outs)], open(f"{OUT}/{ds}_{v}_raw.json", "w"))
            Bm = np.array([o[0] if o[0] is not None else [0] * K for o in outs])
            Sk, Bk = S[ok], Bm[ok]
            per_h = [kappa(Sk[:, j], Bk[:, j]) for j in range(K)]
            pos = np.array_split(np.arange(K), 4)
            cost = usage[v][0] * 0.15e-6 + usage[v][1] * 0.6e-6
            rows.append({"dataset": ds, "variant": f"batched-{v}", "parse_ok": ok.mean(), "prevalence": Bk.mean(),
                         "agreement_vs_single": (Sk == Bk).mean(), "kappa_pooled": kappa(Sk.ravel(), Bk.ravel()),
                         "kappa_mean_per_hyp": np.nanmean(per_h),
                         "recall_of_single_yes": (Bk[Sk == 1] == 1).mean(), "precision_vs_single_yes": (Sk[Bk == 1] == 1).mean() if Bk.sum() else np.nan,
                         "agreement_by_position_quartile": [round(float((Sk[:, p] == Bk[:, p]).mean()), 3) for p in pos],
                         **downstream(Bm, d, ds, hs), "calls": usage[v][2], "cost_usd": cost})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30); print(res.round(3).to_string(index=False))
