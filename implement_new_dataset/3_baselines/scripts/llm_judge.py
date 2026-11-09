"""LLM-as-a-judge for qualitative hypothesis evaluation .
gpt-4o-2024-11-20 rates every hypothesis of HypotheSAEs / BERTopic / HypoGeniC (DBpedia, 10-K) on 1-5 scales
modelled on the paper's human study (helpfulness, interpretability) plus specificity and checkability; a second
prompt flags near-duplicate pairs within each method's set (redundancy). Each rating is asked 3x at T=0.7 to measure
judge self-consistency. Also: does the judge's helpfulness relate to holdout predictiveness (HypotheSAEs, 10-K)?
Usage (repo root): OPENAI_KEY_SAE=... python implement_new_dataset/3_baselines/scripts/llm_judge.py   (~$0.6)
Output: implement_new_dataset/3_baselines/results/eval3_llm_judge/{raw,per_hyp,summary}.csv"""
import json, os, sys, itertools, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
client = OpenAI(api_key=os.environ["OPENAI_KEY_SAE"]); MODEL = "gpt-4o-2024-11-20"; lock = threading.Lock(); tok = [0, 0]
B = "implement_new_dataset/3_baselines/results"; OUT = "implement_new_dataset/3_baselines/results/eval3_llm_judge"; os.makedirs(OUT, exist_ok=True)
HYP = {"dbpedia": {"HypotheSAEs": pd.read_csv("implement_new_dataset/2_pipeline/results/dbpedia/holdout_hypotheses.csv").hypothesis.tolist(),
                   "BERTopic": json.load(open(f"{B}/3.2_bertopic/dbpedia/hypotheses.json"))["hypotheses"],
                   "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/dbpedia/hypotheses.json"))["hypotheses"]},
       "10k": {"HypotheSAEs": [h["interpretation"] for h in json.load(open("implement_new_dataset/2_pipeline/results/10k/pre_eval_hypotheses.json"))["hypotheses"]],
                "BERTopic": json.load(open(f"{B}/3.2_bertopic/10k/hypotheses.json"))["hypotheses"],
                "HypoGeniC": json.load(open(f"{B}/3.2_hypogenic/10k/hypotheses.json"))["hypotheses"]}}
CTX = {"dbpedia": "Task: discover which kinds of encyclopedia (Wikipedia/DBpedia) articles belong to a hidden target set of article categories.",
       "10k": "Task: a financial economist wants to discover which properties of the MD&A text of companies' 10-K reports are associated with higher or lower stock return volatility in the following year."}
RUBRIC = """You are reviewing a hypothesis produced by an automated hypothesis-generation system.
{ctx}
A hypothesis is a natural-language property that a single text either has or does not have; it is used to annotate texts and is then tested statistically against the outcome.

HYPOTHESIS: "{h}"

Rate it on four 1-5 scales (5 = best):
- helpfulness: would a domain researcher find this a useful, non-trivial idea to investigate for this task?
- interpretability: is it clear and unambiguous what the property means?
- specificity: is it a specific, concrete property (5) rather than vague or generic (1)?
- checkability: could an annotator reliably decide yes/no for one text from the text alone (5), or does it require outside knowledge / comparisons / the outcome itself (1)?
Answer with JSON only: {{"helpfulness": n, "interpretability": n, "specificity": n, "checkability": n}}"""
DUP = """Here are hypotheses (text properties) produced by one system for this task. {ctx}
List every pair of hypotheses that are near-duplicates, i.e. would mark largely the same texts. Use their numbers.
{lst}
Answer with JSON only: {{"duplicate_pairs": [[i, j], ...]}}"""


def call(prompt, T):
    r = client.chat.completions.create(model=MODEL, temperature=T, response_format={"type": "json_object"},
                                       messages=[{"role": "user", "content": prompt}])
    with lock: tok[0] += r.usage.prompt_tokens; tok[1] += r.usage.completion_tokens
    return json.loads(r.choices[0].message.content)


jobs = [(ds, m, i, h, rep) for ds in HYP for m in HYP[ds] for i, h in enumerate(HYP[ds][m]) for rep in range(3)]
with ThreadPoolExecutor(32) as ex:
    res = list(ex.map(lambda j: call(RUBRIC.format(ctx=CTX[j[0]], h=j[3]), 0.7), jobs))
R = pd.DataFrame([{"dataset": j[0], "method": j[1], "idx": j[2], "hypothesis": j[3], "rep": j[4], **r} for j, r in zip(jobs, res)])
R.to_csv(f"{OUT}/raw.csv", index=False)
crit = ["helpfulness", "interpretability", "specificity", "checkability"]
agg = R.groupby(["dataset", "method", "idx", "hypothesis"])[crit].mean().reset_index()
# self-consistency: share of hypotheses whose 3 ratings are identical / within 1 point
cons = R.groupby(["dataset", "method", "idx"])[crit].agg(lambda s: s.max() - s.min())
print("judge self-consistency (share of hypotheses with max-min <= 1 across 3 samples):")
print((cons <= 1).mean().round(3).to_dict(), "| identical:", (cons == 0).mean().round(3).to_dict())
dups = []
for ds in HYP:
    for m, hs in HYP[ds].items():
        lst = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hs))
        p = call(DUP.format(ctx=CTX[ds], lst=lst), 0.0).get("duplicate_pairs", [])
        involved = len({x for pr in p for x in pr})
        dups.append({"dataset": ds, "method": m, "n_hyp": len(hs), "dup_pairs": len(p), "hyps_in_dup_pairs": involved,
                     "redundancy_frac": involved / len(hs)})
summ = agg.groupby(["dataset", "method"])[crit].mean().round(2).reset_index().merge(pd.DataFrame(dups), on=["dataset", "method"])
summ["mean_words"] = [np.mean([len(h.split()) for h in HYP[r.dataset][r.method]]) for r in summ.itertuples()]
pd.set_option("display.width", 220); print(summ.round(2).to_string(index=False))
summ.to_csv(f"{OUT}/summary.csv", index=False); agg.to_csv(f"{OUT}/per_hyp.csv", index=False)
# judge vs measured predictiveness (HypotheSAEs 10-K holdout)
ev = pd.read_csv("implement_new_dataset/2_pipeline/results/10k/holdout_hypotheses.csv")[["hypothesis", "regression_pval"]]
a = agg[(agg.dataset == "10k") & (agg.method == "HypotheSAEs")].merge(ev, on="hypothesis")
for c in crit:
    r = spearmanr(a[c], -np.log10(a.regression_pval)); print(f"10-K HypotheSAEs: judge {c} vs holdout -log10 p: rho={r[0]:.2f} p={r[1]:.2f}")
print(f"cost ${tok[0]*2.5e-6 + tok[1]*10e-6:.2f}")
