"""Control for batch_groups.py: the batch prompt with ONE hypothesis per call (g1). If g1 already says 'yes' much more often
than the repo's single-call prompt, the drop in quality comes from the prompt wording, not from batching."""
import sys, numpy as np, pandas as pd, os
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from batch_groups import annotate, HYP, downstream, kappa
import hypothesaes.annotate as A
from batch_all import MODEL
rows = []
for ds, hs in HYP.items():
    d = pd.read_json(f"implement_new_dataset/3_baselines/results/eval1_llm_labels_val100/{ds}_val100.json", lines=True); texts = d.text.tolist()
    ref = A.annotate(tasks=[(t, h) for t in texts for h in hs], model=MODEL, use_cache_only=True, show_progress=False,
                     cache_path=os.path.join(A.CACHE_DIR, "valsub100_hypothesis-eval.json"))
    S = np.array([[ref[h][t] for h in hs] for t in texts]); u = [0, 0, 0]
    with ThreadPoolExecutor(64) as ex:
        Bm = np.array(list(ex.map(lambda t: annotate(t, hs, 1, False, u), texts)))
    rows.append({"dataset": ds, "variant": "g1 (batch prompt, 1 per call)", "prevalence": Bm.mean(), "single_prevalence": S.mean(),
                 "kappa_pooled": kappa(S.ravel(), Bm.ravel()), "recall_single_yes": (Bm[S == 1] == 1).mean(),
                 "precision_vs_single": (S[Bm == 1] == 1).mean(), **downstream(Bm, d, ds, hs), "calls": u[2],
                 "cost_usd": u[0] * 0.15e-6 + u[1] * 0.6e-6})
    print(rows[-1], flush=True)
os.makedirs("implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/single_prompt_control", exist_ok=True); pd.DataFrame(rows).to_csv("implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/batching/single_prompt_control/summary.csv", index=False)
