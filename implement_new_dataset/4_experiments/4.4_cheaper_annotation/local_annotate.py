"""Annotate HypotheSAEs hypotheses on the DBpedia heldout set with a small local LLM served by vLLM,
then compare annotators (local models vs. the gpt-4o-mini annotations already in the cache).

The repo's annotation code is used unchanged (prompt `annotate.txt`, parsing, caching, thread pool);
only the transport is swapped: requests go to vLLM's OpenAI-compatible Chat Completions endpoint, with
Qwen3 "thinking" turned off via chat_template_kwargs so the model answers directly.

Usage:
  # 1) annotate with one local model (vLLM server already running on --base_url)
  python implement_new_dataset/4_experiments/4.4_cheaper_annotation/local_annotate.py annotate --model Qwen/Qwen3-0.6B --base_url http://127.0.0.1:8123/v1
  # 2) compare all annotators that have been run
  python implement_new_dataset/4_experiments/4.4_cheaper_annotation/local_annotate.py compare --models Qwen/Qwen3-0.6B meta-llama/Llama-3.2-1B-Instruct
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
import hypothesaes.annotate as ann_mod  # noqa: E402
from hypothesaes.annotate import annotate, CACHE_DIR  # noqa: E402
from hypothesaes.evaluation import score_hypotheses  # noqa: E402

DEFAULT_HYP = ("implement_new_dataset/2_pipeline/results/dbpedia/"
               "holdout_hypotheses.csv")
REF_CACHE = "dbpedia_text-embedding-3-small"  # gpt-4o-mini-2024-07-18, T=0 annotations from the paper-style run
REF_NAME = "gpt-4o-mini"

# per-dataset paths / task type (dbpedia = default, unchanged)
DATASETS = {
    "dbpedia": {"holdout": "data/dbpedia/processed/{size}/holdout.json", "hyp_csv": DEFAULT_HYP,
                "ref_cache": REF_CACHE, "prefix": "dbpedia", "label": "label", "classification": True,
                "out": "implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/dbpedia/local_llm"},
    "10k": {"holdout": "data/10k/holdout.json", "hyp_csv": "implement_new_dataset/2_pipeline/results/10k/holdout_hypotheses.csv",
               "ref_cache": "10k_text-embedding-3-small", "prefix": "10k", "label": "y", "classification": False,
               "out": "implement_new_dataset/4_experiments/4.4_cheaper_annotation/results/10k/local_llm"},
}


def cfg(args):
    c = DATASETS[args.dataset]
    return {k: (v.format(size=args.size) if isinstance(v, str) else v) for k, v in c.items()}


# repo prompt variants: (system prompt, user prompt, parser)
PROMPTS = {
    "annotate": (None, "annotate", "plain"),              # repo default few-shot prompt (used for gpt-4o-mini)
    "simple": (None, "annotate-simple", "plain"),         # "Yes"/"No" only
    "json": ("annotate-system-json", "annotate-user-json", "json"),
}


def model_tag(model, prompt="annotate"):
    tag = model.split("/")[-1]
    return tag if prompt == "annotate" else f"{tag}_{prompt}"


def parse_spec(spec):
    """'Qwen/Qwen3-0.6B:simple' -> ('Qwen/Qwen3-0.6B', 'simple')"""
    m, _, p = spec.partition(":")
    return m, (p or "annotate")


def load_inputs(args):
    c = cfg(args)
    hyp = pd.read_csv(os.path.join(ROOT, args.hyp_csv or c["hyp_csv"]))["hypothesis"].tolist()
    ho = pd.read_json(os.path.join(ROOT, c["holdout"]), lines=True)
    if args.n_hyp:
        hyp = hyp[: args.n_hyp]
    return hyp, ho


def to_arrays(results, hyp, texts):
    """concept -> 0/1 array; missing (parse failure) -> 0, counted."""
    arrs, missing = {}, 0
    for h in hyp:
        r = results.get(h, {})
        a = np.array([r.get(t, -9) for t in texts])
        missing += int((a == -9).sum())
        arrs[h] = np.where(a == -9, 0, a)
    return arrs, missing


def cmd_annotate(args):
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="local-no-auth")
    stats = {"calls": 0, "in": 0, "out": 0}

    def local_completion(prompt=None, *, system_prompt=None, model=None, temperature=None,
                         max_output_tokens=None, timeout=None, **_):
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": prompt}]
        r = client.chat.completions.create(
            model=model, messages=msgs, temperature=0.0 if temperature is None else temperature,
            max_tokens=args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        stats["calls"] += 1; stats["in"] += r.usage.prompt_tokens; stats["out"] += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    ann_mod.get_completion = local_completion  # annotate_single_text looks this name up at call time

    hyp, ho = load_inputs(args)
    texts = ho["text"].tolist()
    tag = model_tag(args.model, args.prompt)
    sys_name, user_name, parser = PROMPTS[args.prompt]
    c = cfg(args)
    cache_path = os.path.join(CACHE_DIR, f"{c['prefix']}_localannot_{tag}_hypothesis-eval.json")
    tasks = [(t, h) for t in texts for h in hyp]
    print(f"{tag}: {len(hyp)} hypotheses x {len(texts)} texts = {len(tasks)} annotations, {args.n_workers} workers")
    t0 = time.time()
    results = annotate(tasks=tasks, model=args.model, cache_path=cache_path, n_workers=args.n_workers,
                       max_words_per_example=256, temperature=0.0, progress_desc=f"Annotating ({tag})",
                       annotate_prompt_name=user_name, system_prompt_name=sys_name,
                       parse_fn=ann_mod.parse_completion_json if parser == "json" else ann_mod.parse_completion,
                       max_retries=1)  # T=0 is deterministic, retrying a parse failure gives the same output
    wall = time.time() - t0
    arrs, missing = to_arrays(results, hyp, texts)
    metrics, df = safe_score(arrs, ho[c["label"]].values, c["classification"])
    out = {"model": args.model, "prompt": args.prompt, "wall_min": wall / 60, "n_annotations": len(tasks), "parse_failures": missing,
           "calls": stats["calls"], "prompt_tokens": stats["in"], "completion_tokens": stats["out"],
           "throughput_per_s": stats["calls"] / wall if wall > 0 else None,
           "auroc": metrics.get("auroc"), "r2": metrics.get("r2"), "significant": metrics["Significant"][:2],
           "n_dropped_constant": metrics["n_dropped_constant"]}
    print(json.dumps(out, indent=2, default=str))
    od = os.path.join(ROOT, c["out"])
    os.makedirs(od, exist_ok=True)
    df.to_csv(os.path.join(od, f"eval_{tag}.csv"), index=False)
    json.dump(out, open(os.path.join(od, f"run_{tag}.json"), "w"), indent=2, default=str)


def safe_score(arrs, y, classification=True):
    """score_hypotheses, dropping hypotheses that are constant (all yes/no) on heldout,
    which would make the multivariate logit singular. Dropped ones get NaN separation."""
    keep = {h: a for h, a in arrs.items() if 0 < a.mean() < 1}
    metrics, df = score_hypotheses(keep, y, classification=classification, corrected_pval_threshold=0.05)  # paper: 0.05/H
    dropped = [h for h in arrs if h not in keep]
    if dropped:
        df = pd.concat([df, pd.DataFrame({"hypothesis": dropped})], ignore_index=True)
    metrics["n_dropped_constant"] = len(dropped)
    return metrics, df


def cohen_kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_compare(args):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    hyp, ho = load_inputs(args)
    c = cfg(args)
    texts, y = ho["text"].tolist(), ho[c["label"]].values

    ann = {}
    ref = annotate(tasks=[(t, h) for t in texts for h in hyp], model="gpt-4o-mini", show_progress=False,
                   cache_path=os.path.join(CACHE_DIR, f"{c['ref_cache']}_hypothesis-eval.json"), use_cache_only=True)
    ann[REF_NAME], _ = to_arrays(ref, hyp, texts)
    fails = {REF_NAME: 0}
    for spec in args.models:
        m, pr = parse_spec(spec)
        tag = model_tag(m, pr)
        res = annotate(tasks=[(t, h) for t in texts for h in hyp], model=m, show_progress=False, use_cache_only=True,
                       uncached_value=-9,
                       cache_path=os.path.join(CACHE_DIR, f"{c['prefix']}_localannot_{tag}_hypothesis-eval.json"))
        ann[tag], fails[tag] = to_arrays(res, hyp, texts)

    # per-annotator predictiveness
    rows, sep, sig = [], {}, {}
    for name, A in ann.items():
        metrics, df = safe_score(A, y, c["classification"])
        df = df.set_index("hypothesis").loc[hyp]
        sep[name] = df["separation_score"].values
        thr = metrics["Significant"][2]
        sig[name] = set(df.index[df["regression_pval"] < thr])
        X = np.stack([A[h] for h in hyp], 1)
        rows.append({"annotator": name, ("auroc" if c["classification"] else "r2"):
                     metrics.get("auroc") if c["classification"] else metrics.get("r2"),
                     "significant": f"{metrics['Significant'][0]}/{metrics['Significant'][1]}",
                     "dropped_constant": metrics["n_dropped_constant"], "mean_prevalence": X.mean(), "mean_abs_separation": np.nanmean(np.abs(sep[name])),
                     "n_hyp_sep>0.3": int(np.nansum(sep[name] > 0.3)), "missing_as_no": fails[name]})
    per = pd.DataFrame(rows)

    # pairwise agreement
    names = list(ann)
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            Xa = np.stack([ann[a][h] for h in hyp], 1); Xb = np.stack([ann[b][h] for h in hyp], 1)
            kappas = [cohen_kappa(Xa[:, k], Xb[:, k]) for k in range(len(hyp))]
            ok = ~np.isnan(sep[a]) & ~np.isnan(sep[b])
            pair_rows.append({"pair": f"{a} vs {b}", "agreement": (Xa == Xb).mean(),
                              "kappa_pooled": cohen_kappa(Xa.ravel(), Xb.ravel()),
                              "kappa_mean_per_hyp": np.nanmean(kappas),
                              "spearman_separation": spearmanr(sep[a][ok], sep[b][ok])[0],
                              "significant_overlap": f"{len(sig[a] & sig[b])} (of {len(sig[a])}/{len(sig[b])})"})
    pair = pd.DataFrame(pair_rows)

    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print("=== Predictiveness of heldout annotations, per annotator ===")
    print(per.round(3).to_string(index=False))
    print("\n=== Agreement between annotators ===")
    print(pair.round(3).to_string(index=False))
    od = os.path.join(ROOT, c["out"])
    os.makedirs(od, exist_ok=True)
    per.to_csv(os.path.join(od, "compare_per_annotator.csv"), index=False)
    pair.to_csv(os.path.join(od, "compare_pairwise.csv"), index=False)
    pd.DataFrame({"hypothesis": hyp, **{f"sep_{k}": v for k, v in sep.items()}}).to_csv(
        os.path.join(od, "compare_separation_by_hypothesis.csv"), index=False)
    print("saved to", od)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["annotate", "compare"]:
        p = sub.add_parser(name)
        p.add_argument("--size", default="12k")
        p.add_argument("--dataset", default="dbpedia", choices=list(DATASETS))
        p.add_argument("--hyp_csv", default=None, help="defaults to the dataset's hypothesis file")
        p.add_argument("--n_hyp", type=int, default=None, help="use only the first n hypotheses")
        if name == "annotate":
            p.add_argument("--model", required=True)
            p.add_argument("--base_url", default="http://127.0.0.1:8123/v1")
            p.add_argument("--n_workers", type=int, default=256)
            p.add_argument("--max_tokens", type=int, default=128)
            p.add_argument("--prompt", default="annotate", choices=list(PROMPTS))
        else:
            p.add_argument("--models", nargs="+", required=True,
                           help="model[:prompt], e.g. Qwen/Qwen3-0.6B:simple meta-llama/Llama-3.2-1B-Instruct:json")
    args = ap.parse_args()
    cmd_annotate(args) if args.cmd == "annotate" else cmd_compare(args)


if __name__ == "__main__":
    main()
