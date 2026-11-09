"""HypotheSAEs pipeline for any dataset, driven by a YAML config.

The code defaults below (PAPER_DEFAULTS) are the paper's settings; configs/sample.yaml (next to this file) lists every
option with these default values. Each dataset has one config file in configs/ (configs/dbpedia.yaml, configs/10k.yaml) that
holds its data paths and every setting we changed from the paper, with a comment on why. `--set key.sub=value` overrides any config value for a one-off run (written to the log).

Stages (run in this order; each one is resumable and saves its outputs in output_dir):
  grid       train one SAE per (M, K) in sae.grid, LASSO-select H neurons, OLS/logistic on them, pick (M, K) by the
             validation metric -> grid_MK.csv, chosen_MK.json. Not needed if the grid has one entry.   [no LLM]
             Also a reference: ridge (regression) / logistic (classification) on all embedding dimensions, saved
             in chosen_MK.json.
             With a local embedder and its own output folder it is the free signal check of a new embedding:
               --set embedding.model=Qwen/Qwen3-Embedding-4B --set embedding.backend=local \
               --set output_dir=implement_new_dataset/4_experiments/4.1_embedding/results/10k/{embedding_tag}
  select     embeddings, baselines (raw embeddings / all SAE neurons / selected neurons), SAE, neuron selection
             -> selected_neurons_<tag>.csv, metrics_<tag>.json                                             [no LLM]
  llm        interpret the selected neurons (n candidates each) and score fidelity; keep the best candidate
             -> pre_eval_hypotheses_<run>.json, <run> = timestamp YYYYMMDD-HHMMSS of this run, so a new run
             never overwrites an earlier one (at T=0.7 every run gives different hypotheses)
  holdout    annotate the holdout and evaluate (cached annotations are reused). Every annotation is appended to a JSONL
             log as soon as it returns, so an interruption loses nothing. Then runs the analyses in after_holdout.
             -> holdout_hypotheses_<run>.csv, holdout_metrics_<run>.json, usage_holdout_<run>.csv
  recovery   ground-truth recovery for synthetic tasks (paper section 5.2): Hungarian matching of hypotheses to the
             positive classes, F1 and LLM surface similarity -> recovery_{summary.json,matches.csv,corrmatrix.csv}
The stages after llm work on one set of hypotheses, chosen with --run <run> (default: the newest). Their outputs carry
the same <run> suffix. Results made before 2026-11-09 have no suffix (run ""), e.g. pre_eval_hypotheses.json.

Usage (from the repository root; --config takes a dataset name or a path to a .yaml file):
  python implement_new_dataset/2_pipeline/run_pipeline.py --config dbpedia --stage select
  OPENAI_KEY_SAE=... python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage llm
Paths in the configs (data, caches, checkpoints, outputs) are relative to the repository root.

Other scripts use the helpers: cfg = load_config("10k"); load_data(cfg); embed(cfg, texts); get_sae(cfg, E, M, K);
chosen_MK(cfg); r2(y, p); UsageTracker.
"""
import argparse
import copy
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))                   # implement_new_dataset/2_pipeline/
ROOT = os.path.dirname(os.path.dirname(HERE))                        # repository root
sys.path.insert(0, ROOT)
from hypothesaes.quickstart import train_sae  # noqa: E402
from hypothesaes.select_neurons import select_neurons  # noqa: E402

# --------------------------------------------------------------------------------------------- paper settings
PAPER_DEFAULTS = {
    "vars": {},                                   # template variables for paths, e.g. {size} (set in configs)
    "data": {"text_col": "text", "group_col": None},
    "embedding": {"model": "text-embedding-3-small", "backend": "openai", "batch_size": 64},
    "sae": {"grid": None,                         # list of [M, K]; chosen by validation performance (paper section 3)
            "n_epochs": 100, "patience": 3},      # library training defaults
    "selection": {"method": "lasso", "n_select": 20},
    "interpretation": {"model": "gpt-4o", "temperature": 0.7, "n_candidates": 3, "n_examples": 20,
                       "max_words": 128,          # max_words_per_example in the paper's reproduction notebook
                       "high_pct": [90, 100],     # App. B.2: highly activating = top decile of positive activations
                       "low_pct": [0, 10],        # weakly activating "hard negatives" = bottom decile; "zero" = 0-acts
                       "task_instructions": None,  # one per dataset (the paper's notebook writes one per dataset)
                       "n_workers": 10},
    # step 5, fidelity: the LLM says yes/no for 200 train texts (100 high-bin + 100 low-bin) per candidate
    "scoring": {"model": "gpt-4o-mini", "temperature": 0.0, "reasoning_effort": None, "max_words": 256,
                "n_examples": 200, "n_workers": 120},
    # step 6, holdout: the LLM says yes/no for every holdout text x hypothesis
    "annotation": {"model": "gpt-4o-mini", "temperature": 0.0, "reasoning_effort": None, "max_words": 256,
                   "n_workers": 120},
    "evaluation": {"pval": 0.05},                 # Bonferroni 0.05/H (paper footnote 4; the library default is 0.1/H)
    "select_report_split": "val",                 # split for the no-LLM baselines of the select stage
    "diagnostic_col": None,                       # optional label column summarized per neuron / hypothesis
    "after_holdout": [],                          # analyses run right after the holdout stage
    "recovery": None,     # see configs/sample.yaml for its keys
}
PRICES = {  # $ per 1M tokens: (input, cached input, output). Check https://openai.com/api/pricing if these change.
    "gpt-5.2": (1.75, 0.175, 14.0), "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-4o": (2.50, 1.25, 10.0), "gpt-4o-2024-11-20": (2.50, 1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.075, 0.60), "gpt-4o-mini-2024-07-18": (0.15, 0.075, 0.60),
    "text-embedding-3-small": (0.02, 0.02, 0.0)}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path_or_name, overrides=()):
    """config = PAPER_DEFAULTS <- configs/<name>.yaml <- --set overrides; path templates filled from vars."""
    path = path_or_name if path_or_name.endswith(".yaml") else os.path.join(HERE, "configs", f"{path_or_name}.yaml")
    cfg = _merge(PAPER_DEFAULTS, yaml.safe_load(open(path)))
    for o in overrides:
        key, val = o.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    v = dict(cfg["vars"], name=cfg["name"], embedding_tag=cfg["embedding"]["model"].split("/")[-1])
    for k in ["cache_name", "checkpoint_dir", "output_dir"]:
        cfg[k] = cfg[k].format(**v)
    cfg["data"]["dir"] = cfg["data"]["dir"].format(**v)
    for k in ["checkpoint_dir", "output_dir"]:
        cfg[k] = os.path.join(ROOT, cfg[k])
    cfg["data"]["dir"] = os.path.join(ROOT, cfg["data"]["dir"])
    cfg["classification"] = cfg["data"]["task"] == "classification"
    return cfg


# --------------------------------------------------------------------------------------------- shared helpers
def load_data(cfg):
    """dfs, texts, labels for train / val / holdout."""
    d = cfg["data"]
    dfs = {s: pd.read_json(os.path.join(d["dir"], f"{s}.json"), lines=True, dtype={"cik": str})
           for s in ["train", "val", "holdout"]}
    text_col = d["text_col"]
    y = {s: df[d["label_col"]].values.astype(int if cfg["classification"] else float) for s, df in dfs.items()}
    return dfs, {s: df[text_col].tolist() for s, df in dfs.items()}, y


def embed(cfg, texts, tracker=None):
    from hypothesaes.embedding import get_local_embeddings, get_openai_embeddings
    e = cfg["embedding"]
    if tracker is not None:
        tracker.phase = "embedding"
    allt = texts["train"] + texts["val"] + texts["holdout"]
    if e["backend"] == "openai":
        t2e = get_openai_embeddings(allt, model=e["model"], cache_name=cfg["cache_name"])
    else:
        t2e = get_local_embeddings(allt, model=e["model"], batch_size=e["batch_size"], cache_name=cfg["cache_name"])
    return {s: np.stack([t2e[t] for t in texts[s]]).astype(np.float32) for s in texts}


def get_sae(cfg, E, M, K):
    return train_sae(embeddings=E["train"], val_embeddings=E["val"], M=M, K=K, checkpoint_dir=cfg["checkpoint_dir"],
                     n_epochs=cfg["sae"]["n_epochs"], patience=cfg["sae"]["patience"], show_progress=False)


def chosen_MK(cfg):
    """(M, K): the only grid entry, or the one picked by the grid stage (chosen_MK.json)."""
    grid = cfg["sae"]["grid"]
    if not grid:
        raise SystemExit("the config needs sae.grid, a list of [M, K]")
    if len(grid) == 1:
        return tuple(grid[0])
    c = json.load(open(os.path.join(cfg["output_dir"], "chosen_MK.json")))
    return c["M"], c["K"]


def r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def select(cfg, A_train, y_train, n=None, verbose=False):
    s = cfg["selection"]
    kw = {"classification": cfg["classification"], "verbose": verbose} if s["method"] == "lasso" else {}
    sel, scores = select_neurons(activations=A_train, target=y_train, n_select=n or s["n_select"], method=s["method"],
                                 **kw)
    return [int(i) for i in sel], [float(c) for c in scores]


def score_col(cfg):
    m = cfg["selection"]["method"]
    return "lasso_coef" if m == "lasso" else f"{m}_score"


def fit_metric(cfg, Xtr, ytr, Xte, yte):
    """out-of-sample metric of a plain model on features: logistic AUC (classification) or OLS R^2 (regression)."""
    if cfg["classification"]:
        return roc_auc_score(yte, LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return r2(yte, LinearRegression().fit(Xtr, ytr).predict(Xte))


def annot_kwargs(cfg, section="annotation"):
    """LLM kwargs of the yes/no annotator of a config section ("annotation" = holdout, "scoring" = fidelity)."""
    a, kw = cfg[section], {}
    if a["temperature"] is not None:
        kw["temperature"] = a["temperature"]
    if a["reasoning_effort"] and a["model"].startswith("gpt-5"):
        kw["reasoning_effort"] = a["reasoning_effort"]
    return kw


CACHE_MODELS = os.path.join(ROOT, "annotation_cache", "cache_models.json")


def check_cache_model(cache_file, model):
    """The library keys cached annotations by (hypothesis, text) only, not by model: with another model and the same
    cache name it would silently reuse the old model's answers. annotation_cache/cache_models.json records which
    model filled each cache file; a different model is refused (use another cache_name for it)."""
    reg = json.load(open(CACHE_MODELS)) if os.path.exists(CACHE_MODELS) else {}
    if reg.get(cache_file, model) != model:
        raise SystemExit(f"annotation cache {cache_file} was filled by {reg[cache_file]}, not {model}: answers would be "
                         f"reused from the wrong model. Set another cache_name for this run (e.g. --set cache_name=...)")
    if cache_file not in reg:
        reg[cache_file] = model
        json.dump(reg, open(CACHE_MODELS, "w"), indent=1, sort_keys=True)


def pct_kwargs(cfg):
    i = cfg["interpretation"]
    low = None if i["low_pct"] == "zero" else tuple(float(x) for x in i["low_pct"])
    return {"high_percentile": tuple(float(x) for x in i["high_pct"]), "low_percentile": low}


class UsageTracker:
    """Wraps the shared OpenAI client to count tokens and cost per (phase, model)."""

    def __init__(self):
        from hypothesaes.llm_api import get_client
        self.lock, self.phase, self.stats = threading.Lock(), "init", {}
        client = get_client()  # get_completion() reuses this cached client
        orig, orig_emb = client.responses.create, client.embeddings.create

        def add(model, **counts):
            with self.lock:
                s = self.stats.setdefault((self.phase, model),
                                          {"calls": 0, "input": 0, "cached": 0, "output": 0, "reasoning": 0})
                s["calls"] += 1
                for k, v in counts.items():
                    s[k] += v

        def wrapped(*a, **k):
            r = orig(*a, **k)
            u = r.usage
            add(k.get("model"), input=u.input_tokens, output=u.output_tokens,
                cached=getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0,
                reasoning=getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0)
            return r

        def wrapped_emb(*a, **k):
            r = orig_emb(*a, **k)
            add(k.get("model"), input=r.usage.total_tokens)
            return r

        client.responses.create, client.embeddings.create = wrapped, wrapped_emb

    def report(self):
        rows = []
        for (phase, model), s in self.stats.items():
            p_in, p_cached, p_out = PRICES.get(model, (np.nan, np.nan, np.nan))
            cost = ((s["input"] - s["cached"]) * p_in + s["cached"] * p_cached + s["output"] * p_out) / 1e6
            rows.append({"phase": phase, "model": model, **s, "cost_usd": cost})
        df = pd.DataFrame(rows)
        if len(df):
            total = df.drop(columns=["phase", "model"]).sum()
            df = pd.concat([df, pd.DataFrame([{"phase": "TOTAL", "model": "", **total.to_dict()}])], ignore_index=True)
        return df


def durable_annotation_log(cache_name):
    """Make every finished annotation durable: append it to <cache>_hypothesis-eval.log.jsonl at once, and merge an
    earlier interrupted run's log into the JSON cache first (the library only writes its cache at the end)."""
    import hypothesaes.annotate as A
    cpath = os.path.join(A.CACHE_DIR, f"{cache_name}_hypothesis-eval.json")
    lpath = cpath.replace(".json", ".log.jsonl")
    if os.path.exists(lpath):
        cache = A.get_annotation_cache(cpath)
        n0 = len(cache)
        for line in open(lpath):
            try:
                r = json.loads(line)
                cache[r["k"]] = r["v"]
            except Exception:
                pass
        A.save_annotation_cache(cpath, cache)
        print(f"merged annotation log: {n0} -> {len(cache)} cached", flush=True)
    lock, logf = threading.Lock(), open(lpath, "a")
    orig = A._store_annotation

    def store_and_log(results, concept, text, annotation, cache=None):
        orig(results, concept, text, annotation, cache)
        if cache is not None and annotation is not None:
            with lock:
                logf.write(json.dumps({"k": A.generate_cache_key(concept, text), "v": annotation}) + "\n")
                logf.flush()
    A._store_annotation = store_and_log


def fpath(cfg, base, ext):
    """output file of the current hypothesis run: <base>_<run><ext>, or <base><ext> for the unsuffixed early run."""
    return os.path.join(cfg["output_dir"], f"{base}_{cfg['run']}{ext}" if cfg["run"] else f"{base}{ext}")


def list_runs(out):
    """hypothesis runs in an output folder, oldest first ("" = files from before timestamps were added)."""
    import glob
    runs = [os.path.basename(f)[len("pre_eval_hypotheses"):-len(".json")].lstrip("_")
            for f in glob.glob(os.path.join(out, "pre_eval_hypotheses*.json"))]
    return sorted(runs)  # "" sorts first, timestamps sort chronologically


def resolve_run(cfg, run=None):
    runs = list_runs(cfg["output_dir"])
    if run is not None:
        if run not in runs:
            raise SystemExit(f"no pre_eval_hypotheses for run '{run}' in {cfg['output_dir']}; runs: {runs}")
        return run
    if not runs:
        raise SystemExit(f"no pre_eval_hypotheses*.json in {cfg['output_dir']}; run the llm stage first")
    return runs[-1]


def top_values(col_values, mask, k=3):
    return "; ".join(f"{a}({b})" for a, b in pd.Series(col_values[mask]).value_counts().head(k).items())


# --------------------------------------------------------------------------------------------- stages
def stage_grid(cfg, dfs, E, y):
    out = cfg["output_dir"]
    split_key = cfg["data"]["group_col"]
    if cfg["classification"]:
        emb = fit_metric(cfg, E["train"], y["train"], E["val"], y["val"])
        print(f"[baseline] logistic on raw embeddings, val AUC = {emb:.4f}")
    else:  # ridge: plain OLS on 1,000+ embedding dimensions overfits
        p = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(E["train"], y["train"]).predict(E["val"])
        emb = r2(y["val"], p)
        print(f"[baseline] ridge on raw embeddings, val R2 = {emb:.4f}")
    rows = []
    for M, K in cfg["sae"]["grid"]:
        t = time.time()
        sae = get_sae(cfg, E, M, K)
        A = {s: sae.get_activations(E[s]) for s in ["train", "val"]}
        sel, coefs = select(cfg, A["train"], y["train"])
        row = {"M": M, "K": K}
        if cfg["classification"]:
            row["val_auc_H"] = fit_metric(cfg, A["train"][:, sel], y["train"], A["val"][:, sel], y["val"])
        else:
            m = LinearRegression().fit(A["train"][:, sel], y["train"])
            pv = m.predict(A["val"][:, sel])
            row.update({"train_r2_H": r2(y["train"], m.predict(A["train"][:, sel])), "val_r2_H": r2(y["val"], pv)})
            if split_key:  # chunk predictions averaged per report
                g = pd.DataFrame({"k": dfs["val"][split_key].values, "y": y["val"], "p": pv}).groupby("k").mean()
                row["val_r2_H_report"] = r2(g.y.values, g.p.values)
        row.update({"n_nonzero_coef": int(np.sum(np.array(coefs) != 0)), "dead": int((A["train"].max(0) == 0).sum()),
                    "sec": round(time.time() - t, 1)})
        rows.append(row)
        print(row, flush=True)
    crit = "val_auc_H" if cfg["classification"] else "val_r2_H"
    g = pd.DataFrame(rows).sort_values(crit, ascending=False)
    g.to_csv(os.path.join(out, "grid_MK.csv"), index=False)
    best = g.iloc[0]
    json.dump({"M": int(best.M), "K": int(best.K), crit: float(best[crit]),
               ("emb_logistic_val_auc" if cfg["classification"] else "emb_ridge_val_r2"): float(emb)},
              open(os.path.join(out, "chosen_MK.json"), "w"), indent=2)
    print(g.round(4).to_string(index=False))
    print("chosen:", int(best.M), int(best.K))


def stage_select(cfg, dfs, E, y):
    out, rs = cfg["output_dir"], cfg["select_report_split"]
    M, K = chosen_MK(cfg)
    met = "auc" if cfg["classification"] else "r2"
    res = {"emb": fit_metric(cfg, E["train"], y["train"], E[rs], y[rs])}
    sae = get_sae(cfg, E, M, K)
    A = {s: sae.get_activations(E[s]) for s in E}
    dead = int((A["train"].max(axis=0) == 0).sum())
    res["sae_all"] = fit_metric(cfg, A["train"], y["train"], A[rs], y[rs])
    sel, scores = select(cfg, A["train"], y["train"], verbose=True)
    res["sae_selected"] = fit_metric(cfg, A["train"][:, sel], y["train"], A[rs][:, sel], y[rs])
    for k, v in res.items():
        print(f"[{rs}] {k}: {met} = {v:.4f}")
    method, n = cfg["selection"]["method"], len(sel)
    rows = []
    for idx, sc in zip(sel, scores):
        act = A["train"][:, idx]
        top = np.argsort(-act)[:100]
        top = top[act[top] > 0]
        r = {"neuron_idx": idx, f"target_{method}": sc, "prevalence": float((act > 0).mean())}
        if cfg["classification"]:
            r["top100_pos_rate"] = float(y["train"][top].mean()) if len(top) else np.nan
        else:
            r["top100_mean_y"] = float(y["train"][top].mean()) if len(top) else np.nan
        if cfg["diagnostic_col"]:
            c = cfg["diagnostic_col"]
            r[f"top100_{c}"] = top_values(dfs["train"][c].values, top)
        rows.append(r)
    sel_df = pd.DataFrame(rows).sort_values(f"target_{method}", ascending=False)
    pd.set_option("display.max_colwidth", 80, "display.width", 200)
    print(sel_df.round(3).to_string(index=False))
    tag = f"M{M}_K{K}_{method}_n{n}"
    sel_df.to_csv(os.path.join(out, f"selected_neurons_{tag}.csv"), index=False)
    json.dump({"embedder": cfg["embedding"]["model"], **cfg["vars"], "M": M, "K": K,
               **{f"{k}_{met}": v for k, v in res.items()},
               "report_split": rs, "selection_method": method, "n_select": n, "dead_neurons": dead,
               "selected_neurons": sel}, open(os.path.join(out, f"metrics_{tag}.json"), "w"), indent=2)
    print("saved to", out)


def stage_llm(cfg, dfs, texts, E, y, tracker):
    from hypothesaes.interpret_neurons import (NeuronInterpreter, InterpretConfig, ScoringConfig, SamplingConfig,
                                               LLMConfig, sample_percentile_bins)
    cfg["run"] = time.strftime("%Y%m%d-%H%M%S")
    ckpt = fpath(cfg, "pre_eval_hypotheses", ".json")
    print("hypothesis run:", cfg["run"], "->", ckpt)
    M, K = chosen_MK(cfg)
    sae = get_sae(cfg, E, M, K)
    A = {s: sae.get_activations(E[s]) for s in ["train", "val"]}
    sel, coefs = select(cfg, A["train"], y["train"])
    assert len(sel) == cfg["selection"]["n_select"] and all(c != 0 for c in coefs), f"zero coefficients: {coefs}"
    i, pct = cfg["interpretation"], pct_kwargs(cfg)
    sc = cfg["scoring"]
    check_cache_model(f"{cfg['cache_name']}_interp-scoring.json", sc["model"])
    interp = NeuronInterpreter(interpreter_model=i["model"], annotator_model=sc["model"],
                               n_workers_interpretation=i["n_workers"], n_workers_annotation=sc["n_workers"],
                               cache_name=cfg["cache_name"])
    tracker.phase = "interpretation"
    t0 = time.time()
    extra = {"task_specific_instructions": i["task_instructions"]} if i["task_instructions"] else {}  # else library's
    icfg = InterpretConfig(sampling=SamplingConfig(function=sample_percentile_bins, n_examples=i["n_examples"],
                                                   max_words_per_example=i["max_words"], sampling_kwargs=pct),
                           llm=LLMConfig(temperature=i["temperature"]), n_candidates=i["n_candidates"], **extra)
    interps = interp.interpret_neurons(texts=texts["train"], activations=A["train"], neuron_indices=sel, config=icfg)
    tracker.phase = "scoring"
    t1 = time.time()
    fid = interp.score_interpretations(texts=texts["train"], activations=A["train"], interpretations=interps,
                                       config=ScoringConfig(n_examples=sc["n_examples"],
                                                            max_words_per_example=sc["max_words"],
                                                            sampling_function=sample_percentile_bins,
                                                            sampling_kwargs=pct), **annot_kwargs(cfg, "scoring"))
    coef_of = dict(zip(sel, coefs))
    rows = []
    for idx, f in fid.items():
        best = max(interps[idx], key=lambda s: f[s]["f1"] if s is not None else -1)
        rows.append({"neuron_idx": int(idx), score_col(cfg): coef_of[idx], "interpretation": best,
                     "f1_fidelity": float(f[best]["f1"]), "precision": float(f[best]["precision"]),
                     "recall": float(f[best]["recall"])})
    json.dump({"M": M, "K": K, "selected": sel, "coefs": coefs, "hypotheses": rows,
               "all_candidates": {str(k): v for k, v in interps.items()},
               "all_fidelity": {str(k): v for k, v in fid.items()},
               "wall_min": {"interpretation": (t1 - t0) / 60, "scoring": (time.time() - t1) / 60},
               "usage": tracker.report().to_dict(orient="records")},
              open(ckpt, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.max_colwidth", 120, "display.width", 250)
    print(pd.DataFrame(rows).sort_values(score_col(cfg)).round(3).to_string(index=False))
    print(tracker.report().round(4).to_string(index=False))


def stage_holdout(cfg, dfs, texts, y, tracker):
    from hypothesaes.annotate import annotate_texts_with_concepts
    from hypothesaes.quickstart import evaluate_hypotheses
    done = fpath(cfg, "holdout_metrics", ".json")
    print("hypothesis run:", repr(cfg["run"]))
    hyp = pd.DataFrame(json.load(open(fpath(cfg, "pre_eval_hypotheses", ".json")))["hypotheses"])
    a, akw = cfg["annotation"], annot_kwargs(cfg)
    check_cache_model(f"{cfg['cache_name']}_hypothesis-eval.json", a["model"])
    durable_annotation_log(cfg["cache_name"])
    tracker.phase = "holdout_eval"
    t = time.time()
    hs, th = hyp["interpretation"].tolist(), texts["holdout"]
    for s in range(0, len(th), 200):  # same arguments as evaluate_hypotheses, which then finds everything cached
        annotate_texts_with_concepts(texts=th[s:s + 200], concepts=hs, max_words_per_example=a["max_words"],
                                     model=a["model"], cache_name=cfg["cache_name"], n_workers=a["n_workers"],
                                     show_progress=False, **akw)
        try:
            cost = f"${tracker.report().cost_usd.iloc[-1]:.2f}"
        except Exception:
            cost = "$0 (all cached)"
        print(f"  annotated {min(s + 200, len(th))}/{len(th)} texts, cost this run so far {cost}", flush=True)
    metrics, ev = evaluate_hypotheses(hypotheses_df=hyp, texts=th, labels=y["holdout"], cache_name=cfg["cache_name"],
                                      annotator_model=a["model"], max_words_per_example=a["max_words"],
                                      classification=cfg["classification"], n_workers_annotation=a["n_workers"],
                                      corrected_pval_threshold=cfg["evaluation"]["pval"], annotation_llm_kwargs=akw)
    ev = ev.merge(hyp, left_on="hypothesis", right_on="interpretation", how="left").drop(columns=["interpretation"])
    if cfg["diagnostic_col"]:  # which values of the diagnostic column each hypothesis fires on (cached annotations)
        c = cfg["diagnostic_col"]
        ann = annotate_texts_with_concepts(texts=th, concepts=ev["hypothesis"].tolist(), model=a["model"],
                                           cache_name=cfg["cache_name"], show_progress=False,
                                           max_words_per_example=a["max_words"], use_cache_only=True, **akw)
        ev[f"holdout_top_{c}"] = [top_values(dfs["holdout"][c].values, ann[h] == 1) for h in ev["hypothesis"]]
    usage = tracker.report()
    ev.to_csv(fpath(cfg, "holdout_hypotheses", ".csv"), index=False)
    usage.to_csv(fpath(cfg, "usage_holdout", ".csv"), index=False)
    json.dump({"metrics": metrics, "wall_min": (time.time() - t) / 60,
               "cost_usd": float(usage.cost_usd.iloc[-1]) if len(usage) else 0.0},
              open(done, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    pd.set_option("display.max_colwidth", 110, "display.width", 250)
    cols = ["hypothesis", "separation_score", "regression_coef", "regression_pval", "feature_prevalence",
            score_col(cfg), "f1_fidelity"]
    print(ev[[x for x in cols if x in ev]].sort_values("separation_score").round(3).to_string(index=False))
    print("holdout metrics:", metrics)
    print(usage.round(4).to_string(index=False))
    for name in cfg["after_holdout"]:
        STAGES[name](cfg, dfs, texts, y, tracker)


def stage_recovery(cfg, dfs, texts, y, tracker=None):
    """Paper section 5.2: annotate hypotheses on the holdout; Hungarian matching on the (reference x hypothesis)
    correlation matrix between annotations and true class membership; for matched pairs report F1 similarity and
    LLM surface similarity (same / related / distinct = 1 / 0.5 / 0, averaged over samples, T=0.7)."""
    import concurrent.futures
    import re
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr
    from hypothesaes.annotate import annotate_texts_with_concepts, CACHE_DIR
    from hypothesaes.evaluation import evaluate_predicate_surface_similarity
    rc = dict({"sim_samples": 5, "sim_temperature": 0.7}, **cfg["recovery"])
    tracker = tracker or UsageTracker()
    hypotheses = pd.read_csv(fpath(cfg, "holdout_hypotheses", ".csv"))["hypothesis"].tolist()
    # reference hypotheses = the positive classes, phrased like the paper's Wiki topics 'l1 >> l2 >> l3'
    camel = lambda s: re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    pos = json.load(open(os.path.join(ROOT, rc["positive_classes_file"])))[rc["positive_classes_key"]]
    cols = rc["hierarchy_cols"]
    hier = pd.concat(dfs.values())[cols].drop_duplicates().set_index(cols[-1])
    refs = {c: " >> ".join(camel(x) for x in [*(hier.loc[c, h] for h in cols[:-1]), c]) for c in pos}
    ref_c = list(refs)
    a = cfg["annotation"]
    ann = annotate_texts_with_concepts(texts=texts["holdout"], concepts=hypotheses, model=a["model"],
                                       cache_name=cfg["cache_name"], show_progress=False,
                                       max_words_per_example=a["max_words"], use_cache_only=True, **annot_kwargs(cfg))
    cls_hold = dfs["holdout"][cols[-1]].values
    C, F = np.zeros((len(ref_c), len(hypotheses))), np.zeros((len(ref_c), len(hypotheses)))
    for i, c in enumerate(ref_c):
        yy = cls_hold == c
        for j, h in enumerate(hypotheses):
            aa = ann[h].astype(bool)
            C[i, j] = pearsonr(yy, aa)[0] if aa.std() > 0 and yy.std() > 0 else 0.0
            F[i, j] = 2 * (aa & yy).sum() / (aa.sum() + yy.sum()) if (aa.sum() + yy.sum()) else 0.0
    pairs = list(zip(*linear_sum_assignment(-C)))
    tracker.phase = "recovery_surface_sim"
    sim_path = os.path.join(CACHE_DIR, f"{cfg['cache_name']}_surface-sim_{rc['sim_model']}.json")
    sim = json.load(open(sim_path)) if os.path.exists(sim_path) else {}
    key = lambda r, h: f"{r}|||{h}"
    todo = [(refs[ref_c[i]], hypotheses[j]) for i, j in pairs if key(refs[ref_c[i]], hypotheses[j]) not in sim]
    print(f"surface similarity: {len(pairs)} matched pairs, {len(todo)} to query with {rc['sim_model']}")

    def one(p):
        try:
            return p, evaluate_predicate_surface_similarity(p[0], p[1], n_samples=rc["sim_samples"],
                                                            model=rc["sim_model"], temperature=rc["sim_temperature"])
        except Exception as e:  # e.g. no parsable answer in any sample
            print("surface-sim failed:", e)
            return p, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        for (r, h), s in ex.map(one, todo):
            if s is not None:
                sim[key(r, h)] = s
    json.dump(sim, open(sim_path, "w"))
    m = pd.DataFrame([{"reference_class": ref_c[i], "reference": refs[ref_c[i]], "hypothesis": hypotheses[j],
                       "holdout_corr": C[i, j], "f1_similarity": F[i, j],
                       "surface_similarity": sim.get(key(refs[ref_c[i]], hypotheses[j]), np.nan)} for i, j in pairs]
                     ).sort_values("holdout_corr", ascending=False)
    summary = {"n_references": len(ref_c), "n_hypotheses": len(hypotheses), "n_matched": len(pairs),
               "sim_model": rc["sim_model"], "sim_samples": rc["sim_samples"],
               "mean_f1_similarity": float(m["f1_similarity"].mean()),
               "mean_surface_similarity": float(m["surface_similarity"].mean()),
               "mean_holdout_corr": float(m["holdout_corr"].mean()),
               "n_exact_surface_sim_eq_1": int((m["surface_similarity"] >= 0.999).sum()),
               "n_surface_sim_ge_0.5": int((m["surface_similarity"] >= 0.5).sum()),
               "n_refs_any_hyp_f1_ge_0.5": int((F.max(axis=1) >= 0.5).sum())}
    pd.set_option("display.max_colwidth", 70, "display.width", 250)
    print(m.round(3).to_string(index=False))
    print("recovery summary:", json.dumps(summary, indent=2))
    m.to_csv(fpath(cfg, "recovery_matches", ".csv"), index=False)
    pd.DataFrame(C, index=ref_c, columns=hypotheses).to_csv(fpath(cfg, "recovery_corrmatrix", ".csv"))
    tracker.report().to_csv(fpath(cfg, "usage_recovery", ".csv"), index=False)
    json.dump(summary, open(fpath(cfg, "recovery_summary", ".json"), "w"), indent=2)


STAGES = {"recovery": stage_recovery}


def main():
    ap = argparse.ArgumentParser(description="HypotheSAEs pipeline driven by a YAML config")
    ap.add_argument("--config", required=True, help="dataset name (configs/<name>.yaml next to this script) or a path to a .yaml file")
    ap.add_argument("--stage", required=True, choices=["grid", "select", "llm", "holdout", "recovery"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set selection.n_select=50 (repeatable)")
    ap.add_argument("--run", default=None, help="hypothesis run (timestamp suffix) for holdout / recovery; "
                    "default: the newest. Use --run '' for the unsuffixed files from before 2026-11-09")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.set:
        print("config overrides:", args.set)
    os.makedirs(cfg["output_dir"], exist_ok=True)
    dfs, texts, y = load_data(cfg)
    print({s: (len(t), round(float(np.mean(y[s])), 3)) for s, t in texts.items()})
    cfg["run"] = resolve_run(cfg, args.run) if args.stage in ("holdout", "recovery") else None
    if args.stage == "recovery":
        if not cfg[args.stage]:
            raise SystemExit(f"this config has no '{args.stage}' section")
        return STAGES[args.stage](cfg, dfs, texts, y)
    uses_api = args.stage in ("llm", "holdout") or cfg["embedding"]["backend"] == "openai"
    tracker = UsageTracker() if uses_api else None
    if args.stage == "holdout":
        return stage_holdout(cfg, dfs, texts, y, tracker)
    E = embed(cfg, texts, tracker)
    print("embedding dim:", E["train"].shape[1])
    if args.stage == "grid":
        stage_grid(cfg, dfs, E, y)
    elif args.stage == "select":
        stage_select(cfg, dfs, E, y)
    elif args.stage == "llm":
        stage_llm(cfg, dfs, texts, E, y, tracker)
    if tracker is not None and args.stage != "llm":
        print(tracker.report().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
