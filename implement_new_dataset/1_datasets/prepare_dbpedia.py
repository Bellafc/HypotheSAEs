"""Prepare DBpedia (l3 labels) for HypotheSAEs at three sizes.

Usage:
    python implement_new_dataset/1_datasets/prepare_dbpedia.py --raw_dir data/dbpedia/raw --out_dir data/dbpedia

Outputs (jsonl, same format as demo_data/):
    {out_dir}/positive_l3_classes.json
    {out_dir}/processed/{12k,full}/{train,val,holdout}.json
Columns: text, l1, l2, l3, label (1 if l3 in positive classes)

Example row (data/dbpedia/processed/12k/train.json, row 3):
    {"text": "1740 Paavo Nurmi, provisional designation 1939 UA, is an asteroid from the inner regions of the
              asteroid belt. It was discovered by Finnish astronomer Yrjö Väisälä at Turku Observatory ...",  (164 words)
     "l1": "Place", "l2": "CelestialBody", "l3": "Planet", "label": 1}
  text  = the beginning of a Wikipedia article
  label = 1 if its l3 class is one of the 50 positive classes (Planet is one) -> the target to predict;
  the hypotheses should RECOVER those 50 classes (e.g. "describes an asteroid" <-> Planet).
  l1 / l2 / l3 are used only to evaluate recovery, never for training.

Design:
  - Positive classes: N_POS l3 classes sampled at random (fixed seed) from classes with
    >= MIN_CLASS_SIZE docs, so y=1 is ~40% of the data and not skewed toward any l2 group.
  - Splits are nested (12k ⊂ 56k ⊂ full) and stratified by l3. The 56k split is only an intermediate sampling step
    (12k is drawn from it, so keeping it keeps 12k unchanged); it is not written, since no result uses it.
  - train/val come from the original train/val files; holdout (2K, shared by all sizes)
    comes from the original test file, so there's no leakage.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_POS = 50
MIN_CLASS_SIZE = 2000
HOLDOUT_N = 2000
SIZES = {  # name: (n_train, n_val); None = use the whole original split
    "12k": (8000, 2000),
    "56k": (50000, 3000),  # intermediate only, not saved
    "full": (None, None),
}


def strat_sample(df, n, seed):
    """Stratified (by l3) subsample of exactly n rows."""
    if n is None or n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    sub, _ = train_test_split(df, train_size=n, stratify=df["l3"], random_state=seed)
    return sub.reset_index(drop=True)


def save_jsonl(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/dbpedia/raw")
    ap.add_argument("--out_dir", default="data/dbpedia")
    args = ap.parse_args()

    raw = {s: pd.read_csv(os.path.join(args.raw_dir, f"DBPEDIA_{s}.csv")) for s in ["train", "val", "test"]}
    all_df = pd.concat(raw.values(), ignore_index=True)

    # ---- choose positive l3 classes ----
    vc = all_df["l3"].value_counts()
    candidates = sorted(vc[vc >= MIN_CLASS_SIZE].index.tolist())
    rng = np.random.default_rng(SEED)
    pos = sorted(rng.choice(candidates, size=N_POS, replace=False).tolist())
    pos_set = set(pos)
    share = all_df["l3"].isin(pos_set).mean()
    pos_info = {
        "seed": SEED,
        "n_positive_classes": N_POS,
        "min_class_size": MIN_CLASS_SIZE,
        "positive_share_all_data": round(float(share), 4),
        "positive_l3_classes": pos,
        "positive_by_l2": all_df[all_df["l3"].isin(pos_set)].groupby("l2")["l3"].unique().apply(sorted).to_dict(),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "positive_l3_classes.json"), "w") as f:
        json.dump(pos_info, f, indent=2, default=list)
    print(f"Positive classes: {N_POS}, y=1 share over all data = {share:.3f}")

    for s in raw:
        raw[s] = raw[s][["text", "l1", "l2", "l3"]].copy()
        raw[s]["label"] = raw[s]["l3"].isin(pos_set).astype(int)

    # ---- shared holdout ----
    holdout = strat_sample(raw["test"], HOLDOUT_N, SEED)

    # ---- nested train/val: build from largest to smallest ----
    order = sorted(SIZES, key=lambda k: -(SIZES[k][0] or 10**9))  # full, 56k, 12k
    prev_train, prev_val = raw["train"], raw["val"]
    for name in order:
        n_tr, n_va = SIZES[name]
        tr = strat_sample(prev_train, n_tr, SEED)
        va = strat_sample(prev_val, n_va, SEED)
        prev_train, prev_val = tr, va
        if name == "56k":
            continue
        d = os.path.join(args.out_dir, "processed", name)
        save_jsonl(tr, os.path.join(d, "train.json"))
        save_jsonl(va, os.path.join(d, "val.json"))
        save_jsonl(holdout, os.path.join(d, "holdout.json"))
        print(f"[{name}] train={len(tr)} (pos {tr.label.mean():.3f}, l3={tr.l3.nunique()}) "
              f"val={len(va)} (pos {va.label.mean():.3f}) holdout={len(holdout)} (pos {holdout.label.mean():.3f})")
        prev_train, prev_val = tr, va


if __name__ == "__main__":
    main()
