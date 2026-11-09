"""Export cached text-embedding-3-small embeddings (train/val) of DBpedia 12k and 10-K to
data/baselines/emb/<dataset>_<split>.npy, so that BERTopic can run in its own venv without new API calls.
Usage (main python, from the repo root): python implement_new_dataset/3_baselines/scripts/export_embeddings.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # implement_new_dataset/
import paths  # noqa: E402  (puts the library, run_pipeline.py and shared modules on sys.path)
ROOT = paths.ROOT
os.chdir(ROOT)  # relative paths below are from the repository root
from hypothesaes.embedding import load_embedding_cache  # noqa: E402

OUT = f"{ROOT}/data/baselines/emb"; os.makedirs(OUT, exist_ok=True)
for name, cache, path in [("dbpedia", "dbpedia_text-embedding-3-small", "data/dbpedia/processed/12k/{}.json"),
                          ("10k", "10k_text-embedding-3-small", "data/10k/{}.json")]:
    c = load_embedding_cache(cache)
    for s in ["train", "val"]:
        d = pd.read_json(os.path.join(ROOT, path.format(s)), lines=True)
        np.save(f"{OUT}/{name}_{s}.npy", np.stack([c[t] for t in d.text]).astype(np.float32))
        print(name, s, len(d))
