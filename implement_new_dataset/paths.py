"""Shared paths for implement_new_dataset/. Importing this module also puts the repository root (the hypothesaes
library), the pipeline (run_pipeline.py) and the shared experiment modules on sys.path.

Data, embedding caches, annotation caches and SAE checkpoints stay at the repository root (data/, emb_cache/,
annotation_cache/, checkpoints/), where the library expects them."""
import os
import sys

NEW = os.path.dirname(os.path.abspath(__file__))           # implement_new_dataset/
ROOT = os.path.dirname(NEW)                                  # repository root
PIPELINE = os.path.join(NEW, "2_pipeline")
BASELINES = os.path.join(NEW, "3_baselines")
BASELINE_RESULTS = os.path.join(BASELINES, "results")
VAL100 = os.path.join(BASELINE_RESULTS, "eval1_llm_labels_val100")  # the 100 val texts shared by several experiments
EXP = os.path.join(NEW, "4_experiments")


def pipeline_results(ds):
    """results of the main run (configs/<ds>.yaml): implement_new_dataset/2_pipeline/results/<ds>/"""
    return os.path.join(PIPELINE, "results", ds)


def exp_dir(name):
    """implement_new_dataset/4_experiments/<name>/, e.g. exp_dir('4.3_fidelity')"""
    return os.path.join(EXP, name)


for p in [ROOT, PIPELINE, exp_dir("4.6_neuron_selection"), exp_dir("4.4_cheaper_annotation")]:
    if p not in sys.path:
        sys.path.insert(0, p)
