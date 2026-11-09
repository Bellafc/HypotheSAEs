"""Write HypoGeniC-format training data + config.yaml for DBpedia (12k) and 10-K into data/baselines/hypogenic/.
10-K labels: y binarized at the train median (HypoGeniC is classification-only).
Usage (main python, from the repo root): python implement_new_dataset/3_baselines/scripts/prep_hypogenic.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # implement_new_dataset/
import paths as _paths  # noqa: E402
ROOT = _paths.ROOT  # repo root

import json, os
import pandas as pd
H = ROOT
OUT = f"{ROOT}/data/baselines/hypogenic"

DB_TASK = ("an encyclopedia article opening (from DBpedia/Wikipedia) belongs to a hidden TARGET set of article categories "
           "(label 'yes') or not (label 'no')")
TK_TASK = ("the company that filed a 10-K report will have HIGH or LOW stock return volatility in the 12 months after "
           "filing (above vs. below the median of its filing year), based on an excerpt from the report's MD&A section. "
           "Company names, people, dates and years are masked with placeholders such as [COMPANY], [PERSON], [DATE], [YEAR_T]")
CFG = {
 "dbpedia": dict(obs='An article opening is the following: "${text}"\nThe label is: ${label}.',
                 who="You're a data scientist studying a text classification dataset.",
                 task=DB_TASK, labels=("yes", "no"), q="Does the article belong to the target set? Answer should be \"yes\" or \"no\"."),
 "10k": dict(obs='An MD&A excerpt is the following: "${text}"\nThe volatility label is: ${label}.',
              who="You're a financial economist studying how the text of companies' annual reports relates to their future stock return volatility.",
              task=TK_TASK, labels=("high", "low"), q="Will the company have high or low volatility? Answer should be \"high\" or \"low\"."),
}

def yaml_block(s, ind=6):
    return "\n".join(" " * ind + l for l in s.split("\n"))

def write(name, df):
    c = CFG[name]; d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    data = {"text": df.text.tolist(), "label": df.lab.tolist()}
    for split in ["train", "val", "test"]:
        json.dump(data if split == "train" else {k: v[:50] for k, v in data.items()}, open(f"{d}/{name}_{split}.json", "w"))
    a, b = c["labels"]
    gen_sys = (f"{c['who']}\nWe want to generate hypotheses that are useful for predicting whether {c['task']}.\n"
               "Using the given examples, please propose ${num_hypotheses} possible hypotheses.\n"
               "These hypotheses should identify specific patterns that occur across the provided examples.\n"
               f"Each hypothesis should describe what makes the label more likely to be \"{a}\" or \"{b}\".\n"
               "Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis].")
    gen_user = ("We have seen some examples:\n${observations}\nPlease generate hypotheses that are useful for the prediction.\n"
                "Propose ${num_hypotheses} possible hypotheses. Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... ${num_hypotheses}. [hypothesis].\nProposed hypotheses:")
    inf_sys = (f"{c['who']}\nFrom past experiences, you learned a pattern.\nYou need to determine whether the pattern holds for the current example, "
               f"and also predict the label.\nGive an answer. The answer should be \"{a}\" or \"{b}\".\nGive your final answer in the format of {{Final answer: answer}}")
    inf_user = ("Our learned patterns: ${hypothesis}\n" + c["obs"].split("\n")[0] + "\n\n"
                "Given the pattern you learned above, predict the label.\nThink step by step.\n"
                "First step: Consider if the pattern can be applied to the example.\n"
                f"Second step: Based on the pattern, what is the label? {c['q']}\n"
                "Final step: give your final answer in the format of {Final answer: answer}")
    y = f"""task_name: {name}

train_data_path: ./{name}_train.json
val_data_path: ./{name}_val.json
test_data_path: ./{name}_test.json
prompt_templates:
  observations:
    multi_content: |
{yaml_block(c['obs'])}
  few_shot_prefix: |
    We have seen some examples:

  batched_generation:
    system: |-
{yaml_block(gen_sys)}
    user: |-
{yaml_block(gen_user)}
  inference:
    system: |-
{yaml_block(inf_sys)}
    user: |-
{yaml_block(inf_user)}
"""
    open(f"{d}/config.yaml", "w").write(y)
    print(name, len(df), df.lab.value_counts().to_dict())

db = pd.read_json(f"{H}/data/dbpedia/processed/12k/train.json", lines=True)
db["lab"] = db.label.map({1: "yes", 0: "no"}); write("dbpedia", db)
tk = pd.read_json(f"{H}/data/10k/train.json", lines=True)
med = tk.y.median(); tk["lab"] = (tk.y > med).map({True: "high", False: "low"}); write("10k", tk)
os.makedirs(f"{ROOT}/implement_new_dataset/3_baselines/results/3.2_hypogenic", exist_ok=True)
json.dump({"10k_train_median_y": float(med)}, open(f"{ROOT}/implement_new_dataset/3_baselines/results/3.2_hypogenic/label_info.json", "w"))
