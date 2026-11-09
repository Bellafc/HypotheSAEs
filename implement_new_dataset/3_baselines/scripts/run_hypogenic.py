"""HypoGeniC with the defaults of examples/generation.py (gpt-4o-mini, num_train=75, k=10, alpha=0.5, ...),
max_num_hypotheses = H (50 DBpedia / 20 10-K). Needs the HypoGeniC venv (see implement_new_dataset/README.md) and the data written by
prep_hypogenic.py. Output: implement_new_dataset/3_baselines/results/3.2_hypogenic/<dataset>/hypotheses.json (+ bank_snapshots/).
Usage: OPENAI_API_KEY=... <venv_hypogenic>/bin/python implement_new_dataset/3_baselines/scripts/run_hypogenic.py dbpedia 50   (or 10k 20)"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))  # implement_new_dataset/
import paths as _paths  # noqa: E402
ROOT = _paths.ROOT  # repo root

import json, logging, os, sys, time
from hypogenic.extract_label import extract_label_register
from hypogenic.tasks import BaseTask
from hypogenic.prompt import BasePrompt
from hypogenic.utils import set_seed
from hypogenic.LLM_wrapper import llm_wrapper_register
from hypogenic.algorithm.generation import DefaultGeneration
from hypogenic.algorithm.inference import DefaultInference
from hypogenic.algorithm.replace import DefaultReplace
from hypogenic.algorithm.update import DefaultUpdate
from hypogenic.logger_config import LoggerConfig
LoggerConfig.setup_logger(level=logging.WARNING)

name, H = sys.argv[1], int(sys.argv[2])
d = f"{ROOT}/data/baselines/hypogenic/{name}"
res_dir = f"{ROOT}/implement_new_dataset/3_baselines/results/3.2_hypogenic/{name}"
out = f"{res_dir}/bank_snapshots/"; os.makedirs(out, exist_ok=True)
t0 = time.time()
api = llm_wrapper_register.build("gpt")(model="gpt-4o-mini", path_name=None)
task = BaseTask(f"{d}/config.yaml", extract_label=None, from_register=extract_label_register)  # 'default' extractor
seed = 42; set_seed(seed)
train_data, _, _ = task.get_data(75, 25, 25, seed)
prompt_class = BasePrompt(task)
inference_class = DefaultInference(api, prompt_class, train_data, task)
generation_class = DefaultGeneration(api, prompt_class, inference_class, task)
update = DefaultUpdate(generation_class=generation_class, inference_class=inference_class,
                       replace_class=DefaultReplace(H), save_path=out, num_init=10, k=10, alpha=5e-1,
                       update_batch_size=10, num_hypotheses_to_update=1, save_every_n_examples=10)
bank = update.batched_initialize_hypotheses(10, init_batch_size=10, init_hypotheses_per_batch=10,
                                            cache_seed=None, temperature=1e-5, max_tokens=1000)
bank = update.update(current_epoch=0, hypotheses_bank=bank, current_seed=seed, cache_seed=None)
update.save_to_json(bank, sample="final", seed=seed, epoch=0)
ranked = sorted(bank, key=lambda h: bank[h].reward, reverse=True)
res = {"hypotheses": ranked[:H], "rewards": [bank[h].reward for h in ranked[:H]], "n_bank": len(bank),
       "cost_usd": api.get_cost(), "minutes": (time.time() - t0) / 60}
json.dump(res, open(f"{res_dir}/hypotheses.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "hypotheses"}), "\n" + "\n".join(f"- {h}" for h in ranked[:H]))
