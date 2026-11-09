# HypotheSAEs on two new datasets: DBpedia and 10-K

Everything we added to the HypotheSAEs repository is in this folder; the library in `hypothesaes/` is unchanged.
The folders follow the sections of the report. Run every command from the repository root.

```
implement_new_dataset/
├── paths.py                  shared paths (imported by the scripts)
├── 1_datasets/               data preparation
├── 2_pipeline/               run_pipeline.py + configs/ + results/{dbpedia,10k}/
├── 3_baselines/              scripts/ + results/ (one folder per report subsection)
└── 4_experiments/            one folder per experiment: script(s) + results/
```

Large or regenerable files stay at the repository root and are not committed: `data/`, `emb_cache/`,
`annotation_cache/` (API caches), `checkpoints/` (SAE weights), `logs/`. Scripts that call the OpenAI API read the
key from `OPENAI_KEY_SAE`. With the caches in place, every result below can be recomputed without API calls.

## 1. Datasets (`1_datasets/`)

| Script | Output (`data/`) |
|---|---|
| `prepare_dbpedia.py` | `dbpedia/processed/12k/{train,val,holdout}.json`, `dbpedia/positive_l3_classes.json` |
| `prepare_10k.py` (stages `prepare`, `split`) | `10k/{train,val,holdout}.json`, `10k/reports.json`, `10k/split_info.json` |

- **DBpedia** (Kaggle "DBpedia Classes"; raw CSVs in `data/dbpedia/raw/`): 50 of the l3 classes with at least 2,000
  articles are drawn as positive classes (seed 42), y = 1 for 38.2% of the texts. 8,000 train / 2,000 val /
  2,000 holdout, stratified by l3. Hypotheses should recover the 50 classes.
- **10-K** (10-K Corpus, http://clip.csie.org/10K/; raw files in `data/10k/raw/`): MD&A sections filed in 2010-2013,
  cleaned, masked (rules + spaCy NER: company, tickers, people, dates, years relative to the fiscal year), cut into
  10-sentence chunks (at most 5 per report), split by company. y = post-filing log volatility minus the train mean of
  its filing year, divided by the train SD. `implement_new_dataset/1_datasets/10k_excluded_holdout_keys.json` lists 2,000 reports kept out of the
  holdout because they were used in an earlier evaluation.

## 2. Pipeline (`2_pipeline/`)

`run_pipeline.py` runs the paper's six steps for any dataset. Its defaults (`PAPER_DEFAULTS`) are the paper's settings;
`configs/<dataset>.yaml` holds the dataset's paths and every setting we changed, with a comment on why;
`configs/sample.yaml` lists every option with its default value.

```
python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage grid      # steps 1-3 for each (M, K): choose M, K
python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage select    # steps 1-3 + no-LLM baselines
OPENAI_KEY_SAE=... python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage llm      # steps 4-5
OPENAI_KEY_SAE=... python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage holdout  # step 6
OPENAI_KEY_SAE=... python implement_new_dataset/2_pipeline/run_pipeline.py --config dbpedia --stage recovery  # DBpedia only
```

- `--set key.sub=value` overrides a config value for one run (used in 4.1).
- Each `llm` run writes `pre_eval_hypotheses_<YYYYMMDD-HHMMSS>.json` (never overwritten); `holdout` and `recovery` take
  `--run <timestamp>` (default: newest). The reported results predate the timestamps and have none (`--run ''`).
- Engineering on top of the library: annotation caches are keyed only by (hypothesis, text), so
  `annotation_cache/cache_models.json` records which model filled each cache and a different model is refused;
  every holdout annotation is appended to a log as it returns, so an interrupted run loses nothing; the paper's
  recovery metric (Hungarian matching, F1 and LLM surface similarity) is implemented as the `recovery` stage.

Results in `results/{dbpedia,10k}/`: `pre_eval_hypotheses.json` (hypotheses, all candidates, fidelity),
`holdout_hypotheses.csv` (per-hypothesis statistics), `holdout_metrics.json`, `usage_holdout.csv` (tokens, cost);
10-K also `grid_MK.csv`, `chosen_MK.json`; DBpedia also `metrics_*`, `selected_neurons_*` (select stage) and
`recovery_*`.
DBpedia: holdout AUC 0.890, 22/50 significant (p < 0.05/H), surface-similarity recovery 0.750. 10-K: holdout R^2 0.082,
2/20 significant.

## 3. Baselines (`3_baselines/`)

| Report section | Script(s) in `scripts/` | Results in `results/` |
|---|---|---|
| 3.1 Plain predictive models | `bow_baseline_dbpedia.py`, `kogan_baseline_10k.py` | `3.1_plain_models/` |
| 3.2 BERTopic | `export_embeddings.py`, then `run_bertopic.py <dataset> <H>` | `3.2_bertopic/<dataset>/` |
| 3.2 HypoGeniC | `prep_hypogenic.py`, then `run_hypogenic.py <dataset> <H>` | `3.2_hypogenic/<dataset>/` |
| Evaluation 1: LLM labels on 100 val texts | `eval_baselines_val100.py` (~$2.4) | `eval1_llm_labels_val100/` (also holds the 100 texts) |
| Evaluation 2: cosine instead of LLM labels, full val | `eval_baselines_cosine.py` | `eval2_cosine_val/` |
| Evaluation 3: LLM as judge | `llm_judge.py` (~$0.6) | `eval3_llm_judge/` |

BERTopic and HypoGeniC run in their own environments (CPU torch is enough); their inputs go to `data/baselines/`:

```bash
git clone https://github.com/ChicagoHAI/hypothesis-generation.git
uv venv -p python3.11 venv_hypogenic
uv pip install -p venv_hypogenic/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -p venv_hypogenic/bin/python -e ./hypothesis-generation

uv venv -p python3.11 venv_bertopic
uv pip install -p venv_bertopic/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -p venv_bertopic/bin/python openai==1.16.2 "httpx<0.28" transformers==4.44.2 scikit-learn==1.5.2 \
    scipy "numpy<2" tqdm dataclasses-json sentence-transformers bertopic==0.16.4 umap-learn hdbscan pandas
OPENAI_API_KEY=... venv_bertopic/bin/python implement_new_dataset/3_baselines/scripts/run_bertopic.py 10k 20
OPENAI_API_KEY=... venv_hypogenic/bin/python implement_new_dataset/3_baselines/scripts/run_hypogenic.py 10k 20
```

## 4. Further experiments (`4_experiments/`)

Comparisons use the validation set or existing results; the holdout was not re-run for them.

| Folder | Script(s) | What it does | Cost |
|---|---|---|---|
| `4.1_embedding/` | `run_pipeline.py` with `--set` (below) | Qwen3-Embedding-0.6B / 4B instead of text-embedding-3-small | free (GPU) |
| `4.2_ica/` | `ica_features.py --stage screen\|llm --dataset ...` | ICA (and PCA / random-rotation / raw controls) instead of the SAE, on our datasets and the paper's | screen free; llm ~$12 |
| `4.3_fidelity/` | `fidelity.py` | does train fidelity predict holdout quality? | free (cached) |
| `4.4_cheaper_annotation/` | (a) `embedding_annotate.py`; (b) `local_annotate.py`, `run_local_annotators.sh`; (c) `batch_all.py`, `batch_groups.py`, `single_prompt_control.py` | (a) cosine similarity, (b) small local LLMs via vLLM, (c) several hypotheses per call, instead of one gpt-4o-mini call per (text, hypothesis) | (a), (b) free; (c) ~$1.2 |
| `4.5_seed_stability/` | `seed_stability.py` | retrain the SAE with 5 seeds | free (GPU) |
| `4.6_neuron_selection/` | `compare_selection.py` (DBpedia), `compare_selection_10k.py` | 9 neuron-selection methods (+ stability selection on 10-K) | free |

4.1 commands (results in `4.1_embedding/results/<dataset>/<embedding>/`):

```
python implement_new_dataset/2_pipeline/run_pipeline.py --config 10k --stage grid \
  --set embedding.model=Qwen/Qwen3-Embedding-4B --set embedding.backend=local \
  --set 'output_dir=implement_new_dataset/4_experiments/4.1_embedding/results/10k/{embedding_tag}'
python implement_new_dataset/2_pipeline/run_pipeline.py --config dbpedia --stage select \
  --set embedding.model=Qwen/Qwen3-Embedding-4B --set embedding.backend=local \
  --set 'output_dir=implement_new_dataset/4_experiments/4.1_embedding/results/dbpedia/{embedding_tag}'
```

The text-embedding-3-small rows of the 4.1 table come from the main run (`2_pipeline/results/`).

Findings, one line each:

- **4.1** The 4B embedding carries the most information, but its advantage disappears after compression into H SAE
  features (10-K val R^2 with 20 features: 0.161 OpenAI, 0.130 Qwen-0.6B, 0.115 Qwen-4B).
- **4.2** ICA recovers synthetic topics as well as or better than the SAE (Wiki-5 recovery F1 0.63 vs 0.40) but is worse
  on broad real targets (Yelp R^2 0.49 vs 0.60); after the full LLM pipeline it is on par on DBpedia and 10-K.
- **4.3** Train fidelity predicts holdout quality on DBpedia (rho = 0.54-0.55) but not detectably on 10-K (20 hypotheses).
- **4.4** Cosine similarity nearly replaces the LLM on DBpedia (kappa 0.72, 16 of 22 significant hypotheses kept); local
  LLMs under 1B parameters fail (kappa 0.02-0.17); batching hypotheses cuts cost up to 14x but drops DBpedia recovery
  F1 from 0.67 to 0.40-0.47.
- **4.5** Val performance is stable across seeds (DBpedia AUC 0.899 +/- 0.012, 10-K R^2 0.135 +/- 0.017), but only 73% /
  56% of the selected features reappear under another seed.
- **4.6** Positive-direction-only LASSO is best on DBpedia (AUC 0.944 vs 0.913) and worst on 10-K (R^2 0.094 vs 0.161);
  the paper's LASSO is a safe default.
