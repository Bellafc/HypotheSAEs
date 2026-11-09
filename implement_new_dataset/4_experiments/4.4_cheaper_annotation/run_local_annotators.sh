#!/usr/bin/env bash
# Second pass: each small model with the repo prompt it can actually follow
# (Qwen3-0.6B -> annotate-simple; Llama-3.2-1B -> JSON prompt), then compare all annotators.
# Usage: HF_TOKEN=... DATASET=dbpedia|10k bash implement_new_dataset/4_experiments/4.4_cheaper_annotation/run_local_annotators.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repository root
PY=/usr/local/bin/python
VLLM=/workspace/vllm_env/bin/vllm
export PATH=/workspace/vllm_env/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
PORT=8123
RUNS=("Qwen/Qwen3-0.6B simple 8" "meta-llama/Llama-3.2-1B-Instruct json 128")
DATASET=${DATASET:-dbpedia}
mkdir -p logs

stop_servers() {  # by PID, matching only the vLLM server processes
  for pid in $(ps -eo pid,args | awk '/vllm_env\/bin\/(vllm|python).* serve /{print $1}'); do kill "$pid" 2>/dev/null; done
  for i in $(seq 1 30); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || break; sleep 2; done
}

for RUN in "${RUNS[@]}"; do
  read -r MODEL PROMPT MAXTOK <<< "$RUN"
  TAG=${DATASET}_$(basename "$MODEL")_$PROMPT
  CUR=$(curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -o "\"id\":\"[^\"]*\"" | head -1)
  if [[ "$CUR" != "\"id\":\"$MODEL\"" ]]; then
    stop_servers
    echo "[$(date -u +%H:%M:%S)] starting vLLM for $MODEL"
    $VLLM serve "$MODEL" --port $PORT --gpu-memory-utilization 0.85 --max-model-len 4096 \
          --enable-prefix-caching > "logs/vllm_${TAG}.log" 2>&1 &
    for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 5; done
  fi
  echo "[$(date -u +%H:%M:%S)] annotating with $MODEL ($PROMPT prompt)"
  $PY implement_new_dataset/4_experiments/4.4_cheaper_annotation/local_annotate.py annotate --dataset "$DATASET" --model "$MODEL" --prompt "$PROMPT" --max_tokens "$MAXTOK" \
      --base_url "http://127.0.0.1:$PORT/v1" --n_workers 256 > "logs/local_annotate_${TAG}.log" 2>&1
  echo "[$(date -u +%H:%M:%S)] annotate exit code $? for $TAG"
done
stop_servers

echo "[$(date -u +%H:%M:%S)] comparing annotators"
$PY implement_new_dataset/4_experiments/4.4_cheaper_annotation/local_annotate.py compare --dataset "$DATASET" --models "Qwen/Qwen3-0.6B:simple" \
    "meta-llama/Llama-3.2-1B-Instruct:json" > logs/compare_annotators_${DATASET}.log 2>&1
echo "[$(date -u +%H:%M:%S)] done (exit $?)"
