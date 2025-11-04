#!/usr/bin/env bash
#
# Fully operated sbatch for multi-node SGLang inference on AMD MI210.
# - Partial GPUs per node (e.g., use 4 out of 8 GPUs per node).
# - Set HIP_VISIBLE_DEVICES precisely per node to avoid conflicts and NUMA mix-ups.
# - Architecture: multiple workers + a router — one SGLang worker per node runs local TP; the router coordinates data parallelism across nodes.
# - Compared with reference [0]: the reference starts a single multi-node server via two srun commands (--tp/--nnodes/--node-rank/--nccl-init-addr). This script aggregates independent workers via sglang_router.
#   Benefit: easier topology control and resource isolation for partial GPU usage per node; convenient for same-NUMA vs cross-NUMA throughput comparisons on MI210.
#
# [0] Multi-Node LLM Inference with SGLang on SLURM-Enabled Clusters: https://aflah02.substack.com/p/multi-node-llm-inference-with-sglang
#SBATCH -p faculty
#SBATCH --qos=gtqos
#SBATCH --nodes 2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -t 12:00:00
#SBATCH -J qwen30b_tp4_dp2
#SBATCH -o $HOME/mi210-infer-osdi26/logs/qwen30b/sbatch_%j.out
#SBATCH -e $HOME/mi210-infer-osdi26/logs/qwen30b/sbatch_%j.err
set -euo pipefail

# Notes:
# - --gpus-per-node=4 matches MI210 nodes with 8 GPUs when using only 4 GPUs (partial usage).
# - --ntasks=2 means two tasks (we run one worker per node with a router coordinating across nodes).
# - Tune values per your cluster; for same- vs cross-NUMA experiments, place the 4 GPUs within a single NUMA domain vs across NUMA domains to compare throughput.

# --- User tunables ---------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-$HOME/models/qwen30b}"
DIST_PORT="${DIST_PORT:-50000}"
TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-5050}"
SERVER_WARMUP="${SERVER_WARMUP:-90}"
WORKER_GPUS_PER_NODE="${WORKER_GPUS_PER_NODE:-4}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-31000}"
LOG_TAG="${LOG_TAG:-tp${TP_SIZE}_dp${DP_SIZE}_router}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph}"
EXTRA_ROUTER_ARGS="${EXTRA_ROUTER_ARGS:-}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-20000}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-128}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_DATASET_PATH="${BENCH_DATASET_PATH:-$HOME/sglang-test/SharGPT.json}"
HIP_DEVICES_DEFAULT="${HIP_DEVICES_DEFAULT:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-$HOME/mi210-infer-osdi26/logs/qwen30b/output_logs}"
export NCCL_SOCKET_IFNAME=enp135s0f0np0
export GLOO_SOCKET_IFNAME=enp135s0f0np0
export FULL_MAP=True
export EXPERT_ROUTE=False
export REQUEST_LEVEL=False
export NO_PROXY=127.0.0.1,localhost,0.0.0.0
export no_proxy=127.0.0.1,localhost,0.0.0.0
unset http_proxy HTTPS_PROXY https_proxy HTTP_PROXY
#
# NIC note:
# - Set NCCL/GLOO IFNAME to your high-speed NIC (IB/RoCE/Ethernet). Example uses enp135s0f0np0.
# - If using IB, typical names are ib0: export NCCL_SOCKET_IFNAME=ib0; export GLOO_SOCKET_IFNAME=ib0.
# - Add all allocated nodes to NO_PROXY/no_proxy to avoid proxy overhead across nodes.
# ----------------------------------------------------------------------------

declare -a WORKER_SRUN_PIDS=()
declare -a WORKER_URLS=()

SCRIPT_ROOT="$HOME/mi210-infer-osdi26/scripts"
BENCH_SCRIPT="${BENCH_SCRIPT:-$SCRIPT_ROOT/qwen30b/sglang_bench.sh}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$BENCH_SCRIPT" ]]; then
  echo "[ERROR] Benchmark script not found at $BENCH_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$BENCH_DATASET_PATH" ]]; then
  echo "[WARN] Benchmark dataset not found at $BENCH_DATASET_PATH; random sampling will fallback to download." >&2
fi

source "$HOME/.bashrc"
conda activate sglang-env

export NCCL_DEBUG=${NCCL_DEBUG:-DEBUG}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp135s0f0np0}

cleanup() {
  if [[ -n "${ROUTER_PID:-}" ]]; then
    kill "$ROUTER_PID" >/dev/null 2>&1 || true
    wait "$ROUTER_PID" >/dev/null 2>&1 || true
  fi
  if [[ ${#WORKER_SRUN_PIDS[@]} -gt 0 ]]; then
    for pid in "${WORKER_SRUN_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    done
  fi
}
trap cleanup EXIT

set_local_gpus() {
  # Per-node visible device setting:
  # - Prefer SLURM_STEP_GPUS (e.g., "0,1,2,3") available inside an srun step.
  # - Else, use job-level SLURM_JOB_GPUS.
  # - Else, fallback to HIP_DEVICES_DEFAULT (0,1,2,3).
  if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    HIP_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    HIP_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
  else
    HIP_VISIBLE_DEVICES="${HIP_DEVICES_DEFAULT}"
  fi
  export HIP_VISIBLE_DEVICES
  export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
  export HSA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
  export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
}

NODELIST_SRC="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
if [[ -z "$NODELIST_SRC" ]]; then
  echo "[ERROR] SLURM did not provide a node list (SLURM_NODELIST/SLURM_JOB_NODELIST empty)." >&2
  exit 1
fi
mapfile -t NODE_LIST < <(scontrol show hostnames "$NODELIST_SRC")
if [[ ${#NODE_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] Failed to resolve hostnames from node list: $NODELIST_SRC" >&2
  exit 1
fi
NUM_WORKERS="${#NODE_LIST[@]}"

if (( NUM_WORKERS > 0 )); then
  hosts_csv=$(IFS=','; echo "${NODE_LIST[*]}")
  export NO_PROXY="${NO_PROXY},${hosts_csv}"
  export no_proxy="${no_proxy},${hosts_csv}"
fi

echo "[INFO] SLURM allocated nodes (${NUM_WORKERS}): ${NODE_LIST[*]}"
echo "[INFO] WORKER_GPUS_PER_NODE=$WORKER_GPUS_PER_NODE"
echo "[INFO] TP_SIZE=$TP_SIZE (per worker); router handles data parallelism across workers."

# Compare with reference [0]:
# - The reference starts a single multi-node server using two srun commands with --tp/--nnodes/--node-rank/--nccl-init-addr.
# - Here, each node runs an independent SGLang worker (local TP), and sglang_router coordinates DP across nodes.
#   This approach is simpler and more robust when GPUs per node are partially utilized; it also facilitates same-/cross-NUMA comparisons on MI210.

mkdir -p SLURM_Logs
for idx in "${!NODE_LIST[@]}"; do
  node="${NODE_LIST[$idx]}"
  port=$((WORKER_BASE_PORT + idx))
  worker_url="http://${node}:${port}"
  WORKER_URLS+=("$worker_url")
  echo "[INFO] Launching worker on ${node} (port $port)"
  srun --nodes=1 \
       --nodelist="$node" \
       --gres="gpu:${WORKER_GPUS_PER_NODE}" \
       --ntasks=1 \
       --cpus-per-task="$SLURM_CPUS_PER_TASK" \
       --output="SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${node}_worker.out" \
       --error="SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${node}_worker.err" \
       bash -lc "$(declare -f set_local_gpus); set_local_gpus; \
         unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
         # Per-node cache directory to avoid sqlite or other cache conflicts
         export OUTLINES_CACHE_DIR=\"/tmp/\$(hostname -s)_cache_${SLURM_JOB_ID}\"; \
         echo \"[$(hostname -f)] worker HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES\"; \
         python3 -m sglang.launch_server \
           --tp-size $TP_SIZE \
           --pp-size $PP_SIZE \
           --disable-radix-cache \
           --enable-dp-attention \
           --mem-fraction-static 0.75 \
           --disable-cuda-graph \
           --max-running-requests 5050 \
           --max-micro-batch-size $BATCH_SIZE \
           --chunked-prefill-size -1 \
           --max-prefill-tokens 32768 \
           --attention-backend triton \
           --log-level debug \
           --log-requests \
           --log-requests-level 3 \
           --decode-log-interval 1 \
           --enable-metrics-for-all-schedulers \
           --enable-request-time-stats-logging \
           --model-path $MODEL_PATH \
           --host 0.0.0.0 \
           --port $port \
           --device cuda \
           $EXTRA_SERVER_ARGS" &
  WORKER_SRUN_PIDS+=("$!")
done

for idx in "${!NODE_LIST[@]}"; do
  node="${NODE_LIST[$idx]}"
  port=$((WORKER_BASE_PORT + idx))
  echo "[INFO] Waiting for worker ${node}:${port} to become ready"
  until nc -z "$node" "$port"; do
    sleep 1
    echo "[INFO] Still waiting for ${node}:${port}"
  done
  echo "[INFO] Worker ${node}:${port} is ready."
done

WORKER_URLS_JOINED=$(IFS=' '; echo "${WORKER_URLS[*]}")

ROUTER_LOG="SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_router.out"
echo "[INFO] Launching router on $(hostname -f):${ROUTER_PORT}"
echo "[INFO] Worker URLs: ${WORKER_URLS[*]}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
python3 -m sglang_router.launch_router \
  --worker-urls "${WORKER_URLS[@]}" \
  --host 0.0.0.0 \
  --port "$ROUTER_PORT" \
  --policy "$ROUTER_POLICY" \
  $EXTRA_ROUTER_ARGS \
  >"$ROUTER_LOG" 2>&1 &
ROUTER_PID=$!

echo "[INFO] Router log: $ROUTER_LOG"

echo "[INFO] Waiting for router localhost:${ROUTER_PORT} to accept connections"
until nc -z localhost "$ROUTER_PORT"; do
  sleep 1
  echo "[INFO] Router still starting..."
done
echo "[INFO] Router is ready on port ${ROUTER_PORT}"

# Health check: ensure the router has detected all workers and they are healthy.

echo "[INFO] Waiting for router to report healthy workers"
while true; do
  workers_json=$(curl --noproxy '*' -sf "http://127.0.0.1:${ROUTER_PORT}/workers" || true)
  if [[ -n "$workers_json" ]]; then
    if WORKERS_JSON="$workers_json" EXPECTED_WORKERS="$WORKER_URLS_JOINED" python3 - <<'PY'
import json
import os

payload = os.environ.get("WORKERS_JSON", "")
expected = [u for u in os.environ.get("EXPECTED_WORKERS", "").split(" ") if u]
try:
    data = json.loads(payload)
except json.JSONDecodeError:
    raise SystemExit(1)
workers = {w.get("url"): w.get("status") for w in data.get("workers", [])}
if expected and all(workers.get(url) == "healthy" for url in expected):
    print("[INFO] Router has healthy worker(s).")
    raise SystemExit(0)
raise SystemExit(1)
PY
    then
      break
    fi
  fi
  sleep 2
  echo "[INFO] Router waiting on healthy workers..."
done

BATCH_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
declare -a BATCH_PIDS=()

for idx in "${!WORKER_URLS[@]}"; do
  worker_url="${WORKER_URLS[$idx]}"
  worker_host="${worker_url#http://}"
  worker_host="${worker_host%%:*}"
  worker_port="${worker_url##*:}"
  result_file="$LOG_DIR/${LOG_TAG}_worker${idx}_batch_${BATCH_TIMESTAMP}.json"
  echo "[INFO] Launching single-batch benchmark (size=$BATCH_SIZE) against $worker_url"
  (
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    # Generate batch requests via sglang.bench_serving to evaluate decode throughput.
    WORKER_URL="$worker_url" \
    BATCH_SIZE_ENV="$BATCH_SIZE" \
    INPUT_LEN_ENV="$BENCH_INPUT_LEN" \
    OUTPUT_LEN_ENV="$BENCH_OUTPUT_LEN" \
    DATASET_PATH_ENV="$BENCH_DATASET_PATH" \
    TOKENIZER_PATH_ENV="$MODEL_PATH" \
    RESULT_JSON_ENV="$result_file" \
    python3 - <<'PY'
import json
import os
import time

import requests

from sglang.bench_serving import get_tokenizer, sample_random_requests

worker_url = os.environ["WORKER_URL"].rstrip("/")
batch_size = int(os.environ["BATCH_SIZE_ENV"])
input_len = int(os.environ["INPUT_LEN_ENV"])
output_len = int(os.environ["OUTPUT_LEN_ENV"])
dataset_path = os.environ["DATASET_PATH_ENV"]
tokenizer_path = os.environ["TOKENIZER_PATH_ENV"]
result_path = os.environ["RESULT_JSON_ENV"]

tokenizer = get_tokenizer(tokenizer_path)
requests_batch = sample_random_requests(
    input_len=input_len,
    output_len=output_len,
    num_prompts=batch_size,
    range_ratio=1.0,
    tokenizer=tokenizer,
    dataset_path=dataset_path,
    random_sample=True,
    return_text=False,
)

payload = {
    "input_ids": [req.prompt for req in requests_batch],
    "sampling_params": {
        "temperature": 0.0,
        "max_new_tokens": output_len,
        "ignore_eos": True,
        "stream_interval": output_len,
    },
    "return_logprob": False,
    "stream": False,
}

url = worker_url + "/generate"
start = time.time()
response = requests.post(url, json=payload, timeout=600)
elapsed = time.time() - start
response.raise_for_status()
data = response.json()

with open(result_path, "w") as f:
    json.dump({"elapsed": elapsed, "response": data}, f)

print(f"Completed batch request to {worker_url} in {elapsed:.3f}s, status={response.status_code}")
PY
  ) > "$LOG_DIR/${LOG_TAG}_worker${idx}_bench_${BATCH_TIMESTAMP}.log" 2>&1 &
  BATCH_PIDS+=("$!")
done

# Wait for all batch benchmarks to finish
for pid in "${BATCH_PIDS[@]}"; do
  wait "$pid"
done

wait

# Notes (comparison with the reference):
# - The reference constructs a single multi-node server via two srun commands and sets --nccl-init-addr/--nnodes/--node-rank.
# - This script uses a router + multiple workers, leveraging the router for cross-node DP while each worker performs local TP.
#   When GPUs per node are partially used (e.g., 4/8), this approach is more robust for resource isolation, visible-device assignment, and avoiding cache conflicts.
# - If you prefer the "no-router" direct server aggregation style, replace the worker launch above with:
#   srun ... python3 -m sglang.launch_server --model-path "$MODEL_PATH" --tp "$TP_SIZE" \
#       --nccl-init-addr "${NODE_LIST[0]}:${DIST_PORT}" --nnodes "$NUM_WORKERS" --node-rank "$idx"
#   and remove the router logic. Note: on MI210 with dual NUMA, you should still set HIP_VISIBLE_DEVICES per node and isolate cache directories.