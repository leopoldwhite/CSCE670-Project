#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5
# Default locations (override via env vars if needed)
REPO_DIR="your_repo_path"
MODEL_ID="${MODEL_ID:-PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-ppo}"
DATASET="${DATASET:-$REPO_DIR/data/nq_search/test.parquet}"
#OUTPUT_JSON="${OUTPUT_JSON:-$REPO_DIR/outputs/nq_eval_iterative_sft_RL-600steps_results.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_DIR/outputs/nq_eval_searchr1_results.json}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVAL_TOPK="${RETRIEVAL_TOPK:-3}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-60}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_TURNS="${MAX_TURNS:-5}"
TENSOR_PARALLEL_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

cd "$REPO_DIR"

CMD=(python infer.py
    --model-id "$MODEL_ID"
    --dataset "$DATASET"
    --output-json "$OUTPUT_JSON"
    --retrieval-url "$RETRIEVAL_URL"
    --retrieval-topk "$RETRIEVAL_TOPK"
    --retrieval-timeout "$RETRIEVAL_TIMEOUT"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --max-turns "$MAX_TURNS"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
fi

# Forward any extra CLI arguments so the user can override defaults.
CMD+=("$@")

echo "Running: ${CMD[*]}"
"${CMD[@]}"

