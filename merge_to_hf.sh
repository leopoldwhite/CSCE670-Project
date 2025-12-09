#!/usr/bin/env bash
set -euo pipefail

# Merge a VERL FSDP actor checkpoint into a Hugging Face format directory.
#
# Usage:
#   ./merge_to_hf.sh [LOCAL_DIR] [TARGET_DIR]
#
# Examples:
#   ./merge_to_hf.sh \
#     /home/ybai/working/refRepos/verl/verl_checkpoints/ppo-format-e2h-gaussian-beta0.4-sigma0.75-human_initiative-200steps/global_step_200/actor \
#     models/merged_hf/ppo-format-e2h-gs200
#
# Env knobs:
#   TRUST_REMOTE_CODE=1      # pass --trust-remote-code
#   USE_CPU_INIT=1           # pass --use_cpu_initialization (for large models)

DEFAULT_LOCAL_DIR=".verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-it-em-sft-Iterative-SFT/actor/global_step_600"
DEFAULT_TARGET_DIR="models/nq-search-r1-ppo-qwen2.5-3b-it-em-sft-Iterative-SFT"

LOCAL_DIR=${1:-$DEFAULT_LOCAL_DIR}
TARGET_DIR=${2:-$DEFAULT_TARGET_DIR}

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "[Error] LOCAL_DIR does not exist: $LOCAL_DIR" >&2
  exit 1
fi

# The merger expects huggingface config under $LOCAL_DIR/huggingface
if [[ ! -d "$LOCAL_DIR/huggingface" ]]; then
  echo "[Warn] $LOCAL_DIR/huggingface not found. Ensure checkpoint contains huggingface config." >&2
fi

mkdir -p "$(dirname "$TARGET_DIR")"

EXTRA_ARGS=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--trust-remote-code")
fi
if [[ "${USE_CPU_INIT:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--use_cpu_initialization")
fi

echo "Merging FSDP checkpoint → HF format"
echo "  local_dir : $LOCAL_DIR"
echo "  target_dir: $TARGET_DIR"

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR" \
  ${EXTRA_ARGS[@]:-}

echo "Done. You can load it via:"
echo "  from transformers import AutoTokenizer, AutoModelForCausalLM" \
     "\n  tok = AutoTokenizer.from_pretrained('$TARGET_DIR', trust_remote_code=True)" \
     "\n  model = AutoModelForCausalLM.from_pretrained('$TARGET_DIR', torch_dtype='auto', trust_remote_code=True)"

