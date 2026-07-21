#!/usr/bin/env bash
set -euo pipefail

# Edit only this section, then run:  bash run_train.sh
GPUS=1 #8                                  # 1 or 8
CUDA_VISIBLE_DEVICES=0 #"0,1,2,3,4,5,6,7"
STABLEWM_HOME="logs/home"                      # optional legacy cluster env; not required by LeRobot
LOCAL_DATASET_DIR="logs/home"                  # optional alias for DATASET_ROOT
CONFIG="configs/train_8gpu.yaml"
DATASET_REPO_ID="lerobot_with_return"                         # required unless CONFIG already contains it
DATASET_ROOT="/path/to/lerobot_with_return"                              # empty = use the HF cache
DATASET_REVISION=""                            # optional
OUTPUT_DIR="outputs/wcm_v2"
EPOCHS=""                                      # empty = keep CONFIG value
PER_DEVICE_BATCH_SIZE=""                       # empty = keep CONFIG value
EVAL_BATCH_SIZE=""                             # empty = keep CONFIG value
NUM_WORKERS=""                                 # empty = keep CONFIG value
PRECISION=""                                   # fp32 or bf16; empty = CONFIG
RESUME=""                                      # optional full-resume checkpoint
WANDB_MODE="offline"
ALLOW_CPU_SMOKE=0                               # 1 only for an intentional Gloo smoke test
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$GPUS" != "1" && "$GPUS" != "8" ]]; then
  echo "GPUS must be 1 or 8 (got: $GPUS)" >&2
  exit 2
fi
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
if [[ -n "$STABLEWM_HOME" ]]; then export STABLEWM_HOME; else unset STABLEWM_HOME 2>/dev/null || true; fi
if [[ -n "$LOCAL_DATASET_DIR" ]]; then export LOCAL_DATASET_DIR; else unset LOCAL_DATASET_DIR 2>/dev/null || true; fi
if [[ -z "$DATASET_ROOT" && -n "$LOCAL_DATASET_DIR" ]]; then DATASET_ROOT="$LOCAL_DATASET_DIR"; fi
export WANDB_MODE
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WCM_DATASET_REPO_ID="$DATASET_REPO_ID"
export WCM_EXPECTED_WORLD_SIZE="$GPUS"
export WCM_OUTPUT_DIR="$OUTPUT_DIR"

set_optional_env() {
  local name="$1" value="$2"
  if [[ -n "$value" ]]; then
    export "$name=$value"
  else
    unset "$name" 2>/dev/null || true
  fi
}

set_optional_env WCM_DATASET_ROOT "$DATASET_ROOT"
set_optional_env WCM_DATASET_REVISION "$DATASET_REVISION"
set_optional_env WCM_EPOCHS "$EPOCHS"
set_optional_env WCM_PER_DEVICE_BATCH_SIZE "$PER_DEVICE_BATCH_SIZE"
set_optional_env WCM_EVAL_BATCH_SIZE "$EVAL_BATCH_SIZE"
set_optional_env WCM_NUM_WORKERS "$NUM_WORKERS"
set_optional_env WCM_PRECISION "$PRECISION"
set_optional_env WCM_RESUME "$RESUME"

if [[ "$ALLOW_CPU_SMOKE" == "1" ]]; then
  export WCM_ALLOW_CPU_DDP=1
  export WCM_FORCE_CPU=1
else
  unset WCM_ALLOW_CPU_DDP WCM_FORCE_CPU 2>/dev/null || true
  if [[ "$GPUS" == "8" ]]; then
    if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() >= 8 else 1)'; then
      echo "GPUS=8 requires at least eight visible CUDA devices. Set ALLOW_CPU_SMOKE=1 only for a CPU smoke test." >&2
      exit 2
    fi
  fi
fi

if [[ "$GPUS" == "1" ]]; then
  python -m world_critic.train --config "$CONFIG"
else
  torchrun --standalone --nproc-per-node=8 \
    -m world_critic.train --config "$CONFIG"
fi
