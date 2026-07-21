#!/usr/bin/env bash
set -euo pipefail

# A single, deliberately boring entry point for both training and offline
# evaluation.  Edit only the variables in this section, then run:
#
#     bash run_wcm.sh
#
# ``GPUS`` may be 1 or 8.  The launcher chooses ``python`` for one process and
# ``torchrun`` for eight processes; no extra command-line arguments are needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------- edit these -----------------------------
MODE="train"                         # train | eval
GPUS=1                                # 1 or 8
CUDA_VISIBLE_DEVICES="0"              # e.g. "0" or "0,1,2,3,4,5,6,7"
STABLEWM_HOME=""                      # optional compatibility env used by cluster scripts
LOCAL_DATASET_DIR=""                  # optional alias used when DATASET_ROOT is empty
WANDB_MODE="offline"

TRAIN_CONFIG="configs/train_8gpu.yaml"
DATASET_REPO_ID=""                         # empty = keep the YAML/checkpoint dataset id
DATASET_ROOT=""                      # local LeRobot root; empty keeps YAML value
DATASET_REVISION=""                  # optional Hub/local revision

OUTPUT_DIR="outputs/wcm_v2"
CHECKPOINT="${OUTPUT_DIR}/deploy.pt"
EVAL_OUTPUT_DIR="${OUTPUT_DIR}/eval"
EVAL_SPLIT="val"                     # train | val | all
PLOT_EPISODE_CURVES=1                # 1: write one PNG per episode, 0: disable
MAX_CURVE_EPISODES=""                # empty = plot every episode
CURVE_OUTPUT_DIR=""                 # empty: ${EVAL_OUTPUT_DIR}/episode_curves
MAX_BATCHES=""                      # empty: evaluate the complete split
LOG_EVERY_BATCHES=20                 # unbuffered fetch/forward diagnostics; set to 1 for debugging

# Common runtime knobs.  Leave empty to keep the value in TRAIN_CONFIG.
EPOCHS=""
PER_DEVICE_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
NUM_WORKERS=""
PRECISION=""
RESUME=""

# Set to 1 only for an intentional CPU/Gloo smoke test.  Production 8-GPU
# runs should leave this at 0; the launcher then fails early if eight GPUs are
# not visible instead of starting a misleading partial job.
ALLOW_CPU_SMOKE=0
# -----------------------------------------------------------------------

if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "MODE must be train or eval (got: $MODE)" >&2
  exit 2
fi
if [[ "$GPUS" != "1" && "$GPUS" != "8" ]]; then
  echo "GPUS must be 1 or 8 (got: $GPUS)" >&2
  exit 2
fi
if [[ "$EVAL_SPLIT" != "train" && "$EVAL_SPLIT" != "val" && "$EVAL_SPLIT" != "all" ]]; then
  echo "EVAL_SPLIT must be train, val, or all (got: $EVAL_SPLIT)" >&2
  exit 2
fi
if [[ "$PLOT_EPISODE_CURVES" != "0" && "$PLOT_EPISODE_CURVES" != "1" ]]; then
  echo "PLOT_EPISODE_CURVES must be 0 or 1 (got: $PLOT_EPISODE_CURVES)" >&2
  exit 2
fi

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES
fi
python -u -c 'import torch; device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"; print("[launcher] torch={} cuda={} cuda_available={} visible_devices={} device={}".format(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), device), flush=True)'
if [[ -n "$STABLEWM_HOME" ]]; then export STABLEWM_HOME; else unset STABLEWM_HOME 2>/dev/null || true; fi
if [[ -n "$LOCAL_DATASET_DIR" ]]; then export LOCAL_DATASET_DIR; else unset LOCAL_DATASET_DIR 2>/dev/null || true; fi
if [[ -z "$DATASET_ROOT" && -n "$LOCAL_DATASET_DIR" ]]; then DATASET_ROOT="$LOCAL_DATASET_DIR"; fi
export WANDB_MODE
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# These environment variables are consumed by world_critic.config without
# changing the checkpoint/config schema.  Empty optional values are unset so
# a config can still intentionally specify null.
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
  unset WCM_ALLOW_CPU_DDP 2>/dev/null || true
  if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
    echo "GPUS=1 requires at least one visible CUDA device. Set ALLOW_CPU_SMOKE=1 only for an intentional CPU smoke test." >&2
    exit 2
  fi
  unset WCM_FORCE_CPU 2>/dev/null || true
  if [[ "$GPUS" == "8" ]]; then
    if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() >= 8 else 1)'; then
      echo "GPUS=8 requires at least eight visible CUDA devices. Set ALLOW_CPU_SMOKE=1 only for a CPU smoke test." >&2
      exit 2
    fi
  fi
fi

if [[ "$MODE" == "train" ]]; then
  if [[ "$GPUS" == "1" ]]; then
    python -m world_critic.train --config "$TRAIN_CONFIG"
  else
    torchrun --standalone --nproc-per-node=8 \
      -m world_critic.train --config "$TRAIN_CONFIG"
  fi
else
  if [[ "$PLOT_EPISODE_CURVES" == "1" ]]; then
    CURVE_FLAG=(--episode-curves)
  else
    CURVE_FLAG=(--no-episode-curves)
  fi
  EVAL_ARGS=(
    --checkpoint "$CHECKPOINT"
    --output-dir "$EVAL_OUTPUT_DIR"
    --split "$EVAL_SPLIT"
    --batch-size "${EVAL_BATCH_SIZE:-64}"
    --num-workers "${NUM_WORKERS:-8}"
    --expected-world-size "$GPUS"
    "${CURVE_FLAG[@]}"
  )
  if [[ -n "$CURVE_OUTPUT_DIR" ]]; then
    EVAL_ARGS+=(--curve-output-dir "$CURVE_OUTPUT_DIR")
  fi
  if [[ -n "$MAX_BATCHES" ]]; then
    EVAL_ARGS+=(--max-batches "$MAX_BATCHES")
  fi
  if [[ -n "$MAX_CURVE_EPISODES" ]]; then
    EVAL_ARGS+=(--max-curve-episodes "$MAX_CURVE_EPISODES")
  fi
  EVAL_ARGS+=(--log-every-batches "$LOG_EVERY_BATCHES")
  if [[ "$GPUS" == "1" ]]; then
    python -u -m world_critic.evaluate "${EVAL_ARGS[@]}"
  else
    torchrun --standalone --nproc-per-node=8 \
      -m world_critic.evaluate "${EVAL_ARGS[@]}"
  fi
fi
