#!/usr/bin/env bash
set -euo pipefail

# Edit only this section, then run:  bash run_eval.sh
GPUS=1 #8                                  # 1 or 8
CUDA_VISIBLE_DEVICES=0 #"0,1,2,3,4,5,6,7"
STABLEWM_HOME="logs/home"                      # optional legacy cluster env; not required by LeRobot
LOCAL_DATASET_DIR="logs/home"                  # optional alias for DATASET_ROOT
CHECKPOINT="outputs/wcm_v2/checkpoints/best.pt"
OUTPUT_DIR="outputs/wcm_v2/eval_debug"
SPLIT="val"                             # train | val | all
BATCH_SIZE=16 #64
NUM_WORKERS=2 #8
PLOT_EPISODE_CURVES=1                   # 1 = write JSON/CSV/PNG curves
CURVE_OUTPUT_DIR=""                     # empty = ${OUTPUT_DIR}/episode_curves
MAX_BATCHES=""                         # empty = evaluate the complete split
MAX_CURVE_EPISODES=""                  # empty = write every episode curve; metrics stay full-split
LOG_EVERY_BATCHES=20                    # unbuffered fetch/forward diagnostics; set to 1 for debugging

# These values must identify the same dataset used by the checkpoint.  They
# are intentionally editable here so evaluation needs no command-line flags.
DATASET_REPO_ID="lerobot_with_return_val"                         # empty = use the checkpoint's dataset id
DATASET_ROOT="/path/to/lerobot_with_return_val"     
DATASET_REVISION=""
PRECISION=""                            # empty = checkpoint value
WANDB_MODE="offline"
ALLOW_CPU_SMOKE=0                       # 1 only for an intentional Gloo smoke test
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$GPUS" != "1" && "$GPUS" != "8" ]]; then
  echo "GPUS must be 1 or 8 (got: $GPUS)" >&2
  exit 2
fi
if [[ "$SPLIT" != "train" && "$SPLIT" != "val" && "$SPLIT" != "all" ]]; then
  echo "SPLIT must be train, val, or all (got: $SPLIT)" >&2
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
export WCM_DATASET_REPO_ID="$DATASET_REPO_ID"
export WCM_EXPECTED_WORLD_SIZE="$GPUS"

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
set_optional_env WCM_PRECISION "$PRECISION"

if [[ "$ALLOW_CPU_SMOKE" == "1" ]]; then
  export WCM_ALLOW_CPU_DDP=1
  export WCM_FORCE_CPU=1
else
  unset WCM_ALLOW_CPU_DDP WCM_FORCE_CPU 2>/dev/null || true
  if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'; then
    echo "GPUS=1 requires at least one visible CUDA device. Set ALLOW_CPU_SMOKE=1 only for an intentional CPU smoke test." >&2
    exit 2
  fi
  if [[ "$GPUS" == "8" ]]; then
    if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() >= 8 else 1)'; then
      echo "GPUS=8 requires at least eight visible CUDA devices. Set ALLOW_CPU_SMOKE=1 only for a CPU smoke test." >&2
      exit 2
    fi
  fi
fi

CURVE_ARGS=(--no-episode-curves)
if [[ "$PLOT_EPISODE_CURVES" == "1" ]]; then
  CURVE_ARGS=(--episode-curves)
fi
OPTIONAL_ARGS=()
if [[ -n "$CURVE_OUTPUT_DIR" ]]; then
  OPTIONAL_ARGS+=(--curve-output-dir "$CURVE_OUTPUT_DIR")
fi
if [[ -n "$MAX_BATCHES" ]]; then
  OPTIONAL_ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [[ -n "$MAX_CURVE_EPISODES" ]]; then
  OPTIONAL_ARGS+=(--max-curve-episodes "$MAX_CURVE_EPISODES")
fi
OPTIONAL_ARGS+=(--log-every-batches "$LOG_EVERY_BATCHES")
EVAL_ARGS=(
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --expected-world-size "$GPUS"
  "${CURVE_ARGS[@]}"
  "${OPTIONAL_ARGS[@]}"
)

if [[ "$GPUS" == "1" ]]; then
  python -u -m world_critic.evaluate "${EVAL_ARGS[@]}"
else
  torchrun --standalone --nproc-per-node=8 \
    -m world_critic.evaluate "${EVAL_ARGS[@]}"
fi
