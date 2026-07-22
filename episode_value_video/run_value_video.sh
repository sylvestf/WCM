#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Edit this section, then run from WCM_v2:
#   bash episode_value_video/run_value_video.sh
# ---------------------------------------------------------------------------
MODE="render"  # render = consume existing curves; pipeline = run eval first
CHECKPOINT="outputs/wcm_v2/checkpoints/deploy.pt"
EVAL_OUTPUT_DIR="outputs/wcm_v2/eval"
VIDEO_OUTPUT_DIR="outputs/wcm_v2/eval/episode_value_videos"
DATASET_ROOT=""       # empty = checkpoint value / environment override
CAMERA_KEY=""         # empty = first checkpoint data.image_keys entry
EPISODE_ID=""          # empty = all curves; otherwise one integer id
MAX_EPISODES=""        # empty = no limit
NPROC_PER_NODE=1
EVAL_BATCH_SIZE=64
EVAL_NUM_WORKERS=8
ACCENT="#61E4FF"
SPEED="1.0"            # 2.0 = 2x speed; 0.5 = half speed
OUTPUT_FPS=""          # optional fixed output FPS; when set, SPEED is ignored
SOURCE_FPS=""          # only override incorrect/missing dataset/video metadata
OVERWRITE=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --output-dir "$VIDEO_OUTPUT_DIR"
  --accent "$ACCENT"
)

if [[ -n "$OUTPUT_FPS" ]]; then
  COMMON_ARGS+=(--output-fps "$OUTPUT_FPS")
else
  COMMON_ARGS+=(--speed "$SPEED")
fi
if [[ -n "$SOURCE_FPS" ]]; then
  COMMON_ARGS+=(--source-fps "$SOURCE_FPS")
fi

if [[ -n "$DATASET_ROOT" ]]; then
  COMMON_ARGS+=(--dataset-root "$DATASET_ROOT")
fi
if [[ -n "$CAMERA_KEY" ]]; then
  COMMON_ARGS+=(--camera-key "$CAMERA_KEY")
fi
if [[ -n "$EPISODE_ID" ]]; then
  COMMON_ARGS+=(--episode-id "$EPISODE_ID")
fi
if [[ -n "$MAX_EPISODES" ]]; then
  COMMON_ARGS+=(--max-episodes "$MAX_EPISODES")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ "$MODE" == "render" ]]; then
  python -m episode_value_video render \
    --curves "$EVAL_OUTPUT_DIR/episode_curves/episode_curves.json" \
    "${COMMON_ARGS[@]}"
elif [[ "$MODE" == "pipeline" ]]; then
  python -m episode_value_video pipeline \
    --eval-output-dir "$EVAL_OUTPUT_DIR" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-num-workers "$EVAL_NUM_WORKERS" \
    "${COMMON_ARGS[@]}"
else
  echo "MODE must be render or pipeline (got: $MODE)" >&2
  exit 2
fi
