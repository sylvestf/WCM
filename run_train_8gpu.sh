#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this launcher so it is safe to invoke it from
# either the repository root or another working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# This entry point promises an eight-GPU run.  Keep an explicit opt-out for
# CPU/Gloo smoke tests, but do not silently launch a nominal "8-GPU" job with
# fewer visible CUDA devices (which would either fail later or produce a
# misleading experiment).
if [[ "${WCM_ALLOW_CPU_DDP:-0}" != "1" ]]; then
  if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() >= 8 else 1)'; then
    echo "run_train_8gpu.sh requires at least 8 visible CUDA devices. Set WCM_ALLOW_CPU_DDP=1 only for a CPU/Gloo smoke test." >&2
    exit 2
  fi
else
  # Explicitly requested CPU/Gloo smoke mode.  This must override CUDA
  # auto-detection in initialize_distributed when a workstation has one or
  # more visible GPUs but fewer than the eight production devices.
  export WCM_FORCE_CPU=1
fi

torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m world_critic.train \
  --config configs/train_8gpu.yaml
