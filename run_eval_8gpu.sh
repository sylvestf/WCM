#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this launcher so it is safe to invoke it from
# either the repository root or another working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Match the training launcher: fail before torchrun if this is not actually
# an eight-GPU environment.  ``WCM_ALLOW_CPU_DDP=1`` is reserved for an
# intentional CPU/Gloo smoke test.
if [[ "${WCM_ALLOW_CPU_DDP:-0}" != "1" ]]; then
  if ! python -c 'import torch,sys; sys.exit(0 if torch.cuda.device_count() >= 8 else 1)'; then
    echo "run_eval_8gpu.sh requires at least 8 visible CUDA devices. Set WCM_ALLOW_CPU_DDP=1 only for a CPU/Gloo smoke test." >&2
    exit 2
  fi
else
  # Explicitly requested CPU/Gloo smoke mode; see the matching training
  # launcher for why this is needed even when CUDA is installed locally.
  export WCM_FORCE_CPU=1
fi

torchrun \
  --standalone \
  --nproc-per-node=8 \
  -m world_critic.evaluate \
  --checkpoint outputs/wcm_v2/deploy.pt \
  --output-dir outputs/wcm_v2/eval \
  --split val \
  --expected-world-size 8
