# WCM

### A World Critic Model for Vision-Language-Action Reinforcement Learning

Official repository of WCM, A World Critic Model for Vision-Language-Action Reinforcement Learning.

<div align="center">
  <a href="https://github.com/sylvestf/LIBERO-plus">📄 <strong>Paper</strong></a>
  &nbsp;|&nbsp;
  <a href="https://github.com/sylvestf/LIBERO-plus">💾 <strong>Checkpoints &amp; Data</strong></a>
  &nbsp;|&nbsp;
  <a href="https://github.com/sylvestf/LIBERO-plus">🌐 <strong>Website</strong></a>
</div>

WCM is a history-aware critic for partially observable robot control. It jointly learns to estimate the value
of the current state and to predict the next latent state, giving VLA reinforcement learning a representation
that is trained to capture dynamics instead of only fitting scalar returns.

<table align="center" style="border: none; width: 100%;">
  <tr>
    <td align="center" colspan="3" style="border: none; padding: 0 0 12px 0; font-size: 1.1em; font-weight: 500; color:rgb(234, 238, 243);">
      WCM trained on 100 real-world stovetop organization episodes.
    </td>
  </tr>
  <tr>
    <td align="center" style="border: none; padding: 10px; width: 33%;">
      <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <img src="assets/value_suc_00.gif" width="100%" alt="Success">
        <p style="margin: 8px 0 0 0; color: #22863a;"><strong>✅ Success</strong></p>
      </div>
    </td>
    <td align="center" style="border: none; padding: 10px; width: 33%;">
      <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <img src="assets/value_fail_01.gif" width="100%" alt="Failure 1">
        <p style="margin: 8px 0 0 0; color: #d73a49;"><strong>❌ Fail: Object Dropped</strong></p>
      </div>
    </td>
    <td align="center" style="border: none; padding: 10px; width: 33%;">
      <div style="border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <img src="assets/value_fail_02.gif" width="100%" alt="Failure 2">
        <p style="margin: 8px 0 0 0; color: #d73a49;"><strong>❌ Fail: Random Motion</strong></p>
      </div>
    </td>
  </tr>
</table>

## Why WCM?

Robot manipulation is a partially observable problem: one frame can hide motion, contact, and future outcomes.
WCM addresses this representation bottleneck with a lightweight LeJEPA-style architecture:

```text
observation history + language
              |
        encoder / history predictor
          /                    \
   value head V_t        dynamics head z_(t+1)
```

The training objective is

```text
L = L_value + lambda * L_prediction + eta * L_SIGReg.
```

The value head estimates a state value from history and language. The dynamics head is action-conditioned and
predicts the next latent state. This separates action-free value estimation from action-conditioned prediction,
while allowing both objectives to improve the shared representation.

## Highlights

- History-aware value estimation for vision-language-action reinforcement learning.
- Joint value learning, next-latent prediction, and SIGReg regularization.
- Compatible with on-policy PPO / Flow-SDE and off-policy AWR / RECAP pipelines.
- Evaluated with `pi_0`, `pi_0.5`, and OpenVLA-OFT across 149 simulation tasks and 7 real-world tasks.
- No additional critic latency is required during policy-only deployment; WCM is used during RL training and
  evaluation.

## Installation

The runnable implementation targets Python 3.12 or 3.13 and pins the tested stack around PyTorch 2.7,
TorchVision 0.22, LeRobot 0.5, Transformers 5, and TorchCodec 0.5.

Using `uv`:

```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e ".[all]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The first run downloads the configured ViT and CLIP checkpoints from Hugging Face unless they are already in
the local cache. If `uv` is not available, the final command can be replaced with
`python -m pip install -e ".[all]"` inside a Python 3.12/3.13 environment.

## Data

Training expects a LeRobot v3 dataset with task metadata and episode boundaries. The default configuration uses
the following fields:

| Field | Required | Description |
| --- | --- | --- |
| `observation.images.*` | Yes | One or more camera streams; the example config uses `observation.images.front`. |
| `action` | Yes | Action vector used by the dynamics head. |
| `return` | Yes | Scalar value target for each frame. |
| `episode_index`, `frame_index`, task metadata | Yes | Sequence and task metadata used for episode-safe windows and splits. |
| `observation.state` | Optional | Auxiliary proprioceptive input/target. |

Natural-language instructions are resolved from the dataset task metadata. History windows never cross an
episode boundary, and train/validation splitting is performed by episode id. Actions are standardized using
statistics fitted on the training episodes only.

If a dataset does not yet contain `return`, prepare a private output copy with the return-conversion script in
the source checkout:

```bash
bash 1_add_returns.sh
```

When the source has no success feature, provide an explicit `--success-labels` JSON mapping. The converter does
not guess success labels.

## Quick start

The shortest path is a one-GPU run on a prepared LeRobot dataset. First edit
`configs/train_8gpu.yaml` if your camera key, language/task fields, or model settings differ from the example.
Then launch training with runtime overrides:

```bash
bash 2_run_train.sh
```

When training finishes, the best checkpoint is written to `outputs/wcm/checkpoints/best.pt`. Evaluate it with:

```bash
bash 3_run_eval.sh
```

The main scalar metrics are written to `outputs/wcm/eval/summary.json`. Episode-level evaluation additionally
writes JSON/CSV curves and PNG plots under `outputs/wcm/eval/episode_curves/`.

For an eight-GPU run, set the dataset variables and `GPUS=8` in `2_run_train.sh`, then run:

```bash
bash 2_run_train.sh
```

The launcher selects `python` for one GPU and `torchrun` for eight GPUs. It fails early when the requested CUDA
devices are not visible; CPU/Gloo mode is intended only for an explicit smoke test.

## Outputs and checkpoints

Training writes the following artifacts under the configured output directory:

```text
outputs/<run>/
├── resolved_config.json
├── episode_split.json
├── metrics.jsonl
├── checkpoints/
│   ├── best.pt
│   ├── epoch-XXXX.pt
│   └── last.pt
└── deploy.pt
```

`last.pt`, `best.pt`, and `epoch-XXXX.pt` contain full resume state. `deploy.pt` is the compact model/config
bundle intended for inference and offline evaluation.

## Reproducing the paper

The paper evaluates WCM in:

- **Simulation:** ManiSkill (in-distribution and out-of-distribution), MetaWorld, CALVIN, and LIBERO-Plus.
- **Real world:** seven manipulation tasks on a WidowX-250S, including dynamic grasping, deformable-object
  manipulation, long-horizon cleaning, and pick-and-place.

See the [paper]() for complete baselines, per-task results, ablations, and experimental details.


## Citation

The current paper PDF is an anonymized CoRL submission. Replace the author field with the final author list
when the camera-ready version is available.

```bibtex
@inproceedings{wcm2026,
  title     = {WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning},
  author    = {},
  booktitle = {},
  year      = {2026}
}
```
