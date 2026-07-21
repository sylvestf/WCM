# WCM

### A World Critic Model for Vision-Language-Action Reinforcement Learning

Official repository of WCM, A World Critic Model for Vision-Language-Action Reinforcement Learning.

[Paper (PDF)]() · [Method note]()

WCM is a history-aware critic for partially observable robot control. It jointly learns to estimate the value
of the current state and to predict the next latent state, giving VLA reinforcement learning a representation
that is trained to capture dynamics instead of only fitting scalar returns.

<div style="display: flex; gap: 10px; justify-content: center;">
  <video width="30%" controls autoplay muted loop>
    <source src="assets/value_suc_00.mp4" type="video/mp4">
  </video>
  <video width="30%" controls autoplay muted loop>
    <source src="assets/value_fail_01.mp4" type="video/mp4">
  </video>
  <video width="30%" controls autoplay muted loop>
    <source src="assets/value_fail_02.mp4" type="video/mp4">
  </video>
</div>

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

## Results

The following headline numbers are selected from Tables 1-3 of the paper. The reference column names the
strongest relevant baseline reported for that benchmark/setting. LIBERO-Plus compares one-demonstration SFT
with WCM initialized from the same model.

| Benchmark / metric | Backbone | Reference baseline | WCM (ours) |
| --- | --- | ---: | ---: |
| ManiSkill IND average | `pi_0` | 79.2 (pi-stepNFT) | **84.4 +/- 1.2** |
| ManiSkill IND average | `pi_0.5` | 90.9 (Flow-SDE) | **91.9 +/- 0.4** |
| ManiSkill IND average | OpenVLA-OFT | 97.7 (PPO) | **99.0 +/- 0.4** |
| ManiSkill OOD average | `pi_0` | 50.4 (pi-stepNFT) | **51.5 +/- 1.5** |
| ManiSkill OOD average | `pi_0.5` | 59.5 (pi-stepNFT) | **64.4 +/- 1.4** |
| ManiSkill OOD average | OpenVLA-OFT | 77.1 (PPO) | **77.9 +/- 0.8** |
| LIBERO-Plus total | `pi_0` | 39.1 +/- 2.1 (One-SFT) | **72.8 +/- 1.9** |
| LIBERO-Plus total | `pi_0.5` | 38.0 +/- 1.6 (One-SFT) | **73.7 +/- 1.4** |
| LIBERO-Plus total | OpenVLA-OFT | 29.3 +/- 1.5 (One-SFT) | **74.0 +/- 1.8** |

On the zero-shot OpenVLA-OFT ManiSkill setting, WCM improves the IND average from **0.8** to **98.7 +/- 0.3**.
In the real-world WidowX-250S evaluation, WCM improves every task over its corresponding SFT baseline across
7 tasks (50 trajectories per task in the reported slice); see Table 3 in the paper for per-task counts.

Aggregating the counts in Table 3 over the seven real-world tasks gives **199/350 (56.9%)** for OpenVLA-OFT
with WCM versus 167/350 (47.7%) with AWR, and **255/350 (72.9%)** for `pi_0.5` with WCM versus 220/350 (62.9%)
with RECAP.

For the two additional simulation benchmarks shown in Figure 4, WCM reaches 83.4% (`pi_0`) and 75.2% (`pi_0.5`)
on MetaWorld, and average completion lengths of 3.918 (`pi_0`) and 4.748 (`pi_0.5`) on CALVIN.

The ablations in the paper show that adding observation history alone is not enough: the world-prediction
objective is what makes the history representation useful. A history length of 3 performs best on average in the
reported experiments.

All numbers are success rates unless noted otherwise. Error bars are reported exactly as in the paper.

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
