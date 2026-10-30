# Chess easy-to-hard RL

This repo runs the RL stage of our chess easy-to-hard experiment. The question
we're studying: if you only have verified labels for **easy** problems, how
close can you get to training on labeled **hard** problems too, using the
model's own majority vote as the reward on the hard ones?

The model is a 100M-parameter chess transformer, pre-trained on human games and
then SFT'd on chain-of-thought solutions to 800–900 ELO puzzles. You start from
one of those SFT checkpoints and run four RL arms that differ only in their
training data and reward:

| `ARM` | Trains on | Reward |
|---|---|---|
| `all` | easy + hard | ground truth on both (upper reference) |
| `easy` | easy only | ground truth (lower reference) |
| `mixed` | easy + hard | ground truth on easy, majority vote on hard |
| `hard` | hard only | majority vote |

Easy = puzzle ELO 800–900, hard = 900–1200, ~17k puzzles each. "Majority
vote" means the most common answer across the model's own samples for a puzzle
is treated as correct; the ground truth is never seen.

Training is GRPO on [Miles](chess-rl-miles/miles) (FSDP training, SGLang
rollouts): 64 puzzles × 16 samples per step, 2,000 steps, checkpoint every 100.
For the majority-vote rows, the vote is taken over the same 16 samples.

## What you need

- **GPUs:** one per arm. The four arms are independent, so run them in
  parallel. On one L40S an arm takes roughly 2–3 days; H100s are faster.
- **Container:** `radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e`,
  plus `python-chess`. Use the bundled `chess-rl-miles/miles/`, not upstream
  Miles; it has required changes.
- **Data:** [`nandansarkar/chess-e2h-rl`](https://huggingface.co/datasets/nandansarkar/chess-e2h-rl)
- **SFT checkpoints:** [`nandansarkar/chess-100m-sft-elo800-900`](https://huggingface.co/nandansarkar/chess-100m-sft-elo800-900),
  one folder per pre-training budget (`0.53B/`, `1.1B/`, `2.1B/`, `11B/`, `20B/`,
  `40B/` tokens).
  We'll tell you which to run.

## Setup

We run everything with Apptainer on a Slurm cluster. One-time setup:

```bash
git clone https://github.com/nandansarkar16/chess_rl.git && cd chess_rl

# Data and SFT checkpoints
hf download nandansarkar/chess-e2h-rl --repo-type dataset --local-dir /data/rl
hf download nandansarkar/chess-100m-sft-elo800-900 --local-dir /data/sft

# Container image (~13 GB; put it somewhere with space)
apptainer pull miles.sif docker://radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e

# The image doesn't ship python-chess. Install it into a folder from inside
# the image so it matches the image's Python, then add it to PYTHONPATH later.
apptainer exec miles.sif pip install --target ./pydeps python-chess
```

Docker works too: run the same image with `--gpus all --ipc=host`, mount the
repo and data, and `pip install python-chess` inside it.

## Running

Each arm is one job on one GPU. This is how we launch an arm on Slurm (adjust
the `#SBATCH` lines for your cluster):

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
set -eo pipefail

# Conda can leak compiler variables that break Triton inside the container.
unset CC CXX CC_FOR_BUILD CPP GCC

export ARM=mixed                          # all | easy | mixed | hard
export SFT_CKPT=/data/sft/11B             # which SFT checkpoint
export MODEL_ID=100m_11B                  # name used in output paths
export DATA_DIR=/data/rl/900_easy17k_1200_hard
export OUTPUT_DIR=/data/outputs

cd /path/to/chess_rl
apptainer exec --nv \
    --bind /data \
    --env PYTHONPATH=$PWD/pydeps \
    /path/to/miles.sif \
    bash run_rl.sh
```

Notes:
- Apptainer passes your exported variables into the container. `--bind` any
  directory the job reads or writes that your cluster doesn't mount by default
  (data, outputs, the repo if it's outside your home).
- Submit four of these, one per `ARM`. They don't depend on each other.

Add `--dry-run` after `run_rl.sh` to print the training command without starting it. If a job
dies, rerun the same command with `RESUME=1` and it picks up from the last
checkpoint.

Optional settings:

| Variable | Default | |
|---|---|---|
| `NUM_GPUS` | 1 | GPUs per arm |
| `NUM_STEPS` | 2000 | |
| `SAVE_INTERVAL` | 100 | keep at 100; we evaluate every 100 steps |
| `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY` | off | set the key to log to W&B |
| `LEGACY_32_8` | 0 | `1` = our older setup (8 samples trained, vote over 32); don't use unless asked |

FlashAttention 3 is the default and needs H100/H200. On other GPUs (L40S,
A100) add `--attn-implementation flash_attention_2` after `run_rl.sh`. Any
other flag after `run_rl.sh` goes to the launcher
(`python -m chess_rl_miles.scripts.run_chess_miles --help`).

## Sending results back

We run evaluation ourselves and only need the model weights. When an arm
finishes, convert its checkpoints to Hugging Face format and upload them to a
repo in your account:

```bash
hf auth login
# same env vars and apptainer command as the training job, but run export_hf.sh
export HF_REPO=<your-username>/<repo-name>
apptainer exec --nv --bind /data --env PYTHONPATH=$PWD/pydeps /path/to/miles.sif \
    bash export_hf.sh
```

That uploads `global_step_100` … `global_step_2000` (~400 MB each) to
`<MODEL_ID>/<arm>_<reward>_g16/` in your repo, created private. Share it with
us when done.

## Repo layout

```
run_rl.sh                 launch one arm
export_hf.sh              convert an arm's checkpoints to HF format and upload them
e2h/e2h_miles_ttrl.py     majority-vote reward for the mixed and hard arms
e2h/chess_mv_16.yaml      config for mixed/hard (vote over 16)
e2h/chess_mv_32v8.yaml    legacy config (vote over 32, train on 8)
chess-rl-miles/           chess rollout/reward code + the bundled Miles runtime
```

The data files carry the reward routing: each row in `combined_mixed.parquet`
and `hard_mv.parquet` has `extra_info.reward_type` set to `rule_based` or
`majority_vote`, and `e2h_miles_ttrl.py` scores the `majority_vote` rows
against the group's majority answer instead of the label.

Outputs land in `$OUTPUT_DIR/<MODEL_ID>/<arm>_<reward>_g16/`, with
checkpoints under `.../checkpoints/iter_*`, plus `logs/` and sampled
`rollouts/`.
