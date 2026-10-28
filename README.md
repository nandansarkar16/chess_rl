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
  one folder per pre-training budget (`2.1B/`, `11B/`, `20B/`, `40B/` tokens).
  We'll tell you which to run.

## Setup

```bash
hf download nandansarkar/chess-e2h-rl --repo-type dataset --local-dir /data/rl
hf download nandansarkar/chess-100m-sft-elo800-900 --local-dir /data/sft

# Docker
docker run --gpus all --ipc=host -v $PWD:/workspace -v /data:/data -it \
  radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e
pip install python-chess

# or Apptainer
apptainer pull miles.sif docker://radixark/miles@sha256:5b41bff2ecd42f1e71b5d8658e777541a821ef96556ae06b48333d521e0ca25e
pip install --target ./pydeps python-chess
# then prefix commands with: apptainer exec --nv --env PYTHONPATH=$PWD/pydeps miles.sif
```

## Running

Everything below runs inside the container.

```bash
export SFT_CKPT=/data/sft/11B             # which SFT checkpoint
export MODEL_ID=100m_11B                  # name used in output paths
export DATA_DIR=/data/rl/900_easy17k_1200_hard
export OUTPUT_DIR=/data/outputs

ARM=all   bash run_rl.sh                  # one arm per GPU / job
ARM=easy  bash run_rl.sh
ARM=mixed bash run_rl.sh
ARM=hard  bash run_rl.sh
```

Add `--dry-run` to print the training command without starting it. If a job
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
ARM=mixed HF_REPO=<your-username>/<repo-name> bash export_hf.sh   # same env vars as above
```

That uploads `global_step_100` … `global_step_2000` (~400 MB each) to
`<MODEL_ID>/<arm>_<reward>_g16/` in your repo, created private. Share it with
us when done. The raw checkpoints under `OUTPUT_DIR` (~1.2 GB each, including
optimizer state) are only needed for resuming and can be deleted afterwards.

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
