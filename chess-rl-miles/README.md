# Chess RL on Miles

Multi-turn chess reinforcement learning with Miles FSDP training and SGLang
rollouts. The package handles `<call_env>` replies, move scoring, loss masks,
dataset checks, and rollout logs. CISPO is enabled by default
with clipping bounds `[0, 5]` and GRPO advantages.

## Layout

- `chess_rl_miles/`: rollout, reward, data-source, tokenizer, and logging code.
- `chess_rl_miles/scripts/`: training command builder.
- `miles/`: bundled, modified Miles runtime and checkpoint conversion tools.
- `config/chess_multiturn.yaml`: chess environment defaults.
- `tests/`: local regression tests, including an independent offline reward reference.
- `sitecustomize.py`: SGLang detokenizer compatibility hook, loaded through `PYTHONPATH`.

## Setup and local tests

Use Python 3.11 or newer. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export PYTHONPATH=".:miles${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests -q
```

The test dependencies support local CPU checks. Full training requires Linux,
NVIDIA GPUs, and the SGLang/CUDA dependencies supplied by the pinned Miles image (see the top-level README).
Use the bundled runtime; a stock Miles installation lacks the required changes.

## Inspect a training command

For existing local data and an HF-format model directory:

```bash
python -m chess_rl_miles.scripts.run_chess_miles \
  --hf-checkpoint /path/to/model \
  --train-file /path/to/train.parquet \
  --train-file-sha256 EXPECTED_SHA256 \
  --eval-file /path/to/eval.parquet \
  --save-dir ./outputs \
  --run-name chess_miles_trial_01 \
  --hparam-tag-suffix trial_01 \
  --dry-run
```

`--dry-run` hashes the training file and prints the command without starting
training. It creates output directories. It does not validate the model or start
SGLang. `--prepare-data` and `--prepare-sft` can download files even in a dry run.
The launcher accepts `--miles-dir` for an explicit runtime location.

Default data comes from `chess-pre-to-post/chess-rl-data`; SFT models come from
`chess-pre-to-post/sft_trajectory_no_labels`. Access may require an HF token.
The default training filename and expected SHA-256 are in `chess_rl_miles/data.py`.
Custom data must preserve the prompt, reward target, and `extra_info` schema,
including environment replies and puzzle FEN/setup-move metadata used by scoring.

## Source comparison and cleanup

This import keeps the standalone `chess-rl-miles` project scope. Its reusable
code was compared with `interleave-pt-rl/chess/rl`, and the later versions were
retained where they contain the following fixes:

| Area | Difference from the standalone copy |
| --- | --- |
| Moves and rewards | Apply the puzzle setup move before legality checks; use the correct side for castling. |
| Prompts | Require exactly one leading BOS and enforce the prompt limit after BOS insertion. |
| Context | Remove the implicit 16-token margin; support explicit matching SGLang and training limits. |
| FSDP | Preserve FP32 master parameters, Adam state, and checkpoints while using BF16 computation. |
| Loss | Normalize token-mean loss by supervised tokens across the complete optimizer update. |
| Checkpoints | Authenticate committed payloads and restore RNG/data state; reject incompatible resume metadata. |
| Packaging | Bundle the matching Miles runtime and exclude local credentials/caches from Modal source mounts. |

Optional newer controls, including explicit KL estimators, fixed W&B run IDs,
and authenticated initial Adam-state loading, remain available. They do not
implicitly enable optimizer continuation.

The imported source is a local working-tree snapshot from 2026-09-07. The newer
source repository HEAD was `ab1ba9b`, with additional uncommitted changes.
See `miles/README.md` for the upstream base and modified runtime files.

Old run summaries, benchmark claims, versioned experiment/gate launchers,
contract-binding files, dashboards, upload monitors, destructive cleanup tools,
recovery records, caches, data, and checkpoints were omitted. The README was
rewritten around setup and supported entry points. Original workspace files
were preserved.

Local tests cover adapters, command construction, reward agreement, precision,
checkpoint metadata, and loss normalization. This import has not been validated
with a GPU training run.
