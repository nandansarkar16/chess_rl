# Chess RL on Miles

## Scope and layout

- `chess_rl_miles/` contains chess rollout, reward, data, and logging adapters.
- `chess_rl_miles/scripts/run_chess_miles.py` builds the training command.
- `miles/` contains the modified Miles runtime, including FSDP and SGLang code.
  It is part of this project; do not replace it with an upstream checkout or
  apply an old patch over it. Retain its license and modification notices.
- `tests/` contains local regression tests. `tests/fixtures/offline_reward.py`
  is an independent scoring reference for online/offline reward agreement.
- Keep experiment-specific launchers, run ledgers, dashboards, checkpoints,
  datasets, credentials, caches, and recovery records out of this folder.

## Development

- Run commands from this folder. Follow README.md for installation.
- Set `PYTHONPATH=.:miles` when running Python commands from the checkout.
  This also loads the SGLang compatibility hook in `sitecustomize.py`.
- Run `python -m pytest tests -q` after changes to adapters, launch arguments,
  data sources, or the bundled runtime. Add regression coverage for behavior
  changes rather than tests that only duplicate implementation details.
- `run_chess_miles --dry-run` validates the training file and prints the command.
  The preparation flags can still download data or models during a dry run.
- CPU tests do not validate GPU training. Changes to training, tokenizer,
  rollout, or checkpoint behavior require a GPU canary, an optimizer update,
  and save/reload/resume verification before a production experiment.

## Training correctness

- Preserve FP32 optimizer-facing parameters, Adam moments, gradient
  accumulation/reduction, and resumable checkpoint tensors. Use BF16 for
  forward/backward computation and rollout inference. Verify actual dtypes.
- Prompts begin with exactly one BOS. Environment replies and prompt tokens
  are excluded from policy loss; generated response tokens are included.
- Authenticate the full tokenizer vocabulary and preserve token IDs.
- Keep model context, training context, and SGLang context consistent.
- Keep online reward consistent with the independent offline scorer. Test
  setup moves and both white and black castling when changing move handling.
- Preserve global token-mean loss across microbatches and data-parallel ranks.
  Record prompts per batch, samples per prompt, and optimizer updates per batch.
- Verify rollout weights advance after each policy update.
- Record dataset/checkpoint hashes, source revision, image digest, seed,
  optimizer/loss settings, batch sizes, and output identity for each experiment.
- Resume only compatible checkpoints. Changes to data, tokenizer, precision,
  loss, or batch sizes require a fresh experiment identity and output path.
- Keep credentials in environment variables. Do not print
  tokens or add `.env` files and launch recovery records to source control.
- Use explicit run names and output paths for launches and uploads. Inspect
  existing outputs before a resume.

## Communication

Write technical, plain English. Use standard ML terms and exact numbers.
Avoid invented shorthand, promotional language, and speculative run summaries.
State unverified results and testing limitations directly.
