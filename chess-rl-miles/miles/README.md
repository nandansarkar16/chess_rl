# Bundled Miles runtime

Source: https://github.com/radixark/miles

Upstream base: `e20de26c94412301ba2a746e8d942220bad0d00d`.
License: Apache-2.0; see `LICENSE`.

This runtime was imported from the local `interleave-pt-rl/miles` working tree
on 2026-09-07, including uncommitted FSDP/Adam changes. The source repository
HEAD was `ab1ba9b`; this is a working-tree snapshot, not that commit alone.

The chess adapters require these modifications for rollout routing, SGLang
compatibility, CISPO, token-mean loss, FP32 optimizer/checkpoint handling, and
optional Adam-state initialization. Use this tree directly. The old
`our_changes.patch` and `OUR_CHANGES.md` were omitted because they no longer
describe the complete runtime.

Upstream docs, images, Docker recipes, example launch scripts, and its broad
integration-test suite are omitted. The runtime packages, training entry points,
conversion tools, package metadata, and license are retained. Chess regression
tests live in `../tests`.

Files added or modified relative to the upstream base:

- `miles/__pycache__/__init__.cpython-313.pyc`
- `miles/backends/__pycache__/__init__.cpython-313.pyc`
- `miles/backends/experimental/__pycache__/__init__.cpython-313.pyc`
- `miles/backends/experimental/fsdp_utils/__init__.py`
- `miles/backends/experimental/fsdp_utils/__pycache__/__init__.cpython-313.pyc`
- `miles/backends/experimental/fsdp_utils/__pycache__/initial_adam.cpython-313.pyc`
- `miles/backends/experimental/fsdp_utils/__pycache__/precision.cpython-313.pyc`
- `miles/backends/experimental/fsdp_utils/__pycache__/update_weight_utils.cpython-313.pyc`
- `miles/backends/experimental/fsdp_utils/actor.py`
- `miles/backends/experimental/fsdp_utils/arguments.py`
- `miles/backends/experimental/fsdp_utils/checkpoint.py`
- `miles/backends/experimental/fsdp_utils/initial_adam.py`
- `miles/backends/experimental/fsdp_utils/precision.py`
- `miles/backends/experimental/fsdp_utils/update_weight_utils.py`
- `miles/backends/sglang_utils/__pycache__/arguments.cpython-313.pyc`
- `miles/backends/sglang_utils/__pycache__/sglang_engine.cpython-313.pyc`
- `miles/backends/sglang_utils/arguments.py`
- `miles/backends/sglang_utils/sglang_engine.py`
- `miles/backends/training_utils/data.py`
- `miles/backends/training_utils/loss.py`
- `miles/backends/training_utils/loss_hub/advantages.py`
- `miles/backends/training_utils/loss_hub/losses.py`
- `miles/backends/training_utils/loss_hub/math_utils.py`
- `miles/ray/actor_group.py`
- `miles/ray/placement_group.py`
- `miles/ray/rollout.py`
- `miles/ray/train_actor.py`
- `miles/rollout/__pycache__/__init__.cpython-313.pyc`
- `miles/rollout/__pycache__/base_types.cpython-313.pyc`
- `miles/rollout/__pycache__/data_source.cpython-313.pyc`
- `miles/rollout/data_source.py`
- `miles/rollout/env_reply_ordering.py`
- `miles/rollout/generate_utils/__pycache__/__init__.cpython-313.pyc`
- `miles/rollout/generate_utils/__pycache__/generate_endpoint_utils.cpython-313.pyc`
- `miles/rollout/inference_rollout/inference_rollout_eval.py`
- `miles/rollout/inference_rollout/inference_rollout_train.py`
- `miles/rollout/sglang_rollout.py`
- `miles/router/router.py`
- `miles/utils/__pycache__/__init__.cpython-313.pyc`
- `miles/utils/__pycache__/data.cpython-313.pyc`
- `miles/utils/__pycache__/http_utils.cpython-313.pyc`
- `miles/utils/__pycache__/metric_utils.cpython-313.pyc`
- `miles/utils/__pycache__/misc.cpython-313.pyc`
- `miles/utils/__pycache__/processing_utils.cpython-313.pyc`
- `miles/utils/__pycache__/timer.cpython-313.pyc`
- `miles/utils/__pycache__/types.cpython-313.pyc`
- `miles/utils/arguments.py`
- `miles/utils/data.py`
- `miles/utils/tracking_utils/__init__.py`
- `miles/utils/tracking_utils/wandb_utils.py`
- `tools/convert_fsdp_to_hf.py`
- `tools/export_dcp_adam.py`
- `train.py`
- `train_async.py`
