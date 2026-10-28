from __future__ import annotations

import copy
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


class _FakeDataset:
    def __init__(self, size: int):
        self.samples = list(range(size))
        self.shuffle_epochs: list[int] = []

    def __len__(self) -> int:
        return len(self.samples)

    def shuffle(self, epoch: int) -> None:
        self.shuffle_epochs.append(epoch)


class _FakeRolloutDataSource:
    """Small behavioral stand-in for Miles' cursor implementation."""

    def __init__(self, args):
        self.args = args
        self.dataset = (
            _FakeDataset(args.dataset_size)
            if args.rollout_global_dataset
            else None
        )
        self.epoch_id = 0
        self.sample_offset = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata = {"identity": "test-dataset"}

    def get_samples(self, num_samples: int):
        values = self.dataset.samples[
            self.sample_offset : self.sample_offset + num_samples
        ]
        self.sample_offset += len(values)
        groups = []
        for value in values:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                group.append(
                    SimpleNamespace(
                        value=value,
                        group_index=self.sample_group_index,
                        index=self.sample_index,
                    )
                )
                self.sample_index += 1
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def checkpoint_state(self, rollout_id: int) -> dict:
        return {
            "schema": "miles-rollout-data-source-v1",
            "rollout_id": rollout_id,
            "next_rollout_id": rollout_id + 1,
            "dataset_length": len(self.dataset),
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": copy.deepcopy(self.metadata),
        }

    def restore_checkpoint_state(self, state_dict: dict, rollout_id: int) -> None:
        expected = {
            "schema": "miles-rollout-data-source-v1",
            "rollout_id": rollout_id,
            "next_rollout_id": rollout_id + 1,
            "dataset_length": len(self.dataset),
        }
        if any(state_dict.get(key) != value for key, value in expected.items()):
            raise RuntimeError("rollout data-source checkpoint identity mismatch")
        for key in (
            "sample_offset",
            "epoch_id",
            "sample_group_index",
            "sample_index",
        ):
            setattr(self, key, state_dict[key])
        self.metadata = copy.deepcopy(state_dict["metadata"])
        if self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


def _import_precision_source(
    monkeypatch,
    *,
    prompts_per_update: int | None = None,
    samples_per_prompt: int | None = None,
):
    monkeypatch.setenv(
        "CHESS_RL_MILES_DETERMINISTIC_SEED_MODE",
        "sample-index",
    )
    if prompts_per_update is None:
        monkeypatch.delenv("CHESS_RL_PROMPTS_PER_UPDATE", raising=False)
    else:
        monkeypatch.setenv(
            "CHESS_RL_PROMPTS_PER_UPDATE", str(prompts_per_update)
        )
    if samples_per_prompt is None:
        monkeypatch.delenv("CHESS_RL_SAMPLES_PER_PROMPT", raising=False)
    else:
        monkeypatch.setenv(
            "CHESS_RL_SAMPLES_PER_PROMPT", str(samples_per_prompt)
        )
    miles = types.ModuleType("miles")
    miles.__path__ = []
    rollout = types.ModuleType("miles.rollout")
    rollout.__path__ = []
    data_source = types.ModuleType("miles.rollout.data_source")
    data_source.RolloutDataSource = _FakeRolloutDataSource
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(
        sys.modules,
        "miles.rollout.data_source",
        data_source,
    )
    monkeypatch.delitem(
        sys.modules,
        "chess_rl_miles.precision_resume_data_source",
        raising=False,
    )
    return importlib.import_module(
        "chess_rl_miles.precision_resume_data_source"
    )


@pytest.fixture
def precision_source_module(monkeypatch):
    return _import_precision_source(monkeypatch)


@pytest.fixture
def precision_source_module_64x16(monkeypatch):
    return _import_precision_source(
        monkeypatch,
        prompts_per_update=64,
        samples_per_prompt=16,
    )


@pytest.fixture
def precision_source_module_64x32(monkeypatch):
    return _import_precision_source(
        monkeypatch,
        prompts_per_update=64,
        samples_per_prompt=32,
    )


def _args(*, leg: int, **overrides):
    values = {
        "rollout_global_dataset": True,
        "debug_rollout_only": False,
        "sglang_enable_deterministic_inference": True,
        "rollout_shuffle": True,
        "partial_rollout": False,
        "dynamic_sampling_filter_path": None,
        "use_fault_tolerance": False,
        "rollout_batch_size": 256,
        "over_sampling_batch_size": 256,
        "n_samples_per_prompt": 8,
        "num_steps_per_rollout": 1,
        "num_rollout": leg,
        "start_rollout_id": leg - 1,
        "dataset_size": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_precision_source_leg1_exact_cursor_and_checkpoint(
    precision_source_module,
    monkeypatch,
):
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "1")
    source = precision_source_module.PrecisionResumeRolloutDataSource(
        _args(leg=1)
    )
    assert (
        source.sample_offset,
        source.sample_group_index,
        source.sample_index,
    ) == (0, 0, 0)
    with pytest.raises(RuntimeError, match="only after"):
        source.checkpoint_state(0)

    groups = source.get_samples(256)
    assert len(groups) == 256
    assert all(len(group) == 8 for group in groups)
    assert (
        source.sample_offset,
        source.epoch_id,
        source.sample_group_index,
        source.sample_index,
    ) == (256, 0, 256, 2_048)
    state = source.checkpoint_state(0)
    assert state["rollout_id"] == 0
    assert state["next_rollout_id"] == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        source.get_samples(256)
    source.add_samples([])
    with pytest.raises(RuntimeError, match="refuses aborted-sample requeue"):
        source.add_samples([[object()]])


def test_precision_source_leg2_requires_restore_then_continues_exact_cursor(
    precision_source_module,
    monkeypatch,
):
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "2")
    source = precision_source_module.PrecisionResumeRolloutDataSource(
        _args(leg=2)
    )
    with pytest.raises(RuntimeError, match="must restore checkpoint-1"):
        source.get_samples(256)

    checkpoint_one = {
        "schema": "miles-rollout-data-source-v1",
        "rollout_id": 0,
        "next_rollout_id": 1,
        "dataset_length": 512,
        "sample_offset": 256,
        "epoch_id": 0,
        "sample_group_index": 256,
        "sample_index": 2_048,
        "metadata": {"identity": "test-dataset"},
    }
    source.restore_checkpoint_state(checkpoint_one, 0)
    assert source.dataset.shuffle_epochs == [0]
    groups = source.get_samples(256)
    assert len(groups) == 256
    assert groups[0][0].group_index == 256
    assert groups[0][0].index == 2_048
    assert (
        source.sample_offset,
        source.epoch_id,
        source.sample_group_index,
        source.sample_index,
    ) == (512, 0, 512, 4_096)
    state = source.checkpoint_state(1)
    assert state["rollout_id"] == 1
    assert state["next_rollout_id"] == 2


def test_precision_source_leg2_rejects_drifted_checkpoint_cursor(
    precision_source_module,
    monkeypatch,
):
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "2")
    source = precision_source_module.PrecisionResumeRolloutDataSource(
        _args(leg=2)
    )
    state = {
        "schema": "miles-rollout-data-source-v1",
        "rollout_id": 0,
        "next_rollout_id": 1,
        "dataset_length": 512,
        "sample_offset": 255,
        "epoch_id": 0,
        "sample_group_index": 256,
        "sample_index": 2_048,
        "metadata": {"identity": "test-dataset"},
    }
    with pytest.raises(RuntimeError, match="checkpoint-1 cursor drifted"):
        source.restore_checkpoint_state(state, 0)


def test_precision_source_64x16_two_leg_cursor_and_checkpoint(
    precision_source_module_64x16,
    monkeypatch,
):
    module = precision_source_module_64x16
    assert module.EXPECTED_ROLLOUT_BATCH_SIZE == 64
    assert module.EXPECTED_SAMPLES_PER_PROMPT == 16
    assert module.EXPECTED_TRAJECTORIES_PER_UPDATE == 1_024
    assert module.MINIMUM_ADMITTED_PROMPTS == 128

    args = _args(
        leg=1,
        rollout_batch_size=64,
        over_sampling_batch_size=64,
        n_samples_per_prompt=16,
        dataset_size=128,
    )
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "1")
    first = module.PrecisionResumeRolloutDataSource(args)
    groups = first.get_samples(64)
    assert len(groups) == 64
    assert all(len(group) == 16 for group in groups)
    assert (
        first.sample_offset,
        first.epoch_id,
        first.sample_group_index,
        first.sample_index,
    ) == (64, 0, 64, 1_024)
    checkpoint = first.checkpoint_state(0)

    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "2")
    second = module.PrecisionResumeRolloutDataSource(
        _args(
            leg=2,
            rollout_batch_size=64,
            over_sampling_batch_size=64,
            n_samples_per_prompt=16,
            dataset_size=128,
        )
    )
    second.restore_checkpoint_state(checkpoint, 0)
    groups = second.get_samples(64)
    assert groups[0][0].group_index == 64
    assert groups[0][0].index == 1_024
    assert (
        second.sample_offset,
        second.epoch_id,
        second.sample_group_index,
        second.sample_index,
    ) == (128, 0, 128, 2_048)
    assert second.checkpoint_state(1)["next_rollout_id"] == 2


def test_precision_source_64x16_requires_two_prompt_batches(
    precision_source_module_64x16,
    monkeypatch,
):
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "1")
    with pytest.raises(ValueError, match="at least 128"):
        precision_source_module_64x16.PrecisionResumeRolloutDataSource(
            _args(
                leg=1,
                rollout_batch_size=64,
                over_sampling_batch_size=64,
                n_samples_per_prompt=16,
                dataset_size=127,
            )
        )


def test_precision_source_64x32_two_leg_cursor_and_checkpoint(
    precision_source_module_64x32,
    monkeypatch,
):
    module = precision_source_module_64x32
    assert module.EXPECTED_ROLLOUT_BATCH_SIZE == 64
    assert module.EXPECTED_SAMPLES_PER_PROMPT == 32
    assert module.EXPECTED_TRAJECTORIES_PER_UPDATE == 2_048
    assert module.MINIMUM_ADMITTED_PROMPTS == 128

    args = _args(
        leg=1,
        rollout_batch_size=64,
        over_sampling_batch_size=64,
        n_samples_per_prompt=32,
        dataset_size=128,
    )
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "1")
    first = module.PrecisionResumeRolloutDataSource(args)
    groups = first.get_samples(64)
    assert len(groups) == 64
    assert all(len(group) == 32 for group in groups)
    assert (
        first.sample_offset,
        first.epoch_id,
        first.sample_group_index,
        first.sample_index,
    ) == (64, 0, 64, 2_048)
    checkpoint = first.checkpoint_state(0)

    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "2")
    second = module.PrecisionResumeRolloutDataSource(
        _args(
            leg=2,
            rollout_batch_size=64,
            over_sampling_batch_size=64,
            n_samples_per_prompt=32,
            dataset_size=128,
        )
    )
    second.restore_checkpoint_state(checkpoint, 0)
    groups = second.get_samples(64)
    assert groups[0][0].group_index == 64
    assert groups[0][0].index == 2_048
    assert (
        second.sample_offset,
        second.epoch_id,
        second.sample_group_index,
        second.sample_index,
    ) == (128, 0, 128, 4_096)
    assert second.checkpoint_state(1)["next_rollout_id"] == 2


def test_production_source_uses_same_cursor_and_failure_semantics(
    precision_source_module,
    monkeypatch,
):
    monkeypatch.delenv("CHESS_RL_MILES_PRECISION_GATE_LEG", raising=False)
    source = precision_source_module.PrecisionResumeRolloutDataSource(
        _args(leg=3, num_rollout=1_500, start_rollout_id=0)
    )
    groups = source.get_samples(256)
    assert len(groups) == 256
    assert (
        source.sample_offset,
        source.sample_group_index,
        source.sample_index,
    ) == (256, 256, 2_048)
    checkpoint = source.checkpoint_state(0)

    resumed = precision_source_module.PrecisionResumeRolloutDataSource(
        _args(leg=3, num_rollout=1_500, start_rollout_id=1)
    )
    resumed.restore_checkpoint_state(checkpoint, 0)
    next_groups = resumed.get_samples(256)
    assert next_groups[0][0].group_index == 256
    assert next_groups[0][0].index == 2_048
    assert (
        resumed.sample_offset,
        resumed.sample_group_index,
        resumed.sample_index,
    ) == (512, 512, 4_096)
    with pytest.raises(RuntimeError, match="refuses aborted-sample requeue"):
        resumed.add_samples([[object()]])


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"rollout_global_dataset": False}, "global_dataset"),
        ({"debug_rollout_only": True}, "debug_rollout_only"),
        ({"sglang_enable_deterministic_inference": False}, "deterministic"),
        ({"rollout_shuffle": False}, "rollout_shuffle"),
        ({"partial_rollout": True}, "partial_rollout"),
        ({"dynamic_sampling_filter_path": "filter"}, "dynamic sampling"),
        ({"use_fault_tolerance": True}, "fault tolerance"),
        ({"rollout_batch_size": 128}, "rollout_batch_size"),
        ({"over_sampling_batch_size": 128}, "over_sampling_batch_size"),
        ({"n_samples_per_prompt": 4}, "n_samples_per_prompt"),
        ({"num_steps_per_rollout": 2}, "num_steps_per_rollout"),
        ({"dataset_size": 511}, "at least 512"),
    ],
)
def test_precision_source_rejects_contract_drift(
    precision_source_module,
    monkeypatch,
    overrides,
    error,
):
    monkeypatch.setenv("CHESS_RL_MILES_PRECISION_GATE_LEG", "1")
    with pytest.raises(ValueError, match=error):
        precision_source_module.PrecisionResumeRolloutDataSource(
            _args(leg=1, **overrides)
        )
