"""Deterministic checkpoint-aware prompt source for production and its gate.

Production and the BF16/FP32 resume gate deliberately use this same source and
failure behavior. The prompt and sibling counts are bound into the deployed
image through the same environment variables as the launcher. In gate mode,
leg 1 consumes one complete prompt batch. A fresh leg-2 process must restore
the authenticated checkpoint-1 cursor and consume the next complete batch.
Outside gate mode, ordinary committed cursor checkpoint/restore supports the
complete production run.
"""

from __future__ import annotations

import os
from typing import Any

from miles.rollout.data_source import RolloutDataSource


def _policy_update_geometry_from_environment() -> tuple[int, int, int]:
    try:
        prompts = int(os.environ.get("CHESS_RL_PROMPTS_PER_UPDATE", "256"))
        samples = int(os.environ.get("CHESS_RL_SAMPLES_PER_PROMPT", "8"))
    except ValueError as exc:
        raise RuntimeError(
            "policy-update geometry environment values must be integers"
        ) from exc
    if (prompts, samples) not in {(256, 8), (64, 16), (64, 32)}:
        raise RuntimeError(
            "policy-update geometry must be either the legacy 256x8 contract "
            "or an approved 64x16 or 64x32 contract"
        )
    return prompts, samples, prompts * samples


(
    EXPECTED_ROLLOUT_BATCH_SIZE,
    EXPECTED_SAMPLES_PER_PROMPT,
    EXPECTED_TRAJECTORIES_PER_UPDATE,
) = _policy_update_geometry_from_environment()
MINIMUM_ADMITTED_PROMPTS = 2 * EXPECTED_ROLLOUT_BATCH_SIZE


class PrecisionResumeRolloutDataSource(RolloutDataSource):
    """Use deterministic prompt cursors; add exact two-leg gate assertions."""

    strict_exact_once = True

    def __init__(self, args: Any):
        super().__init__(args)
        leg_text = os.environ.get("CHESS_RL_MILES_PRECISION_GATE_LEG")
        leg = int(leg_text) if leg_text in {"1", "2"} else None
        violations: list[str] = []
        if self.dataset is None:
            violations.append("rollout_global_dataset must be enabled")
        if bool(getattr(args, "debug_rollout_only", False)):
            violations.append("debug_rollout_only must be disabled")
        if not bool(
            getattr(args, "sglang_enable_deterministic_inference", False)
        ):
            violations.append(
                "sglang_enable_deterministic_inference must be enabled"
            )
        if os.environ.get("CHESS_RL_MILES_DETERMINISTIC_SEED_MODE") != "sample-index":
            violations.append("deterministic seed mode must equal sample-index")
        if not bool(getattr(args, "rollout_shuffle", False)):
            violations.append("rollout_shuffle must be enabled")
        if bool(getattr(args, "partial_rollout", False)):
            violations.append("partial_rollout must be disabled")
        if getattr(args, "dynamic_sampling_filter_path", None) is not None:
            violations.append("dynamic sampling must be disabled")
        if bool(getattr(args, "use_fault_tolerance", False)):
            violations.append("rollout fault tolerance must be disabled")
        if getattr(args, "rollout_batch_size", None) != EXPECTED_ROLLOUT_BATCH_SIZE:
            violations.append(
                "rollout_batch_size must equal "
                f"{EXPECTED_ROLLOUT_BATCH_SIZE}"
            )
        if getattr(args, "over_sampling_batch_size", None) != EXPECTED_ROLLOUT_BATCH_SIZE:
            violations.append(
                "over_sampling_batch_size must equal "
                f"{EXPECTED_ROLLOUT_BATCH_SIZE}"
            )
        if getattr(args, "n_samples_per_prompt", None) != EXPECTED_SAMPLES_PER_PROMPT:
            violations.append(
                "n_samples_per_prompt must equal "
                f"{EXPECTED_SAMPLES_PER_PROMPT}"
            )
        if getattr(args, "num_steps_per_rollout", None) != 1:
            violations.append("num_steps_per_rollout must equal 1")
        num_rollout = getattr(args, "num_rollout", None)
        if isinstance(num_rollout, bool) or not isinstance(num_rollout, int) or num_rollout <= 0:
            violations.append("num_rollout must be a positive integer")
        if leg is not None:
            expected_num_rollout = 1 if leg == 1 else 2
            if num_rollout != expected_num_rollout:
                violations.append(
                    f"num_rollout must equal {expected_num_rollout!r} for leg {leg!r}"
                )
            expected_start = 0 if leg == 1 else 1
            observed_start = getattr(args, "start_rollout_id", None)
            if observed_start not in {None, expected_start}:
                violations.append(
                    f"start_rollout_id must equal {expected_start!r} for leg {leg!r}"
                )
        if (
            leg is not None
            and self.dataset is not None
            and len(self.dataset) < MINIMUM_ADMITTED_PROMPTS
        ):
            violations.append(
                "post-tokenization dataset must admit at least "
                f"{MINIMUM_ADMITTED_PROMPTS} prompts"
            )
        if violations:
            raise ValueError(
                "precision resume rollout source rejected configuration: "
                + "; ".join(violations)
            )

        self._precision_leg = leg
        self._precision_get_calls = 0
        self._precision_restored = False

    def _expected_before(self) -> tuple[int, int, int]:
        if self._precision_leg == 1:
            return (0, 0, 0)
        return (
            EXPECTED_ROLLOUT_BATCH_SIZE,
            EXPECTED_ROLLOUT_BATCH_SIZE,
            EXPECTED_TRAJECTORIES_PER_UPDATE,
        )

    def get_samples(self, num_samples: int):
        if self._precision_leg is None:
            if num_samples != EXPECTED_ROLLOUT_BATCH_SIZE:
                raise RuntimeError(
                    "deterministic checkpoint source only permits complete "
                    f"{EXPECTED_ROLLOUT_BATCH_SIZE}-prompt requests"
                )
            return super().get_samples(num_samples)
        if self._precision_get_calls != 0:
            raise RuntimeError(
                "precision resume source permits exactly one prompt request per process"
            )
        if num_samples != EXPECTED_ROLLOUT_BATCH_SIZE:
            raise RuntimeError(
                "precision resume source only permits one "
                f"{EXPECTED_ROLLOUT_BATCH_SIZE}-prompt request"
            )
        if self._precision_leg == 2 and not self._precision_restored:
            raise RuntimeError(
                "precision resume leg 2 must restore checkpoint-1 cursor before sampling"
            )
        expected_offset, expected_group, expected_sample = self._expected_before()
        observed_before = (
            self.sample_offset,
            self.sample_group_index,
            self.sample_index,
        )
        if observed_before != (expected_offset, expected_group, expected_sample):
            raise RuntimeError(
                "precision resume cursor before sampling drifted: "
                f"expected={(expected_offset, expected_group, expected_sample)} "
                f"actual={observed_before}"
            )
        epoch_before = self.epoch_id
        groups = super().get_samples(num_samples)
        self._precision_get_calls += 1
        if self.epoch_id != epoch_before or self.epoch_id != 0:
            raise RuntimeError("precision resume source changed dataset epoch")
        expected_after = (
            expected_offset + EXPECTED_ROLLOUT_BATCH_SIZE,
            expected_group + EXPECTED_ROLLOUT_BATCH_SIZE,
            expected_sample + EXPECTED_TRAJECTORIES_PER_UPDATE,
        )
        observed_after = (
            self.sample_offset,
            self.sample_group_index,
            self.sample_index,
        )
        if (
            observed_after != expected_after
            or len(groups) != EXPECTED_ROLLOUT_BATCH_SIZE
        ):
            raise RuntimeError(
                "precision resume source advanced an unexpected cursor: "
                f"expected={expected_after} actual={observed_after} groups={len(groups)}"
            )
        return groups

    def restore_checkpoint_state(self, state_dict: dict, rollout_id: int) -> None:
        if self._precision_leg is None:
            super().restore_checkpoint_state(state_dict, rollout_id)
            return
        if self._precision_leg != 2 or rollout_id != 0:
            raise RuntimeError(
                "only precision resume leg 2 may restore checkpoint-1 cursor"
            )
        super().restore_checkpoint_state(state_dict, rollout_id)
        expected = (
            EXPECTED_ROLLOUT_BATCH_SIZE,
            0,
            EXPECTED_ROLLOUT_BATCH_SIZE,
            EXPECTED_TRAJECTORIES_PER_UPDATE,
        )
        observed = (
            self.sample_offset,
            self.epoch_id,
            self.sample_group_index,
            self.sample_index,
        )
        if observed != expected:
            raise RuntimeError(
                "precision resume checkpoint-1 cursor drifted: "
                f"expected={expected} actual={observed}"
            )
        self._precision_restored = True

    def checkpoint_state(self, rollout_id: int) -> dict:
        if self._precision_leg is None:
            return super().checkpoint_state(rollout_id)
        expected_rollout = self._precision_leg - 1
        if rollout_id != expected_rollout or self._precision_get_calls != 1:
            raise RuntimeError(
                "precision resume source can checkpoint only after its one exact prompt batch"
            )
        return super().checkpoint_state(rollout_id)

    def add_samples(self, samples):
        if samples:
            raise RuntimeError(
                "precision resume source refuses aborted-sample requeue"
            )
