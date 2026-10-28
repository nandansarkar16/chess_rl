"""TTRL rollout for the mixed/hard arms: vote over chess_ttrl_votes, train on n_samples_per_prompt.

Default is 16/16: vote over the same 16 samples GRPO trains on. The legacy
32/8 geometry (VERL's) generates 32 per prompt, takes the majority over all 32,
then runs GRPO on the first 8. The per-group coroutine generates any extra
vote-only samples, scores the trained samples against the majority, and
returns only the train group (Miles asserts n_samples_per_prompt). All vote
answers are stored on that group as metadata.ttrl_vote_answers so the rollout
logger can score entropy over the full TTRL group.

Rows are selected by extra_info.reward_type == "majority_vote", the tag written
by tag_mixed_reward_type.py; every other row takes the stock rule-based path.
"""

from __future__ import annotations

import copy

from chess_rl_miles import batched_rollout as _batched_rollout
from chess_rl_miles.moves import majority_vote
from chess_rl_miles.reward import _candidate_moves, _score_sample

_generate_group_stock = _batched_rollout.generate_group_batched


def _is_majority_vote_row(sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get("reward_type", "")).lower() == "majority_vote"


async def generate_group_ttrl(state, group, sampling_params, evaluation: bool = False):
    args = state.args
    votes = int(getattr(args, "chess_ttrl_votes", 0) or 0)
    if evaluation or not _is_majority_vote_row(group[0]):
        return await _generate_group_stock(
            state, group, sampling_params=sampling_params, evaluation=evaluation
        )

    # votes <= group size (e.g. 16/16) votes over the trained group itself. The
    # stock path would score these rows rule-based, i.e. against ground truth.
    vote_only = [copy.deepcopy(group[0]) for _ in range(max(votes - len(group), 0))]
    voted = await _generate_group_stock(
        state, group + vote_only, sampling_params=sampling_params, evaluation=evaluation
    )
    trained = voted[: len(group)]
    vote_moves = [_candidate_moves(args, sample) for sample in voted]
    vote_answers = [",".join(moves) for moves in vote_moves]
    target, ratio = majority_vote(vote_moves)
    for sample in trained:
        if target:
            sample.reward = _score_sample(
                args,
                sample,
                reward_type="MAJORITY_VOTE",
                majority_target=target,
                majority_ratio=ratio,
            )
        # GRPO sees only n_samples_per_prompt. The vote strings are copied onto
        # those rows so the rollout logger can compute entropy over the full
        # TTRL group.
        sample.metadata["ttrl_votes"] = len(voted)
        sample.metadata["ttrl_vote_answers"] = vote_answers
    return trained


_batched_rollout.generate_group_batched = generate_group_ttrl

ChessTTRLRolloutFn = _batched_rollout.ChessBatchedRolloutFn
