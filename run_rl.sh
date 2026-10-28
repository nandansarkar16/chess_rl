#!/bin/bash
# Launch one chess easy-to-hard RL arm. Run inside the Miles image (see README).
#
#   ARM=all|easy|mixed|hard SFT_CKPT=... DATA_DIR=... OUTPUT_DIR=... bash run_rl.sh [launcher args]
#
# ARM=all   -> combined.parquet       rule-based
# ARM=easy  -> easy.parquet           rule-based
# ARM=mixed -> combined_mixed.parquet rule-based easy + majority-vote hard
# ARM=hard  -> hard_mv.parquet        majority-vote on hard only
#
# Default geometry: 64 prompts x 16 samples per step, majority vote over the
# same 16. LEGACY_32_8=1 restores the old one: 64 x 8, vote over 32.
# Extra args are passed to chess_rl_miles.scripts.run_chess_miles (see --help).

set -eo pipefail

: "${ARM:?export ARM=all|easy|mixed|hard}"
: "${SFT_CKPT:?export SFT_CKPT=/path/to/hf_sft_checkpoint}"
: "${DATA_DIR:?export DATA_DIR=/path/to/rl_parquets}"
: "${OUTPUT_DIR:?export OUTPUT_DIR=/path/to/outputs}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${REPO}/chess-rl-miles"

if [[ "${LEGACY_32_8:-0}" == 1 ]]; then
    N_SAMPLES=8;  MV_CONFIG="${REPO}/e2h/chess_mv_32v8.yaml"; GEOM="g8v32"
else
    N_SAMPLES=16; MV_CONFIG="${REPO}/e2h/chess_mv_16.yaml";   GEOM="g16"
fi
PROMPTS=64

CONFIG_ARGS=()
# Passed after "--"; these win over the same flags emitted by the launcher.
EXTRA_ARGS=()
case "${ARM}" in
    easy)  TRAIN_FILE="easy.parquet";           REWARD_ARM="rb" ;;
    all)   TRAIN_FILE="combined.parquet";       REWARD_ARM="rb" ;;
    mixed) TRAIN_FILE="combined_mixed.parquet"; REWARD_ARM="rb_mv"
           CONFIG_ARGS=(--custom-config "${MV_CONFIG}")
           EXTRA_ARGS+=(--rollout-function-path e2h_miles_ttrl.ChessTTRLRolloutFn) ;;
    hard)  TRAIN_FILE="hard_mv.parquet";        REWARD_ARM="mv"
           CONFIG_ARGS=(--custom-config "${MV_CONFIG}")
           EXTRA_ARGS+=(--rollout-function-path e2h_miles_ttrl.ChessTTRLRolloutFn) ;;
    *)     echo "ERROR: ARM=${ARM}"; exit 1 ;;
esac

MODEL_ID="${MODEL_ID:-$(basename "${SFT_CKPT}")}"
RUN_NAME="${ARM}_${REWARD_ARM}_${GEOM}_${MODEL_ID}"
TRAIN_DATA="${DATA_DIR}/${TRAIN_FILE}"
SAVE_DIR="${OUTPUT_DIR}/${MODEL_ID}/${ARM}_${REWARD_ARM}_${GEOM}"

[[ -f "${SFT_CKPT}/config.json" ]] || { echo "missing ${SFT_CKPT}/config.json"; exit 1; }
[[ -f "${TRAIN_DATA}" ]] || { echo "missing ${TRAIN_DATA}"; exit 1; }

# RESUME=1 continues the arm from its last saved step; --num-rollout stays a
# total, not a delta. Bail on more than one tree: picking the wrong one resumes
# another arm's state.
if [[ "${RESUME:-0}" == 1 ]]; then
    mapfile -t TRACKERS < <(find "${SAVE_DIR}" -name latest_checkpointed_iteration.txt 2>/dev/null)
    [[ ${#TRACKERS[@]} -eq 1 ]] || { echo "want 1 checkpoint tree under ${SAVE_DIR}, found ${#TRACKERS[@]}"; exit 1; }
    RESUME_STEP="$(cat "${TRACKERS[0]}")"
    echo "resume: step ${RESUME_STEP} from $(dirname "${TRACKERS[0]}")"
    # --start-rollout-id is mandatory: the FSDP actor defaults it to 0 before
    # finalize_load compares it against the checkpoint cursor.
    EXTRA_ARGS+=(--load "$(dirname "${TRACKERS[0]}")" --start-rollout-id "${RESUME_STEP}")
fi

# The launcher enables W&B only when WANDB_API_KEY is non-empty.
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"
EXTRA_ARGS+=(--wandb-dir "${WANDB_DIR}")

export PYTHONPATH="${PROJ}:${PROJ}/miles:${REPO}/e2h${PYTHONPATH:+:${PYTHONPATH}}"
echo "ARM=${ARM}_${REWARD_ARM}  GEOM=${GEOM}  SFT=${SFT_CKPT}  SAVE_DIR=${SAVE_DIR}"

cd "${PROJ}"
python -m chess_rl_miles.scripts.run_chess_miles \
    --hf-checkpoint "${SFT_CKPT}" \
    --train-file "${TRAIN_DATA}" \
    --save-dir "${SAVE_DIR}" \
    --model-id "${MODEL_ID}" \
    --run-name "${RUN_NAME}" \
    --no-cispo \
    --rollout-batch-size "${PROMPTS}" \
    --over-sampling-batch-size "${PROMPTS}" \
    --n-samples-per-prompt "${N_SAMPLES}" \
    --global-batch-size $((PROMPTS * N_SAMPLES)) \
    --num-rollout "${NUM_STEPS:-2000}" \
    --save-interval "${SAVE_INTERVAL:-100}" \
    --actor-num-gpus-per-node "${NUM_GPUS:-1}" \
    --sglang-context-length 3072 \
    --cpu-threads "${CPU_THREADS:-$(nproc)}" \
    --wandb-project "${WANDB_PROJECT:-chess-e2h-rl}" \
    --wandb-team "${WANDB_ENTITY:-}" \
    "$@" \
    "${CONFIG_ARGS[@]}" -- "${EXTRA_ARGS[@]}"
echo Done.
