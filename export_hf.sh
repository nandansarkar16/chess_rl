#!/bin/bash
# Convert one arm's Miles checkpoints to HF format and optionally upload them.
# Run inside the Miles image with the same env vars as run_rl.sh.
#
#   ARM=... SFT_CKPT=... OUTPUT_DIR=... [HF_REPO=user/repo] bash export_hf.sh
#
# Writes $SAVE_DIR/hf/global_step_N. With HF_REPO set, uploads them to
# <MODEL_ID>/<arm>/global_step_N in that (private) model repo.

set -eo pipefail

: "${ARM:?export ARM=all|easy|mixed|hard}"
: "${SFT_CKPT:?export SFT_CKPT=/path/to/hf_sft_checkpoint}"
: "${OUTPUT_DIR:?export OUTPUT_DIR=/path/to/outputs}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${REPO}/chess-rl-miles"

[[ "${LEGACY_32_8:-0}" == 1 ]] && GEOM="g8v32" || GEOM="g16"
case "${ARM}" in
    easy|all) REWARD_ARM="rb" ;;
    mixed)    REWARD_ARM="rb_mv" ;;
    hard)     REWARD_ARM="mv" ;;
    *)        echo "ERROR: ARM=${ARM}"; exit 1 ;;
esac
MODEL_ID="${MODEL_ID:-$(basename "${SFT_CKPT}")}"
RUN="${ARM}_${REWARD_ARM}_${GEOM}"
SAVE_DIR="${OUTPUT_DIR}/${MODEL_ID}/${RUN}"
HF_ROOT="${SAVE_DIR}/hf"

export PYTHONPATH="${PROJ}:${PROJ}/miles${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t ITERS < <(find "${SAVE_DIR}" -path "${HF_ROOT}" -prune -o -type d -name 'iter_*' -print | sort)
[[ ${#ITERS[@]} -gt 0 ]] || { echo "no iter_* under ${SAVE_DIR}"; exit 1; }

for ITER in "${ITERS[@]}"; do
    [[ -f "${ITER}/COMMITTED.json" ]] || { echo "[skip] uncommitted ${ITER}"; continue; }
    STEP=$((10#$(basename "${ITER}" | sed 's/^iter_//')))
    HF="${HF_ROOT}/global_step_${STEP}"
    if [[ ! -f "${HF}/model.safetensors" ]] && [[ ! -f "${HF}/model.safetensors.index.json" ]]; then
        echo "[convert] ${ITER} -> ${HF}"
        mkdir -p "${HF_ROOT}"
        python "${PROJ}/miles/tools/convert_fsdp_to_hf.py" \
            --input-dir "${ITER}" --origin-hf-dir "${SFT_CKPT}" --output-dir "${HF}"
    fi
    # The image's Transformers rewrites the YaRN config into a layout the eval
    # stack (vLLM) does not read; keep the SFT model's config.
    cp "${SFT_CKPT}/config.json" "${SFT_CKPT}/generation_config.json" "${HF}/"
done

if [[ -n "${HF_REPO:-}" ]]; then
    hf upload "${HF_REPO}" "${HF_ROOT}" "${MODEL_ID}/${RUN}" --repo-type model --private
fi
echo Done.
