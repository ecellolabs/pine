#!/bin/bash -l
set -euo pipefail

# Runs train/sft_lora.sh on one GPU inside the ms-swift image (setup_swift_image.sh).
# All sft_lora.sh variables pass through (SFT_DIR and OUT are required):
#
#   SFT_DIR=/netscratch/$USER/data/vidore_v3/sft/smoke OUT=/netscratch/$USER/outputs/star/smoke \
#       MAX_STEPS=4 SAVE_STEPS=4 GRAD_ACCUM=1 TIME=00:45:00 bash train/run_train_cluster.sh
#   SFT_DIR=/netscratch/$USER/data/vidore_v3/sft/k4 OUT=/netscratch/$USER/outputs/star/k4 \
#       bash train/run_train_cluster.sh
#
# Resources: 80 GB GPUs only. A short example peaks at ~25 GB, but the loss over a
# ~10k-token reasoning trace allocates ~9 GB more (248k-token vocabulary) and ran out of
# memory on a 44 GB RTX A6000. 48 GB of RAM covers loading the 19 GB of weights plus 4 data
# workers (raise MEMORY if the smoke test is OOM-killed). The model comes from the HF
# cache vLLM already filled.

: "${SFT_DIR:?set SFT_DIR}"
: "${OUT:?set OUT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="/netscratch/$USER/envs/${IMAGE_NAME:-swift}.sqsh"
MOUNTS="/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER"

set -x
srun \
    --container-image="$IMAGE" \
    --container-mounts="$MOUNTS" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="${CPUS:-4}" \
    --mem="${MEMORY:-48}G" \
    --partition="${PARTITION:-A100-80GB,H100,H200}" \
    --gpus-per-task=1 \
    --time="${TIME:-06:00:00}" \
    --export=ALL,HF_HOME="${HF_HOME:-/netscratch/$USER/cache/hf}",ATTN="${ATTN:-sdpa}" \
    bash -c 'export PATH=/opt/env/.venv-swift/bin:$PATH; exec bash "$0"' "$SCRIPT_DIR/sft_lora.sh"
