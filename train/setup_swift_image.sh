#!/bin/bash -l
set -euo pipefail

# Builds a SLURM container image with ms-swift at /opt/env/.venv-swift (srun --container-save),
# like vllm_serving/setup_vllm_cluster_image.sh. Run once from the login node:
#
#   bash train/setup_swift_image.sh            # -> /netscratch/$USER/envs/swift.sqsh
#   FLASH_ATTN=1 bash train/setup_swift_image.sh
#
# No GPU is needed: the CUDA extensions compile with the image's nvcc. Saving the image
# (mksquashfs) needs the memory: 32 GB was OOM-killed, 64 GB matches the vLLM image build.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-/enroot/nvcr.io_nvidia_pytorch_26.05-py3.sqsh}"
OUT_IMAGE="/netscratch/$USER/envs/${IMAGE_NAME:-swift}.sqsh"
MOUNTS="/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER"

[ -f "$BASE_IMAGE" ] || { echo "Base image not found: $BASE_IMAGE" >&2; exit 1; }
echo "Base image : $BASE_IMAGE"
echo "Will save to: $OUT_IMAGE"

set -x
srun \
    --container-image="$BASE_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-save="$OUT_IMAGE" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=8 \
    --mem="${MEMORY:-64}G" \
    --partition="${PARTITION:-A100-40GB,RTXA6000,H100,H200}" \
    --gpus-per-task=0 \
    --time=02:00:00 \
    --immediate=300 \
    --export=ALL,FLASH_ATTN="${FLASH_ATTN:-0}",CAUSAL_CONV1D="${CAUSAL_CONV1D:-0}" \
    /bin/bash -c "$SCRIPT_DIR/install_swift.sh /opt/env/.venv-swift" || {
    # pyxis saves the image even when the install or its import check failed
    rm -f "$OUT_IMAGE"
    echo "Build failed: $OUT_IMAGE removed (see the error above)." >&2
    exit 1
}
echo "Image ready: $OUT_IMAGE"
