#!/bin/bash
set -euo pipefail

# Creates a Python 3.12 venv with ms-swift and the Qwen3.5 training dependencies (pins from
# ms-swift's Qwen3.5 best-practice page). Runs inside the container started by
# setup_swift_image.sh; not meant to be run directly.
#
#   install_swift.sh [VENV_DIR]
#
# FLASH_ATTN=1 also builds flash-attn (slow to compile); without it train with ATTN=sdpa.
# CAUSAL_CONV1D=1 compiles causal-conv1d from source (optional kernel).

VENV_DIR="${1:-/opt/env/.venv-swift}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *) export PATH="$HOME/.local/bin:$PATH";; esac

if [ ! -x "$VENV_DIR/bin/python" ]; then
  uv venv "$VENV_DIR" --python 3.12 --seed
fi
PY="$VENV_DIR/bin/python"

uv pip install --python "$PY" -U ms-swift "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel
# qwen_vl_utils imports torchvision, which nothing above installs; pin torch so the
# resolver picks the matching torchvision instead of replacing torch
TORCH_VERSION=$("$PY" -c "import torch; print(torch.__version__.split('+')[0])")
uv pip install --python "$PY" "torch==$TORCH_VERSION" torchvision
# fast kernels for Qwen3.5's linear-attention layers
uv pip install --python "$PY" -U "flash-linear-attention>=0.4.2" --no-build-isolation
# causal-conv1d is optional (transformers falls back to torch for that small conv). Its setup
# downloads a prebuilt binary for the nearest torch, which failed with an undefined symbol
# against this torch; CAUSAL_CONV1D=1 compiles it from source instead (slow).
if [ "${CAUSAL_CONV1D:-0}" = "1" ]; then
  CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python "$PY" --no-cache --reinstall \
    git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
fi
if [ "${FLASH_ATTN:-0}" = "1" ]; then
  uv pip install --python "$PY" "flash-attn==2.8.3" --no-build-isolation
fi

# exits non-zero if anything training imports is missing; setup_swift_image.sh then deletes
# the image pyxis saved anyway
CHECK="import swift, transformers, torch, torchvision, qwen_vl_utils, peft, fla"
[ "${CAUSAL_CONV1D:-0}" = "1" ] && CHECK="$CHECK, causal_conv1d"
"$PY" -c "$CHECK; print('swift', swift.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| torchvision', torchvision.__version__)"
echo "Done. swift: $VENV_DIR/bin/swift"
