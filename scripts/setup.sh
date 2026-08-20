#!/bin/bash
# Install LocateAnything-3B + locateanything-fix into a venv and download the model.
#   ./setup.sh [BASE_DIR] [PATCH_REPO_DIR]
# Defaults: PATCH = the repo this script lives in, BASE = that repo's parent.
# So a plain `bash scripts/setup.sh` works wherever you cloned it.
set -uo pipefail
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO="$(cd "$_SELF/.." && pwd)"
BASE="${1:-$(dirname "$_REPO")}"
PATCH="${2:-$_REPO}"
mkdir -p "$BASE"; cd "$BASE"

echo "=== [1/5] venv ==="
python3 -m venv venv || exit 1
source venv/bin/activate
python -m pip install -q -U pip wheel setuptools || exit 1
python -V

echo "=== [2/5] torch + torchvision ==="
# Pick the index matching your CUDA. cu128 covers Ampere (sm_86) through Blackwell.
# torchvision MUST come from the same index: the PyPI build has the right version
# number but the wrong CUDA ABI, so torchvision::nms fails to register and
# AutoProcessor will not import.
pip install torch torchvision --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" || exit 1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda)"

echo "=== [3/5] model deps ==="
# transformers is pinned: 5.x changed _check_and_adjust_attn_implementation's
# signature and the model's custom code fails to construct.
pip install "transformers==4.57.1" accelerate huggingface_hub pillow numpy \
            einops timm requests peft || exit 1
# decord is NOT optional if you want video: the model defaults to the torchvision
# backend, and torchvision >= 0.19 removed io.read_video, so decode fails outright.
for p in opencv-python-headless decord lmdb; do
  pip install "$p" || echo "OPTIONAL_FAILED: $p"
done

echo "=== [4/5] locateanything-fix ==="
if [ -d "$PATCH" ]; then
  pip install -e "$PATCH" || exit 1
else
  echo "!! patch repo not found at $PATCH -- clone it there, or pass it as arg 2"
  exit 1
fi

echo "=== [5/5] download nvidia/LocateAnything-3B ==="
export HF_HOME="$BASE/hf"
hf download nvidia/LocateAnything-3B --local-dir "$BASE/model" \
  || huggingface-cli download nvidia/LocateAnything-3B --local-dir "$BASE/model" \
  || exit 1

echo "=== SETUP_COMPLETE ==="
echo "model:  $BASE/model"
echo "venv :  source $BASE/venv/bin/activate"
echo "note :  export PYTHONPATH=$PATCH  (so locateanything_fix is importable)"
