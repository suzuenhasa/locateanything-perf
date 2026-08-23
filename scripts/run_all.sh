#!/usr/bin/env bash
# The image-side measurements in one command: gate on the patch being live and
# numerically equivalent, then run the A/B sweep and the batch throughput sweep.
#
# Video (batch_video.py + mkvideo.py) is deliberately left out -- it needs a clip
# and ffmpeg extraction, so it stays a manual step. See scripts/README.md.
#
# Everything takes --model, so nothing here is tied to one machine.
#   ./run_all.sh [IMAGE_DIR] [QUERIES]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES=${1:-./inbox}
QUERIES=${2:-"cat,kitten"}
MODEL=${LA_MODEL:-nvidia/LocateAnything-3B}
OUT=${OUT:-./results}
PY=${PY:-python}
mkdir -p "$OUT"

echo "### gate: is the patch live on the real code path, and equivalent?"
# Exit code is meaningful -- if the patch is not actually reached, stop here
# rather than collecting an A/B where both arms run the same code.
"$PY" "$HERE/verify_patch.py" --model "$MODEL"

echo "### packed multi-image branch: does it actually execute?"
"$PY" "$HERE/segcheck.py" --model "$MODEL" || true

echo "### A/B sweep: stock vs fixed"
"$PY" "$HERE/ab_sweep.py" --images "$IMAGES" --queries "$QUERIES" --out "$OUT"

echo "### batch throughput across batch sizes and fix combinations"
"$PY" "$HERE/batchbench.py" --model "$MODEL" --out "$OUT/batchbench.json"

echo "wrote $OUT"
