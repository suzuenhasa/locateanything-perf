#!/usr/bin/env bash
# Install LocateAnything-3B and the runtime fix.
#
#   ./setup.sh <sshhost>    copy this repo to a remote box and install there
#   ./setup.sh              on the machine that has the GPU: install here
#   ./setup.sh --check      verify an existing install, change nothing
#   ./setup.sh --sglang     also build the SGLang serving venv (see below)
#   ./setup.sh --fix-decode patch the model so hybrid transcribes text correctly
#
# Idempotent. Everything lands beside this checkout, so removing that directory
# removes the install. Re-run after a container recycle on an ephemeral box.
set -uo pipefail

_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$_SELF/.." && pwd)"
BASE="${LA_BASE:-$(dirname "$REPO")}"

# ---------------------------------------------------------------- pinned set
# Everything below floats otherwise. `hf download nvidia/LocateAnything-3B` with
# no revision takes whatever HEAD is that day, so a clean install six months
# from now is a different checkpoint from the one every number in the README was
# measured on, and nothing records which. Captured from the working install,
# not guessed:
#
#   python 3.12.3   torch 2.11.0+cu128   torchvision 0.26.0+cu128
#   transformers 4.57.1   model c32291ca
#
# LA_UNPINNED=1 takes current HEAD and latest instead, which is how you find out
# whether a newer stack works -- deliberately, rather than by the calendar.
MODEL_REPO="nvidia/LocateAnything-3B"
MODEL_REV="c32291ca5e996f5a7a485845b4f57a233936bba0"
TRANSFORMERS_VER="4.57.1"
if [ -n "${LA_UNPINNED:-}" ]; then MODEL_REV=""; TRANSFORMERS_VER=""; fi

CHECK=0; SGLANG=0; FIXDECODE=0; REMOTE_HOST=""
for a in "$@"; do
  case "$a" in
    --check)  CHECK=1 ;;
    --sglang) SGLANG=1 ;;
    --fix-decode) FIXDECODE=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) REMOTE_HOST="$a" ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '   \033[31mXX\033[0m  %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------- drive a remote box
if [ -n "$REMOTE_HOST" ]; then
  say "Installing on $REMOTE_HOST"
  command -v rsync >/dev/null || die "rsync needed locally"
  ssh "$REMOTE_HOST" "mkdir -p '$BASE/locateanything-perf'" \
    || die "cannot ssh to '$REMOTE_HOST' — check ~/.ssh/config"
  rsync -az --delete --exclude='.git/' --exclude='__pycache__/' --exclude='archive/' \
    "$REPO/" "$REMOTE_HOST:$BASE/locateanything-perf/"
  ok "sources copied"
  ssh -t "$REMOTE_HOST" "cd '$BASE/locateanything-perf' && LA_BASE='$BASE' bash scripts/setup.sh \
    $([ $CHECK -eq 1 ] && echo --check) $([ $SGLANG -eq 1 ] && echo --sglang) \
    $([ $FIXDECODE -eq 1 ] && echo --fix-decode)"
  exit 0
fi

# ------------------------------------------------------------------- hardware
say "GPU"
command -v nvidia-smi >/dev/null || die "no nvidia-smi; this needs an NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/   /'
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
CAPMAJ="${CAP%%.*}"

# The checkpoint is bf16. Ampere (sm_80) and up have hardware bfloat16; before
# that torch emulates it and every matmul goes through a conversion.
if [ -n "$CAP" ]; then
  if [ "${CAPMAJ:-0}" -ge 8 ] 2>/dev/null; then
    ok "compute capability $CAP (hardware bf16)"
  else
    warn "compute capability $CAP is pre-Ampere: bf16 is emulated and will be slow."
    warn "The fix still applies -- it is about attention memory, not dtype."
  fi
fi

# Measured on an RTX 3090, this checkout, bf16: weights are 7.4 GiB resident.
# Activation cost on top is where the fix earns its keep -- at 10,000 patches
# (about 1180x1650) the shipped code peaks at 15.5 GiB above the weights and the
# fix brings that to 0.6 GiB. That is why a 16 GiB card is fine WITH the fix and
# cannot run a normal photo without it.
if [ "$VRAM" -lt 9000 ]; then
  warn "${VRAM} MiB VRAM. The weights alone are 7.4 GiB, leaving almost nothing"
  warn "for activations -- expect to be limited to small images even with the fix."
elif [ "$VRAM" -lt 14000 ]; then
  ok "${VRAM} MiB VRAM -- enough with the fix (0.6 GiB of activations at 10k patches)."
  warn "Without it the same image wants 15.5 GiB on top of the weights and will OOM."
else
  ok "${VRAM} MiB VRAM"
fi

# ---------------------------------------------------------------------- disk
say "Disk"
AVAIL=$(df -Pm "$BASE" 2>/dev/null | awk 'NR==2{print $4}')
# Only count what still has to be downloaded. Demanding 27 GiB when the venv and
# the weights are already on disk turns a working install into a hard failure.
SGLVENV="${LA_SGLVENV:-$BASE/sglvenv}"
NEED=0
[ -x "$BASE/venv/bin/python" ] || NEED=$((NEED + 7500))
[ -s "$BASE/model/config.json" ] || NEED=$((NEED + 7500))
[ "$SGLANG" -eq 1 ] && [ ! -x "$SGLVENV/bin/python" ] && NEED=$((NEED + 9500))
NEED=$((NEED + 2000))                       # working room
if [ -n "$AVAIL" ]; then
  if [ "$AVAIL" -lt "$NEED" ]; then
    # An existing install has already spent most of that, so a shortfall here is
    # only fatal when there is still something left to download.
    [ "$CHECK" -eq 0 ] && die "only $((AVAIL/1024)) GiB free at $BASE. Need about $((NEED/1024)) GiB:
       venv 7.1, weights 7.3, working room$([ $SGLANG -eq 1 ] && echo ', SGLang venv 9.1')."
    warn "$((AVAIL/1024)) GiB free at $BASE — under the $((NEED/1024)) GiB a fresh install needs."
    warn "Fine for running; a re-download would not fit."
  else
    ok "$((AVAIL/1024)) GiB free at $BASE (need ~$((NEED/1024)))"
  fi
fi

# ---------------------------------------------------------------- interpreter
say "Python"
mkdir -p "$BASE"
# Reuse an interpreter only if it ALREADY has torch and a usable transformers.
# An interpreter with torch but the wrong transformers is usually a conda base
# or a system python; pinning transformers there would change an environment
# that is not ours to modify. Build a venv beside the checkout instead.
PY=""
# LA_PY names an interpreter explicitly. Nothing else is guessed: image-specific
# paths like /venv/main belong to one provider's container and are wrong
# everywhere else, so a venv beside the checkout is the only fallback.
for c in "${LA_PY:-}" "$BASE/venv/bin/python"; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import torch, transformers" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ] && [ -x "$BASE/venv/bin/python" ]; then
  PY="$BASE/venv/bin/python"                 # half-built venv from an interrupted run
fi
if [ -z "$PY" ]; then
  [ "$CHECK" -eq 1 ] && die "no interpreter with torch at $BASE/venv"
  warn "no interpreter with torch — building one at $BASE/venv"
  python3 -m venv "$BASE/venv" \
    || die "python3 -m venv failed — on Debian/Ubuntu: apt-get install -y python3-venv"
  "$BASE/venv/bin/python" -m pip install -q -U pip setuptools wheel \
    || die "could not bootstrap pip inside $BASE/venv"
  PY="$BASE/venv/bin/python"
  ok "created $BASE/venv"
fi
ok "$PY ($("$PY" -V 2>&1))"

# CUDA wheel selection from the driver, not hardcoded. torch's default PyPI
# build targets one CUDA version; a driver older than that cannot load it, and
# the failure arrives several GB into the download as "The NVIDIA driver on your
# system is too old". CUDA 12.x wheels run on any 12.x driver.
DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
TORCH_INDEX="${TORCH_INDEX:-}"
if [ -z "$TORCH_INDEX" ]; then
  case "$DRV_CUDA" in
    13.*|1[4-9].*)     TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    12.[89]|12.1[0-9]) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    12.[0-7])          TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    *) TORCH_INDEX="https://download.pytorch.org/whl/cu128"
       warn "could not read the driver's CUDA version; assuming cu128" ;;
  esac
fi
[ -n "$DRV_CUDA" ] && ok "driver supports CUDA $DRV_CUDA — using ${TORCH_INDEX##*/} wheels"

# ---------------------------------------------------------------------- torch
say "torch"
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "torch not installed in $PY"
  warn "installing torch + torchvision (~3 GB, several minutes)"
  # torchvision MUST come from the same index. The PyPI build carries the right
  # version number but a different CUDA ABI, so pip considers the requirement
  # satisfied while torchvision::nms fails to register -- and AutoProcessor then
  # will not import, with an error that names neither torchvision nor CUDA.
  "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX" \
    || die "torch install failed. Usual causes: disk space, or no wheel published
       for this python/CUDA combination."
fi
TORCH_V="$("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)" \
  || die "torch will not import from $PY"
ok "torch $TORCH_V"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || die "torch cannot see the GPU from $PY"
ok "CUDA visible to torch"

# The nms check, run for real rather than assumed. This is the single most
# common broken install for this model and it presents as an import error two
# steps later.
"$PY" - <<'EOF' || die "torchvision::nms is not registered — torchvision was built against a
       different CUDA than torch. Reinstall BOTH from the same index:
         pip install --force-reinstall torch torchvision --index-url <the same URL>"
import torch, torchvision
from torchvision.ops import nms
nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([0.5]), 0.5)
EOF
ok "torchvision $("$PY" -c 'import torchvision;print(torchvision.__version__)') — nms registered"

# ----------------------------------------------------------------------- deps
say "Model dependencies"
TSPEC="transformers${TRANSFORMERS_VER:+==$TRANSFORMERS_VER}"
CUR_TF="$("$PY" -c 'import transformers;print(transformers.__version__)' 2>/dev/null || true)"
if [ -n "$TRANSFORMERS_VER" ] && [ "$CUR_TF" != "$TRANSFORMERS_VER" ]; then
  [ "$CHECK" -eq 1 ] && die "transformers is ${CUR_TF:-absent}, needs $TRANSFORMERS_VER"
  # 5.x changed _check_and_adjust_attn_implementation's signature and the
  # model's custom code fails to construct against it.
  "$PY" -m pip install -q "$TSPEC" || die "could not install $TSPEC"
fi
ok "transformers $("$PY" -c 'import transformers;print(transformers.__version__)')"

MISSING=()
for m in accelerate huggingface_hub PIL numpy einops timm requests peft; do
  "$PY" -c "import $m" >/dev/null 2>&1 || MISSING+=("$([ "$m" = PIL ] && echo pillow || echo "$m")")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  [ "$CHECK" -eq 1 ] && die "missing: ${MISSING[*]}"
  "$PY" -m pip install -q "${MISSING[@]}" || die "could not install: ${MISSING[*]}"
fi
ok "accelerate, huggingface_hub, pillow, numpy, einops, timm, requests, peft"

# transformers scans the custom code's imports before loading it and refuses on
# any missing name, even though the image path never calls these. decord is the
# exception: it IS load-bearing for video, because torchvision >= 0.19 removed
# io.read_video and the model defaults to that backend.
IMPORT_SCAN=()
for m in cv2 decord lmdb; do
  "$PY" -c "import $m" >/dev/null 2>&1 || IMPORT_SCAN+=("$m")
done
if [ ${#IMPORT_SCAN[@]} -gt 0 ]; then
  if [ "$CHECK" -eq 1 ]; then
    warn "absent: ${IMPORT_SCAN[*]} — transformers' import scan will refuse to load the model"
  else
    for p in opencv-python-headless decord lmdb; do
      "$PY" -m pip install -q "$p" 2>/dev/null || warn "optional install failed: $p"
    done
  fi
fi
"$PY" -c "import cv2, decord, lmdb" >/dev/null 2>&1 \
  && ok "cv2, decord, lmdb (needed for transformers' import scan; decord also for video)" \
  || warn "cv2/decord/lmdb incomplete — the model may refuse to load"

TORCH_AFTER="$("$PY" -c 'import torch;print(torch.__version__)')"
[ "$TORCH_V" = "$TORCH_AFTER" ] || die "a dependency moved torch $TORCH_V -> $TORCH_AFTER"
ok "torch unchanged by the dependency install"

# ------------------------------------------------------------------- the fix
say "locateanything-fix"
if ! "$PY" -c "import locateanything_fix" >/dev/null 2>&1; then
  [ "$CHECK" -eq 1 ] && die "locateanything_fix not importable from $PY"
  "$PY" -m pip install -q -e "$REPO" || die "pip install -e $REPO failed"
fi
"$PY" -c "import locateanything_fix" >/dev/null 2>&1 \
  || die "locateanything_fix still will not import after installing $REPO"
ok "locateanything_fix importable"

# ------------------------------------------------------------------- weights
say "Weights (7.3 GB on a cold box — this is the slow part)"
# Deliberately NOT inherited. A box that already exports HF_HOME (vast.ai images
# point it at /workspace) would put the custom-code cache outside the checkout,
# which breaks "delete the directory and the install is gone" and leaves a stale
# transformers_modules copy behind after the checkout is replaced. LA_HF_HOME
# overrides it for a deliberate shared cache.
export HF_HOME="${LA_HF_HOME:-$BASE/hf}"
MODEL_DIR="$BASE/model"
mkdir -p "$HF_HOME"
if [ "$CHECK" -eq 1 ]; then
  [ -s "$MODEL_DIR/config.json" ] || die "no weights at $MODEL_DIR"
  N=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
  [ "$N" -gt 0 ] || die "$MODEL_DIR has no .safetensors"
  ok "weights present ($N shards, $(du -sh "$MODEL_DIR" | cut -f1))"
else
  if [ ! -s "$MODEL_DIR/config.json" ]; then
    # Progress bars stay on. Sending this to /dev/null makes setup sit silent
    # for minutes on a cold box, which reads as "it is not downloading".
    "$PY" - <<EOF || die "download failed (network? disk? HF_HOME=$HF_HOME)"
from huggingface_hub import snapshot_download
snapshot_download("$MODEL_REPO", local_dir="$MODEL_DIR",
                  revision=("$MODEL_REV" or None))
EOF
  fi
  # Prove it landed rather than trusting the exit code.
  [ -s "$MODEL_DIR/config.json" ] || die "download reported success but $MODEL_DIR/config.json is missing"
  N=$(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l)
  [ "$N" -gt 0 ] || die "download reported success but there are no .safetensors in $MODEL_DIR"
  ok "$MODEL_REPO ($N shards, $(du -sh "$MODEL_DIR" | cut -f1)${MODEL_REV:+, rev ${MODEL_REV:0:8}})"
fi

# trust_remote_code executes a COPY of the model's python under
# HF_HOME/modules/transformers_modules, not the files in $MODEL_DIR. Editing the
# model directory therefore appears to work while the edit never runs. Nothing
# here edits it -- the fix is a runtime monkeypatch -- but a stale copy from an
# older revision is a real failure mode, so say where it is.
MODCACHE="$HF_HOME/modules/transformers_modules"
[ -d "$MODCACHE" ] && ok "custom-code cache at $MODCACHE (delete it if you ever change the model's python)"

# -------------------------------------------------------------- decode patch
# patches/04 is opt-in because it is a real trade, not a free win.
#
# The model's `hybrid` decode mode -- the one its own card recommends -- never
# falls back to AR on text, so it emits six speculatively-decoded tokens of
# prose per forward with no verification and every transcription stutters:
# "traveled traveled to the four corners of earth earth". Boxes are unaffected;
# only text is. patches/04 makes the fallback work.
#
# Correct text costs 3.8x on a page (5.14s -> 19.36s) because text then decodes
# one token per forward. If you are reading text in volume, --sglang is the
# better answer: it transcribes just as correctly at 5.32s. If you only ever
# locate things, you need neither.
PATCH4="$REPO/patches/04-hybrid-ar-fallback-on-text.patch"
DECODE_PATCHED=0
grep -q "text_ar" "$MODEL_DIR/generate_utils.py" 2>/dev/null && DECODE_PATCHED=1
if [ "$FIXDECODE" -eq 1 ]; then
  say "Decode fix (patches/04)"
  if [ "$DECODE_PATCHED" -eq 1 ]; then
    ok "already applied"
  else
    [ "$CHECK" -eq 1 ] && die "patches/04 is not applied; in-process transcription will be garbled.
       Run ./setup.sh --fix-decode"
    [ -f "$PATCH4" ] || die "missing $PATCH4"
    command -v patch >/dev/null || die "'patch' not installed (apt-get install -y patch)"
    ( cd "$MODEL_DIR" && patch -s -p0 < "$PATCH4" ) \
      || die "patches/04 did not apply. The checkpoint's python has moved from the
       pinned revision ${MODEL_REV:0:8}; the change is ~30 lines, see $PATCH4."
    grep -q "text_ar" "$MODEL_DIR/generate_utils.py" \
      || die "patch reported success but generate_utils.py has no text_ar branch"
    ok "applied to $MODEL_DIR"
    # transformers executes a COPY of the model's python from here, so without
    # this the patch is inert and appears to have worked.
    rm -rf "$HF_HOME/modules/transformers_modules"
    ok "cleared the trust_remote_code cache (else the patch never runs)"
  fi
elif [ "$DECODE_PATCHED" -eq 0 ] && [ "$CHECK" -eq 1 ]; then
  warn "patches/04 not applied: in-process OCR text will be garbled (boxes are fine)."
  warn "Either ./setup.sh --fix-decode, or use --sglang and read text through that."
fi

# ------------------------------------------------------------------- SGLang
if [ "$SGLANG" -eq 1 ]; then
  say "SGLang serving venv (optional, ~9 GB)"
  # A separate venv on purpose: SGLang pins torch tightly and installing it
  # beside the working install would move the torch every other script here
  # depends on.
  #
  # Worth having for OCR specifically. The in-process engine decodes six tokens
  # per forward through the parallel box decoder, which is right for coordinate
  # tokens and wrong for prose -- measured on a scanned book page it returns
  # "traveled traveled to the four corners of earth earth". SGLang has no such
  # path (grep it: zero references to n_future or mtp), decodes one token at a
  # time, and transcribed the same page cleanly in 5.3s against 19.2s for the
  # in-process AR mode.
  SGLPY="$SGLVENV/bin/python"
  if [ ! -x "$SGLPY" ]; then
    [ "$CHECK" -eq 1 ] && die "no SGLang venv at $SGLVENV"
    python3 -m venv "$SGLVENV" || die "could not create $SGLVENV"
    "$SGLPY" -m pip install -q -U pip setuptools wheel
  fi
  if ! "$SGLPY" -c "import sglang" >/dev/null 2>&1; then
    [ "$CHECK" -eq 1 ] && die "sglang not installed in $SGLVENV"
    warn "installing sglang (~9 GB, this takes a while)"
    "$SGLPY" -m pip install "sglang[all]" || die "sglang install failed"
  fi
  SGLV="$("$SGLPY" -c 'import sglang;print(sglang.__version__)' 2>/dev/null)"
  ok "sglang $SGLV"

  # Locate the in-tree port. Without it there is nothing to patch and nothing
  # will serve this checkpoint.
  LAFILE="$("$SGLPY" - <<'EOF'
import importlib.util as u
s = u.find_spec("sglang.srt.models.locate_anything")
print(s.origin if s else "")
EOF
)"
  [ -n "$LAFILE" ] && [ -f "$LAFILE" ] \
    || die "this sglang has no sglang/srt/models/locate_anything.py, so it cannot
       serve this checkpoint. Upstream added it in sgl-project/sglang#28958 --
       install a build that includes it."
  ok "LocateAnything in-tree at ${LAFILE##*/site-packages/}"

  # patches/02 is REQUIRED, not cosmetic. SGLang wraps MoonViT in its own
  # VisionAttention, whose submodules are attn.qkv_proj / attn.proj, while the
  # HF checkpoint names them wqkv / wo. Without the rename all 54
  # vision-attention tensors (27 blocks x 2) miss params_dict and stay at random
  # init. The tower still loads its MLPs and norms, so the SERVER COMES UP
  # HEALTHY and every request returns a full-image box <0><0><1000><1000>,
  # because the model never actually sees the image. Verified by reversing it:
  # unpatched, 54 "not found in the checkpoint" warnings on load.
  PATCH="$REPO/patches/02-sglang-locateanything-vision-weights.patch"
  if grep -q 'attn.qkv_proj' "$LAFILE" 2>/dev/null; then
    ok "vision-weight rename already present"
  else
    [ "$CHECK" -eq 1 ] && die "patches/02 is NOT applied to $LAFILE — the vision tower
       would load at random init and every box would be the whole image.
       Run ./setup.sh --sglang to apply it."
    [ -f "$PATCH" ] || die "missing $PATCH"
    command -v patch >/dev/null || die "'patch' not installed (apt-get install -y patch)"
    cp "$LAFILE" "$LAFILE.orig"
    patch -s -p0 "$LAFILE" < "$PATCH" \
      || die "patches/02 did not apply to sglang $SGLV. The upstream file has moved;
       the change itself is two lines -- see $PATCH."
    ok "applied patches/02 (original kept at ${LAFILE##*/}.orig)"
  fi
  # Prove it took, rather than trusting patch's exit code.
  grep -q 'attn.qkv_proj' "$LAFILE" \
    || die "patch reported success but $LAFILE still has no attn.qkv_proj rename"
  ok "vision-weight rename verified in the installed file"
fi

# ------------------------------------------------------------------- verify
say "Verify"
# The only check that matters: is the patch live on the code path the model
# actually takes, and is it numerically equivalent to what it replaced. Loads
# the real weights, so it is slow, and it is the difference between "installed"
# and "working".
if [ "${LA_SKIP_VERIFY:-0}" = "1" ]; then
  warn "skipped (LA_SKIP_VERIFY=1)"
else
  PYTHONPATH="$REPO" "$PY" "$REPO/scripts/verify_patch.py" --model "$MODEL_DIR" 2>&1 \
    | grep -E "torch |model loaded|is_applied|equivalence|PATCH_VERIFIED|Error|Traceback" | sed 's/^/   /'
  PYTHONPATH="$REPO" "$PY" "$REPO/scripts/verify_patch.py" --model "$MODEL_DIR" 2>/dev/null \
    | grep -q PATCH_VERIFIED || die "verify_patch.py did not print PATCH_VERIFIED — the install
       loads but the fix is not taking effect on the real code path."
  ok "PATCH_VERIFIED"
fi

cat > "$BASE/env.sh" <<EOF
# source $BASE/env.sh
export PYTHONPATH="$REPO"
export LA_MODEL="$MODEL_DIR"
export HF_HOME="$HF_HOME"
EOF
ok "wrote $BASE/env.sh"

say "Ready"
cat <<EOF
   source $BASE/env.sh
   source $BASE/venv/bin/activate

   Locate things in photographs:
     python $REPO/scripts/tile_ocr.py --image ./photos --modes whole \\
            --task Detection --category "cats"

   Read a page. Tiled and batched is both faster and finds about twice as much
   as a single whole-page pass:
     python $REPO/scripts/tile_ocr.py --image ./pages --modes whole,tiled-batch

   Serve a directory of pages with the model resident:
     python $REPO/scripts/serve.py --bench ./pages --task OCR

   Reproduce the memory claim on this card, no weights needed for the probe:
     python $REPO/scripts/crossbox.py --model \$LA_MODEL --out box.json

   Flags and traps:      $REPO/scripts/README.md
   Verify any time, changing nothing:
     $REPO/scripts/setup.sh --check
EOF
