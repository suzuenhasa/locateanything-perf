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
#   python 3.12.3   torch 2.11.0+cu128   torchvision 0.26.0+cu128   model c32291ca
#
# torch and transformers are deliberately NOT pinned. transformers used to be,
# to 4.57.1, on the strength of the first error 5.x produced. There are six,
# they are all small, and patches/05 makes every one of them version-agnostic.
#
#   MINIMUMS:  torch >= 2.0     transformers >= 4.51     python >= 3.9
#   MAXIMUMS:  none. Both floors come from upstream, not from this repo.
#
# Verified end to end on this checkpoint, same page, byte-identical output
# (28 boxes, 1621 ref chars) at every point:
#
#     torch 2.6.0+cu124    CUDA 12.4   transformers 4.51.3
#     torch 2.11.0+cu130   CUDA 13.0   transformers 5.12.1
#     torch 2.13.0+cu132   CUDA 13.2   transformers 5.15.1     <- newest that exists
#
# torch 2.0 is where scaled_dot_product_attention arrives, and it is also
# transformers' own declared floor; the checkpoint asserts no torch version at
# all. transformers 4.51 is where models.qwen3 appears, which the checkpoint
# imports at module level for a branch it never takes (its architectures are
# Qwen2ForCausalLM). Make that import lazy and the next wall is 4.45, at
# processing_utils.Unpack.
#
# CONSEQUENCE, and the point of all of it: this script installs as little as it
# can. If the box has a working torch it uses it and downloads no wheels; if
# transformers is at or above the floor it is left alone at whatever version it
# is. The only version it will ever change is a transformers below 4.51, and it
# upgrades to the minimum, not the latest.
#
# LA_UNPINNED=1 takes current HEAD instead of the pinned checkpoint revision,
# which is how you find out whether a newer checkpoint works -- deliberately,
# rather than by the calendar.
MODEL_REPO="nvidia/LocateAnything-3B"
MODEL_REV="c32291ca5e996f5a7a485845b4f57a233936bba0"
if [ -n "${LA_UNPINNED:-}" ]; then MODEL_REV=""; fi

# Minimums, not preferences. Neither of these is ours and neither is a ceiling.
TORCH_MIN="2.0"        # scaled_dot_product_attention, and transformers' own floor
TF_MIN="4.51"          # transformers.models.qwen3, which the checkpoint imports at
                       # module level for a branch it never takes
# A >= B, tolerating "2.11.0+cu130"
ver_ge() {
  local a="${1%%+*}" b="${2%%+*}"
  [ "$(printf '%s\n%s\n' "$b" "$a" | sort -V | head -1)" = "$b" ]
}

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
  # Where to install ON THE REMOTE. LA_BASE if you set it, otherwise that box's
  # home directory -- resolved over there, not here. Defaulting to the local
  # $BASE would mirror this machine's layout onto a box that has never heard of
  # it: run this from ~/projects and the weights land in /root/projects.
  RBASE="${LA_BASE:-$(ssh "$REMOTE_HOST" 'echo "$HOME"' 2>/dev/null)}" \
    || die "cannot ssh to '$REMOTE_HOST' — check ~/.ssh/config"
  [ -n "$RBASE" ] || die "cannot ssh to '$REMOTE_HOST' — check ~/.ssh/config"
  ok "installing into $RBASE on $REMOTE_HOST"
  ssh "$REMOTE_HOST" "mkdir -p '$RBASE/locateanything-perf'" \
    || die "cannot create $RBASE/locateanything-perf on $REMOTE_HOST"
  rsync -az --delete --exclude='.git/' --exclude='__pycache__/' --exclude='archive/' \
    "$REPO/" "$REMOTE_HOST:$RBASE/locateanything-perf/" \
    || die "rsync to $REMOTE_HOST failed"
  ok "sources copied"
  # -t only when there IS a terminal. Forcing it without one ("Pseudo-terminal
  # will not be allocated because stdin is not a terminal") leaves this hanging
  # after the sources have copied, which looks like a slow install and is not.
  TT=(-t); [ -t 0 ] || TT=(-T)
  ssh "${TT[@]}" "$REMOTE_HOST" "cd '$RBASE/locateanything-perf' && LA_BASE='$RBASE' bash scripts/setup.sh \
    $([ $CHECK -eq 1 ] && echo --check) $([ $SGLANG -eq 1 ] && echo --sglang) \
    $([ $FIXDECODE -eq 1 ] && echo --fix-decode)"
  exit $?
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
# Empty means "the interpreter we are already using". A second venv is only
# built if LA_SGLVENV names one -- see the note in the SGLang section.
SGLVENV="${LA_SGLVENV:-}"
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

# -------------------------------------------------------------------- network
# Everything past this point downloads. A box that cannot reach PyPI will build
# a venv, spend a minute retrying, and fail with five screens of urllib3
# backtrace that name neither the cause nor the host -- so check first and say
# which of the two it is, because they have different fixes.
#
# The case this exists for: a rented box whose outbound TCP was entirely fine,
# including to PyPI's CDN, but whose UDP port 53 was filtered. Every hostname
# failed to resolve while every IP was reachable, and pip's error blamed pip.
if [ "$CHECK" -eq 0 ]; then
  say "Network"
  DNSBAD=""; NETBAD=""
  for h in pypi.org files.pythonhosted.org huggingface.co download.pytorch.org; do
    getent hosts "$h" >/dev/null 2>&1 || DNSBAD="$DNSBAD $h"
  done
  if [ -n "$DNSBAD" ]; then
    # Distinguish "no DNS" from "no internet": if a bare IP answers on 443 then
    # routing is fine and only name resolution is broken.
    if timeout 6 bash -c 'echo > /dev/tcp/1.1.1.1/443' 2>/dev/null; then
      die "cannot resolve:$DNSBAD
       Outbound TCP works, so this is DNS alone -- usually UDP/53 filtered on the
       host network. Check /etc/resolv.conf, and if TCP/53 is permitted try
       adding 'options use-vc' to it to force DNS over TCP."
    fi
    die "cannot resolve:$DNSBAD
       and cannot open TCP 443 to 1.1.1.1 either, so this box has no working
       outbound network. Nothing here can be downloaded until that is fixed."
  fi
  ok "pypi.org, files.pythonhosted.org, huggingface.co, download.pytorch.org resolve"
  for hp in pypi.org:443 huggingface.co:443; do
    timeout 8 bash -c "echo > /dev/tcp/${hp%:*}/${hp##*:}" 2>/dev/null \
      || NETBAD="$NETBAD $hp"
  done
  [ -z "$NETBAD" ] || die "resolves but cannot connect to:$NETBAD
       Routing or a firewall, not DNS. A proxy may be required (https_proxy)."
  ok "outbound https to pypi.org and huggingface.co"
fi

# ---------------------------------------------------------------- interpreter
say "Python"
mkdir -p "$BASE"
# Reuse an interpreter that already has torch and transformers, rather than
# downloading several GB of wheels next to a working install.
#
# This used to refuse anything but LA_PY or its own venv, on the grounds that
# pinning transformers into a conda base or a system python would change an
# environment that is not ours to modify. That reasoning died with the
# transformers pin: nothing is installed over a version any more, and the model
# runs on everything from 4.51 to 5.15.1. What is left is additive -- the
# missing few of accelerate/peft/einops/timm/decord/lmdb -- so reuse is now the
# default and the download is the fallback, which is the right way round.
#
# LA_NO_SYSTEM_PY=1 restores the old behaviour if you would rather keep a system
# python untouched.
PY=""
CANDIDATES=("${LA_PY:-}" "$BASE/venv/bin/python")
if [ "${LA_NO_SYSTEM_PY:-0}" != "1" ]; then
  # No image-specific paths -- /venv/main belongs to one provider's container
  # and is wrong everywhere else. Only what is on PATH.
  CANDIDATES+=(python3 python)
fi
for c in "${CANDIDATES[@]}"; do
  [ -z "$c" ] && continue
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c "import torch, transformers" >/dev/null 2>&1; then PY="$c"; break; fi
done
case "$PY" in
  ""|"${LA_PY:-__none__}"|"$BASE/venv/bin/python") ;;
  *) warn "reusing $PY, which already has torch — no wheels will be downloaded."
     warn "Any MISSING model deps get pip-installed into it; LA_NO_SYSTEM_PY=1 to opt out." ;;
esac
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
# system is too old". Wheels run on any driver of the same CUDA major.
#
# This used to send every 13.x driver to cu128, which was written when cu13x
# wheels did not exist yet and left modern boxes two CUDA releases behind their
# own driver. Not every minor gets an index -- cu131 and cu133 are 404 -- so the
# list below is the ones that exist, newest first, and anything unrecognised
# falls back rather than guessing at a URL.
DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
TORCH_INDEX="${TORCH_INDEX:-}"
if [ -z "$TORCH_INDEX" ]; then
  case "$DRV_CUDA" in
    13.[2-9]|13.1[0-9]|1[4-9].*) TORCH_INDEX="https://download.pytorch.org/whl/cu132" ;;
    13.[01])                     TORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
    12.9|12.1[0-9])              TORCH_INDEX="https://download.pytorch.org/whl/cu129" ;;
    12.8)                        TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    12.[0-7])                    TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    *) TORCH_INDEX="https://download.pytorch.org/whl/cu128"
       warn "could not read the driver's CUDA version; assuming cu128" ;;
  esac
  # An index that 404s costs you the whole install several GB in. Check it is
  # actually there, and step back to cu128 -- which has existed throughout --
  # rather than failing.
  if command -v curl >/dev/null 2>&1; then
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m 10 "$TORCH_INDEX/torch/" || echo 000)" != "200" ]; then
      warn "${TORCH_INDEX##*/} is not a live wheel index; falling back to cu128"
      TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    fi
  fi
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
ver_ge "$TORCH_V" "$TORCH_MIN" \
  || die "torch $TORCH_V is below the $TORCH_MIN minimum. The model needs
       scaled_dot_product_attention, which arrived in torch 2.0. Nothing in this
       repo can work around that -- it is the one torch API the checkpoint
       cannot do without.
       There is NO upper bound: 2.13.0 is tested and so is 2.6.0."
ok "torch $TORCH_V (min $TORCH_MIN, no maximum)"
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
# Whatever is already here, as long as there is one. Installing a version over
# the top of a working stack is how you break a box that was fine; patches/05
# covers 4.x and 5.x alike.
CUR_TF="$("$PY" -c 'import transformers;print(transformers.__version__)' 2>/dev/null || true)"
if [ -z "$CUR_TF" ]; then
  [ "$CHECK" -eq 1 ] && die "transformers is not installed"
  "$PY" -m pip install -q transformers || die "could not install transformers"
  CUR_TF="$("$PY" -c 'import transformers;print(transformers.__version__)')"
fi
# Below the floor is the ONLY case where this script touches a transformers that
# is already installed, and it says so rather than doing it quietly.
ver_ge "$CUR_TF" "$TF_MIN" || {
  [ "$CHECK" -eq 1 ] && die "transformers $CUR_TF is below the $TF_MIN minimum"
  warn "transformers $CUR_TF is below the $TF_MIN minimum."
  warn "Not our floor: configuration_locateanything.py and modeling_locateanything.py"
  warn "import transformers.models.qwen3 at module level, for a branch this"
  warn "checkpoint never takes, and that module first ships in 4.51.0."
  warn "Upgrading to the minimum only -- nothing newer is required."
  "$PY" -m pip install -q "transformers>=$TF_MIN" \
    || die "could not upgrade transformers to >=$TF_MIN"
  CUR_TF="$("$PY" -c 'import transformers;print(transformers.__version__)')"
}
ok "transformers $CUR_TF (min $TF_MIN, no maximum — patches/05 spans 4.x and 5.x)"

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
# LA_MODEL points at a checkpoint you already have. Every script in this repo
# reads that variable, and env.sh below exports it, but this script used to
# ignore it -- so anyone with the 7.3 GB already on disk somewhere else
# downloaded it a second time. It is honoured only if it looks like a real
# checkout, so a stale or half-written path fails loudly instead of being
# silently adopted.
MODEL_DIR="$BASE/model"
if [ -n "${LA_MODEL:-}" ]; then
  case "$LA_MODEL" in
    */*)
      if [ -s "$LA_MODEL/config.json" ]; then
        MODEL_DIR="$LA_MODEL"
        ok "using the checkpoint already at $MODEL_DIR — not downloading"
      else
        die "LA_MODEL=$LA_MODEL has no config.json.
       Point it at an unpacked checkpoint directory, or unset it to download
       one to \$LA_BASE/model."
      fi ;;
    *)  # a bare hub id, e.g. nvidia/LocateAnything-3B -- not a local directory
        warn "LA_MODEL=$LA_MODEL is a hub id, not a directory; downloading to $MODEL_DIR" ;;
  esac
fi
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

# ------------------------------------------------------- transformers 5.x
# patches/05 is NOT optional and NOT a trade. Without it the checkpoint's custom
# code cannot run on transformers 5.x at all, and every change in it is guarded
# so 4.x behaves exactly as before. Applying it unconditionally is what lets
# this script stop pinning transformers.
#
# Five of the six defects raise. The sixth does not: the rotary embedding's
# buffers are registered persistent=False, so they are absent from the
# checkpoint, and 5.x materialises them from meta without writing to them. The
# model loads, every weight matches the safetensors, and it emits token soup.
# That is why this is applied rather than offered.
PATCH5="$REPO/patches/05-transformers5-compat.patch"
say "transformers compatibility (patches/05)"
if grep -q "rebuild_buffers" "$MODEL_DIR/modeling_qwen2.py" 2>/dev/null; then
  ok "already applied"
else
  [ "$CHECK" -eq 1 ] && die "patches/05 is not applied. On transformers 5.x the model
       will not construct; if it does construct, the rotary buffers are
       uninitialised and the output is noise. Re-run ./setup.sh."
  [ -f "$PATCH5" ] || die "missing $PATCH5"
  command -v patch >/dev/null || die "'patch' not installed (apt-get install -y patch)"
  ( cd "$MODEL_DIR" && patch -s -p0 < "$PATCH5" ) \
    || die "patches/05 did not apply. The checkpoint's python has moved from the
       pinned revision ${MODEL_REV:0:8}; see $PATCH5 for what each hunk does."
  grep -q "rebuild_buffers" "$MODEL_DIR/modeling_qwen2.py" \
    || die "patch reported success but modeling_qwen2.py has no rebuild_buffers"
  ok "applied to $MODEL_DIR"
  rm -rf "$HF_HOME/modules/transformers_modules"
  ok "cleared the trust_remote_code cache (else the patch never runs)"
fi

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
# Probed whether or not --sglang was passed, so the OCR-path report below can
# tell "no SGLang" from "SGLang present but unpatched" -- two different problems
# with two different answers.
SGLPY_ANY="${LA_SGLVENV:+$LA_SGLVENV/bin/python}"
[ -x "${SGLPY_ANY:-}" ] || SGLPY_ANY="$PY"
if [ "$SGLANG" -eq 1 ]; then
  say "SGLang serving (optional)"
  # ONE venv is enough. This used to build a second one, on the reasoning that
  # sglang pins transformers 5.x while the checkpoint needs 4.57.1 -- so a
  # shared venv would get no model at all, and ~4.9 GB of the ~9 GB install was
  # a second copy of torch and the CUDA runtime.
  #
  # That reasoning died with patches/05: the checkpoint runs on 5.x now.
  # Verified end to end on a box whose sglang was already installed --
  # sglang 0.5.16, torch 2.11.0+cu130, transformers 5.12.1, one venv, both the
  # in-process engine and the server working against the same interpreter.
  #
  # So LA_SGLVENV defaults to the interpreter already in use. Point it somewhere
  # else only if you actually want the two stacks separated.
  #
  # Worth having for OCR specifically. The in-process engine decodes six tokens
  # per forward through the parallel box decoder, which is right for coordinate
  # tokens and wrong for prose -- measured on a scanned book page it returns
  # "traveled traveled to the four corners of earth earth". SGLang has no such
  # path (grep it: zero references to n_future or mtp), decodes one token at a
  # time, and transcribed the same page cleanly in 5.3s against 19.2s for the
  # in-process AR mode.
  if [ -n "$SGLVENV" ]; then
    SGLPY="$SGLVENV/bin/python"
    if [ ! -x "$SGLPY" ]; then
      [ "$CHECK" -eq 1 ] && die "no SGLang venv at $SGLVENV"
      python3 -m venv "$SGLVENV" || die "could not create $SGLVENV"
      "$SGLPY" -m pip install -q -U pip setuptools wheel
    fi
  else
    SGLPY="$PY"
    ok "using the same interpreter as the model ($PY)"
  fi
  if ! "$SGLPY" -c "import sglang" >/dev/null 2>&1; then
    [ "$CHECK" -eq 1 ] && die "sglang not importable from $SGLPY"
    warn "installing sglang (~9 GB, this takes a while)"
    # Never into the interpreter that already has a working torch: sglang's
    # exact pins would replace it, which is how a box that was fine stops
    # being fine. Ask for a separate venv instead.
    [ -n "$SGLVENV" ] || die "sglang is not installed in $PY, and installing it
       here would pull its own torch over the one that works. Either use an
       image that ships sglang, or set LA_SGLVENV=/path/to/venv and re-run."
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

  # patches/02 renames the checkpoint's vision-attention tensors (wqkv / wo) to
  # the names SGLang's own MoonViT used (attn.qkv_proj / attn.proj). Get this
  # wrong in either direction and all 54 of them (27 blocks x 2) miss
  # params_dict and stay at random init -- and because the tower still loads its
  # MLPs and norms, THE SERVER COMES UP HEALTHY and every request returns a
  # full-image box <0><0><1000><1000>, since the model never sees the image.
  #
  # Which direction is right depends on the sglang version, so detect it rather
  # than assume. Newer builds name their own modules wqkv / wo, matching the
  # checkpoint, and there upstream has already fixed this -- applying patches/02
  # renames the tensors AWAY from the correct names. It still applies cleanly,
  # which is the trap. Measured on sglang 0.5.16: patched, 108 parameters did
  # not receive weights; reverted, 0.
  MOONVIT="$("$SGLPY" - <<'EOF'
import importlib.util as u
s = u.find_spec("sglang.srt.models.kimi_vl_moonvit")
print(s.origin if s else "")
EOF
)"
  PATCH="$REPO/patches/02-sglang-locateanything-vision-weights.patch"
  if [ -n "$MOONVIT" ] && grep -q 'self\.wqkv' "$MOONVIT" 2>/dev/null; then
    ok "sglang $SGLV names its vision modules wqkv/wo — patches/02 not needed"
    if grep -q 'attn.qkv_proj' "$LAFILE" 2>/dev/null; then
      [ "$CHECK" -eq 1 ] && die "patches/02 IS applied to $LAFILE, and this sglang does
       not want it: the vision tensors are renamed away from the names the
       model actually uses, 108 parameters load at random init, and the server
       comes up healthy returning whole-image boxes. Restore ${LAFILE##*/}.orig."
      [ -f "$LAFILE.orig" ] \
        && { cp "$LAFILE.orig" "$LAFILE"; ok "reverted patches/02 (this sglang does not need it)"; } \
        || die "patches/02 is applied to $LAFILE but there is no .orig to restore.
       Reinstall sglang, or reverse it by hand: patch -R -p0 $LAFILE < $PATCH"
    fi
  else
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
    grep -q 'attn.qkv_proj' "$LAFILE" \
      || die "patch reported success but $LAFILE still has no attn.qkv_proj rename"
    ok "vision-weight rename verified in the installed file"
  fi

  # patches/06. SGLang's kimi_vl_moonvit.py is a verbatim vendoring of the
  # checkpoint's modeling_vit.py, so it carries the same 3-D SDPA defect
  # patches/01 fixes -- and locateanything_fix.apply() cannot reach it, because
  # the server is a different process running a different module. Unpatched, the
  # server loads clean, reports healthy, and dies on the first full-resolution
  # page trying to allocate 10.62 GiB, taking the whole server down with it.
  # Worse here than in-process: --mem-fraction-static has already claimed most
  # of the card before the vision tower asks for scratch.
  PATCH6="$REPO/patches/06-sglang-moonvit-sdpa-4d.patch"
  if [ -z "$MOONVIT" ]; then
    warn "no kimi_vl_moonvit.py in this sglang; skipping patches/06"
  elif grep -q 'LA_VIT_FASTMASK' "$MOONVIT" 2>/dev/null; then
    ok "patches/06 already applied"
  else
    [ "$CHECK" -eq 1 ] && die "patches/06 is NOT applied to $MOONVIT — the server will
       OOM and exit on the first full-resolution page. Run ./setup.sh --sglang."
    [ -f "$PATCH6" ] || die "missing $PATCH6"
    cp "$MOONVIT" "$MOONVIT.orig"
    patch -s -p0 "$MOONVIT" < "$PATCH6" \
      || die "patches/06 did not apply to sglang $SGLV. See $PATCH6; it is the same
       change patches/01 makes to the checkpoint."
    grep -q 'LA_VIT_FASTMASK' "$MOONVIT" \
      || die "patch reported success but $MOONVIT has no LA_VIT_FASTMASK"
    ok "applied patches/06 (original kept at ${MOONVIT##*/}.orig)"
  fi
  warn "launch the server with LA_VIT_FASTMASK=1, or patches/06 stays inert:"
  warn "  LA_VIT_FASTMASK=1 $SGLPY -m sglang.launch_server --model-path $MODEL_DIR \\"
  warn "      --trust-remote-code --port 30000 --mem-fraction-static 0.80"
fi

# ---------------------------------------------------------------- OCR path
# Always reported, --sglang or not. Which OCR path you are on is a real
# performance decision and it is invisible otherwise: nothing errors without
# SGLang, pages just take an order of magnitude longer, and you would have no
# reason to suspect it.
#
# Measured here, 9 distinct page scans, whole-page OCR, one RTX 3090:
#
#     in-process, patches/04 applied      160.7 s    17.90 s/page
#     SGLang, one page at a time           54.7 s     6.08 s/page
#     SGLang, 6 concurrent                 12.9 s     1.44 s/page
#     SGLang, 9 concurrent                 12.1 s     1.35 s/page
#
# The first gap is per-request: the model's decode loop runs at 21% of this
# card's memory-bandwidth roofline where SGLang runs at 73%. The second is that
# the in-process engine serves pages one after another, while SGLang overlaps
# them -- batching tiles WITHIN a page it already does, batching ACROSS pages it
# cannot. That is the part you lose without it.
say "OCR path"
if "$SGLPY_ANY" -c "import sglang" >/dev/null 2>&1; then
  if [ "$SGLANG" -eq 1 ]; then
    ok "SGLang patched — OCR in volume ~1.35 s/page at 9 concurrent"
  else
    warn "SGLang IS installed here, but this run did not patch it."
    warn "Re-run with --sglang to apply patches/02 (version-gated) and patches/06."
    warn "Unpatched it will either serve whole-image boxes or OOM on the first page."
  fi
else
  ok "no SGLang — in-process OCR, which is the correct-but-slower path"
  warn "OCR will run at ~17.9 s/page, and pages are served ONE AT A TIME."
  warn "SGLang does the same 9 pages in 12.1 s total (~1.35 s/page) because it"
  warn "overlaps requests; batching across pages is the part you do not have."
  warn "Boxes are identical either way, and detection/grounding/pointing/GUI are"
  warn "unaffected — this only matters if you are reading text in volume."
  warn ""
  warn "If you want it: install SGLang (its own venv, it pins its own torch),"
  warn "then re-run  ./setup.sh --sglang  to apply the patches to it."
  warn "  LA_SGLVENV=$BASE/sglvenv ./setup.sh --sglang"
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
# $PY is whatever interpreter was actually used -- the venv this script built,
# or one that was already here (LA_PY). Printing $BASE/venv unconditionally sent
# people to activate something that may not exist and is not what ran.
ACTIVATE="${PY%/bin/python}/bin/activate"
cat <<EOF
   source $BASE/env.sh
$([ -f "$ACTIVATE" ] && echo "   source $ACTIVATE" || echo "   (interpreter: $PY)")

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
