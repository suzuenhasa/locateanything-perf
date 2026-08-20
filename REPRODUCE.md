# Reproducing the measurements

The repo's own `scripts/` cover install and the A/B sweep. This is the
cross-machine benchmark used for the numbers in the README, which lives in the
archive rather than the repo (it is evidence tooling, not tooling for using the
fix).

## Cross-machine comparison (`crossbox.py`)

One command per machine, output directly comparable:

```bash
# env: torch cu128, transformers==4.57.1, NO flash-attn
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.1" accelerate huggingface_hub pillow numpy \
    einops timm requests peft opencv-python-headless lmdb decord

export HF_HOME=/root/la/hf
hf download nvidia/LocateAnything-3B --local-dir /root/la/model

export PYTHONPATH=/root/la     # so locateanything_fix is importable
flock /root/.gpu.lock python -u crossbox.py --model /root/la/model --out box.json
```

`decord` is **not optional** — transformers scans the custom code's imports
before loading and refuses on any missing name, even though the image path never
calls it. Same for `cv2` and `lmdb`.

It prints two tables: a kernel probe (3-D vs 4-D SDPA, no weights needed) and the
end-to-end run through the model, keyed to patch counts so results from different
machines line up.

## Results collected

| box | file |
|---|---|
| RTX 3090 (sm_86) | `archive/results/rtx3090.json` |
| H100 PCIe (sm_90), Aug 20 | `archive/results/h100_2026.json` |
| A100 80GB (sm_80) | `archive/results/a100_2026.json` |

Every memory figure matches across all three, including the OOM boundary at
39.06 GiB. That byte-identical agreement is the check that the protocol measures
what it claims to.

## The ten-second version, no weights

```bash
python sdpaprobe.py     # archive/measurement/ — 12.36 GiB -> 0.13 GB
```

Reproduces the defect from geometry alone (16 heads, head_dim 72, bf16) with no
model download. Because the geometry is MoonViT's, this also demonstrates the
same defect in Kimi-VL and NeMo AutoModel's ports.

## Traps

- **Patching the model directory does nothing on its own.** `trust_remote_code`
  executes a copy under `HF_HOME/modules/transformers_modules/`. Clear it, or the
  edit never runs and you get a perfectly clean null result.
- **`apply()` mutates module state.** Any A/B running both arms in one process
  must call `revert()` between them, or arm 2 inherits arm 1.
- **`do_sample=False` never terminates** on this model's MTP paths — it loops
  `<box><0><0><1000><1000></box>` to the token cap, so every image takes an
  identical time-to-cap and the result table looks clean and is fake.
- **Use a control arm.** Run shipped-vs-shipped as a third arm; if it is not
  exactly 0.0000, the comparison is uncalibrated.
