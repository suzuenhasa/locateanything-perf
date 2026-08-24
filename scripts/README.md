# Runtime scripts

What each script takes and what each flag does. The numbers behind any claim
here are in [MEASUREMENTS.md](../MEASUREMENTS.md); install and first run are in
[README.md](../README.md).

Every script takes `--model` (or `LA_MODEL`); nothing is hardcoded to a machine.
`locateanything_fix` must be importable — `pip install -e .` or
`PYTHONPATH=/path/to/locateanything-perf`.

| script | what it does |
|---|---|
| [`setup.sh`](#setupsh) | check the machine, build or reuse an environment, fetch the checkpoint, apply the patches, verify |
| [`verify_patch.py`](#verify_patchpy) | prove the fix is live on the real code path and numerically equivalent. Exit 0/1 |
| [`serve.py`](#servepy) | resident engine: load, patch, compile and warm once, then serve pages warm |
| [`tile_ocr.py`](#tile_ocrpy) | tile a page, run the tiles as one batch, stitch the boxes back |
| [`crossbox.py`](#crossboxpy) | one command per machine, output directly comparable across cards |
| `la_common.py` | prompt templates, output parsing, resize maths. Imported by the two above; not run directly |

## Requirements

```
MINIMUMS:  torch >= 2.0     transformers >= 4.51     python >= 3.9
MAXIMUMS:  none
```

Both are upstream's, not this repo's. Tested on torch 2.6–2.13 and transformers
4.51–5.15.1 with byte-identical output; see
[Versions](../MEASUREMENTS.md#versions) for where each floor comes from.

---

## `setup.sh`

```bash
bash scripts/setup.sh                 # environment, checkpoint, patches/05, verify
bash scripts/setup.sh --check         # verify an existing install, change nothing
bash scripts/setup.sh --fix-decode    # also apply patches/04
bash scripts/setup.sh --sglang        # patch an SGLang you already have
bash scripts/setup.sh <sshhost>       # copy this checkout to a remote machine and run there
```

Checks compute capability, VRAM, disk, DNS and outbound HTTPS, and the driver's
CUDA version before installing anything. Ends by running `verify_patch.py` and
refuses to claim success unless it prints `PATCH_VERIFIED`. Writes
`<BASE>/env.sh`, which exports `PYTHONPATH`, `LA_MODEL` and `HF_HOME`.

| mode | effect |
|---|---|
| *(none)* | environment, checkpoint, `patches/05`, verify |
| `--check` | read-only. Exits non-zero on any problem, so it works as a CI gate |
| `--fix-decode` | applies `patches/04`, which makes in-process transcription correct. Costs ~2x on text; boxes are unaffected |
| `--sglang` | applies `patches/06`, and `patches/02` **only if your SGLang version wants it**. Never installs SGLang |
| `<sshhost>` | rsyncs the checkout and re-runs itself there. Other flags pass through |

### What it will and will not install

- **torch already present** → used as-is at whatever version. It only installs
  torch if there is none, and picks the wheel index from your driver's CUDA
  version (cu132/130/129/128/126), HEAD-checking it first.
- **transformers at or above 4.51** → left alone. The only version it will ever
  change is one below the floor, and it upgrades to the minimum, not the latest.
- **an interpreter that already has both** → reused rather than building a venv
  and downloading several GB next to a working install.
- **SGLang** → never installed. Its pins would replace your torch. `--sglang`
  patches one that is already importable; if there is none, setup says so.
- **has to fetch**: the 7.3 GB checkpoint, and whichever of `accelerate peft
  einops timm decord lmdb opencv-python-headless` are missing.

| variable | effect |
|---|---|
| `LA_PY` | use this interpreter |
| `LA_BASE` | where the venv, model and `env.sh` go (default: the checkout's parent) |
| `LA_MODEL` | a checkpoint you already have — skips the download |
| `LA_HF_HOME` | share an existing HF cache |
| `LA_SGLVENV` | build a separate SGLang venv here, and install SGLang into it |
| `LA_NO_SYSTEM_PY` | never reuse an interpreter found on `PATH` |
| `LA_UNPINNED` | take the checkpoint's current HEAD instead of the pinned revision |
| `LA_SKIP_VERIFY` | skip the final `verify_patch.py` |
| `TORCH_INDEX` | override the wheel index chosen from the driver |

---

## `verify_patch.py`

```bash
python scripts/verify_patch.py --model "$LA_MODEL"
```

Loads the real weights, calls the vision attention both ways on the same input,
and prints `PATCH_VERIFIED` only if the patched path is live *and* numerically
equivalent. Exit 0/1. This is the difference between "installed" and "working" —
nothing else in the repo checks the real code path.

| flag | default | what |
|---|---|---|
| `--model` | `$LA_MODEL` or the hub id | checkpoint path or hub id |
| `--seq` | `1024` | sequence length for the equivalence check |
| `--tol` | `0.05` | max abs diff allowed (bf16) |

---

## `serve.py`

```bash
python scripts/serve.py --bench ./pages --task OCR
python scripts/serve.py --bench ./pages --compile --warmup 6 --temperature 0
python scripts/serve.py --bench ./pages --no-compile --temperature 0     # control
```

Keeps the model resident across requests, so the load, patch, compile and warmup
are paid once. Tiles each page and runs the tiles as one batch.

| flag | default | what |
|---|---|---|
| `--model` | `$LA_MODEL` or the hub id | checkpoint path or hub id |
| `--bench` | *(none)* | directory of images to process in sequence |
| `--task` | `OCR` | see [tasks](#tasks) |
| `--category` | `text` | what to locate. Ignored for `--task OCR` |
| `--tiles` | `8` | tiles per page |
| `--tile-px` | `448x448` | tile size, used only by `--tile-mode pad` |
| `--tile-mode` | `raw` | `raw` feeds the crop at full resolution; `pad` squeezes it into `--tile-px` |
| `--page-patches` | `10000` | patch budget for the whole page before tiling |
| `--max-new-tokens` | `1024` | per tile |
| `--compile` / `--no-compile` | `--compile` | `torch.compile` the language model |
| `--warmup` | `0` | synthetic pages at startup so dynamo finishes tracing before the first real request |
| `--rounds` | `2` | passes over the directory; round 2+ is the warm number |
| `--temperature` | `0.0` | 0 is greedy |
| `--strict` | off | after warmup, make any further recompile raise |
| `--no-hostsync` | off | disable the host-sync fixes (control arm) |
| `--out` | *(none)* | write per-request JSON here |

### Notes

**`--compile` is paid on every process start, not once per machine.** It traces
in memory; what persists is the on-disk inductor cache (~94 MB at
`/tmp/torchinductor_$USER`), which halves the trace but does not remove it —
388 s cold, 181 s warm, against 7.2 s for `--no-compile`. It is 1.33x per
forward, so it pays back at ~78 pages in one process once the cache is warm. If
`/tmp` is not persistent where you run, set `TORCHINDUCTOR_CACHE_DIR` somewhere
that is. [Full numbers](../MEASUREMENTS.md#reading-text--tiling-serving-and-torchcompile).

**`--warmup` feeds *heterogeneous* pages** — each tile region a different line
count, across three aspect ratios. The dimension that varies is the batch, and it
varies during decode as tiles retire at `im_end`, so uniform warmup never traces
B = 5,4,3,2,1. With density-only warmup the first real request still spent 101.7 s
tracing; with this, recompiles are zero from round 0.

**`--temperature 0` for any A/B.** Sampling makes the same page emit 39–112
boxes, which drowns any comparison on wall seconds. Greedy is bit-reproducible
here: identical box and forward counts every round.

**Watch `gave_up`, not `graphs`.** `counters["stats"]["unique_graphs"]` freezes
when a code object hits `cache_size_limit` and then runs eager forever, so a
delta of zero is indistinguishable from total failure.

---

## `tile_ocr.py`

```bash
python scripts/tile_ocr.py --image ./page.jpg --grid 3x3
python scripts/tile_ocr.py --image ./pages/ --modes whole,tiled-batch --patches 10000
python scripts/tile_ocr.py --image ./photos --modes whole --task Detection --category "cats"
```

Cuts the page into a grid, hands the tiles to the batch engine at once, and
remaps the boxes back to page coordinates. The page is resized **once** to
`--patches` and the tiles are crops of that same resized image, so every arm sees
identical pixels.

| flag | default | what |
|---|---|---|
| `--image` | **required** | a file, or a directory of them |
| `--model` | `$LA_MODEL` or the hub id | checkpoint path or hub id |
| `--model-dir` | *(none)* | model python to import from, if separate from `--model` |
| `--task` | `OCR` | see [tasks](#tasks) |
| `--category` | `all the objects` | what to locate. Ignored for `--task OCR` |
| `--grid` | `3x3` | tile grid, `NxM` |
| `--patches` | `10000` | patch budget for the **whole page**; tiles are crops of it |
| `--overlap` | `48` | tile overlap in px |
| `--iou` | `0.55` | dedupe threshold across tile seams |
| `--modes` | `whole,tiled-seq,tiled-batch` | which arms to run |
| `--batch` | `0` | tiles per batch; 0 means all of them in one |
| `--max-new-tokens` | `2048` | per tile |
| `--seed` | `1234` | |
| `--repeats` | `1` | run each mode N times |
| `--out` | `./tile_results` | JSONL of every run |

| mode | what it is |
|---|---|
| `whole` | one call on the whole page |
| `tiled-seq` | the tiles, one call each, in sequence. A **control** — it isolates tiling from batching and should be no faster than `whole` |
| `tiled-batch` | the tiles, one batched call. This is the one that wins |

### Notes

**Tiling is a region-count effect, not a resolution effect.** The model emits one
decode block per region it finds, so latency tracks how many regions land in one
forward stack, not how many pixels went in. Splitting the page lets several small
stacks run as one batch instead of one long serial one.

**The direction is corpus-dependent, and the counts tell you which case you are
in.** On pages the whole-page pass already reads competently, tiling costs about
the same wall clock and roughly doubles the regions. On a page it cannot read — a
dense newspaper front page where the whole-page pass returns five boxes — tiling
costs 10x the time and returns 30–60x the regions. Wall clock alone will call
that a regression; it is the whole-page arm giving up.

**Ask for patches, not a short side.** The processor rescales anything over
`in_token_limit` (25,600 patches, `image_processing_locateanything.py:52`), and
the short side that saturates that budget depends entirely on aspect ratio:

| image | aspect | short side at 25,600 patches |
|---|---|---|
| a 3.7:1 results table | 1500x407 | **1166 px** |
| a 1.7:1 photo | 2048x1206 | 1718 px |
| a 3:4 page | 1200x1600 | 1939 px |

So a short-side ladder is not comparable across images: ask two of those for
1680 and 2240 and you get the same clamped run twice, in a table that looks fine.

**Use plurals in `--category`.** A singular noun returns exactly one instance per
frame no matter how many are present. Over 30 frames each containing two kittens:
`kitten` → 1.00 box/frame (alternating between the two animals), `kittens` →
1.93, `cats` → 2.00, `all the kittens` → **2.00 on every frame**. `all the ...`
matches the model card's own prompt form, which is presumably why it is the most
stable. Easy to misread as the model failing to detect something.

---

## `crossbox.py`

```bash
python scripts/crossbox.py                                # kernel probe only, needs just torch
python scripts/crossbox.py --model "$LA_MODEL" --out box.json   # also end-to-end
```

| flag | default | what |
|---|---|---|
| `--model` | *(none)* | omit for the kernel probe only — no weights needed |
| `--out` | `crossbox.json` | |

One command per machine, emitting the same measurements in the same shape, so a
3090 result and an H100 result sit side by side without asterisks. Records
driver, torch, CUDA and compute capability alongside every number.

Two protocol decisions, both easy to get wrong:

- **Equal patch counts, not equal pixel sizes.** The defect scales with
  patches², so "a 1680px image" is different work at different aspect ratios.
  Every arm is defined by a target patch count, and it reports the count achieved.
- **Peak memory *above* resident weights.** Absolute peak conflates the model
  (identical everywhere) with the activation cost (the thing being measured).

[Results across three cards](../MEASUREMENTS.md#kernel-probe-reproduced-on-three-cards),
including a byte-identical OOM boundary at 39.06 GiB.

---

## Tasks

`--task` selects a prompt template. All except `OCR` interpolate `--category`.

| task | prompt |
|---|---|
| `Detection` | Locate all the instances that matches the following description: … |
| `Grounding` | Locate all the instances that match the following description: … |
| `GroundingOne` | Locate a single instance that matches the following description: … |
| `OCR` | Detect all the text in box format. |
| `GUI` | Locate the region that matches the following description: … |
| `Pointing` | Point to: … |
| `TextGrounding` | Please locate the text referred as … |
| `Layout` | Locate all the instances that matches the following description: … |

Multiple categories are joined with `</c>`, matching the demo: `--category
"title, paragraph, table"`. `Pointing` returns 2 coordinates per hit instead of
4. A category the model cannot find comes back as `<box>None</box>`, which
`la_common.parse_mixed_results` surfaces as `{"type": "none"}` rather than
dropping — otherwise a refusal is indistinguishable from a category nobody asked
for.

---

## Traps

Ways to get a clean-looking result that is entirely fake.

- **A model that loads is not a model that works.** On transformers 5.x without
  `patches/05`, the rotary buffers come back as uninitialised memory. Nothing
  raises: the model loads, every weight matches the safetensors byte for byte,
  and the output is token soup. `verify_patch.py` is the only check that runs the
  real code path.
- **An SGLang server that reports healthy is not one that sees the image.** Get
  `patches/02` wrong in either direction and 54 vision tensors stay at random
  init while the tower still loads its MLPs and norms; every box comes back as the
  whole image. Grep the launch log for `not found in params_dict` and `did not
  receive weights` — both should be zero.
- **Patching the model directory does nothing on its own.** `trust_remote_code`
  executes a *copy* under `$HF_HOME/modules/transformers_modules/`. Clear it, or
  the edit never runs and you get a perfectly clean null result.
- **`do_sample=False` never terminates.** Greedy never emits the End block on the
  MTP paths (`hybrid`/`fast`) and loops `<box><0><0><1000><1000></box>` to
  `max_new_tokens`. Every image then takes an identical "time to reach the cap",
  which looks like a clean result table and is entirely fake. `slow` is
  unaffected; `serve.py`'s `--temperature 0` handles this correctly.
- **`apply()` mutates module state.** Anything running a stock arm and a fixed arm
  in one process must `revert()` between them, or arm 2 inherits arm 1.
- **`is_applied()` is marker-identity based.** Any wrapper around
  `VL_VISION_ATTENTION_FUNCTIONS["sdpa"]` — profiler, logger, counter — drops the
  `_locateanything_sdpa_4d` attribute, so `is_applied()` goes False and
  `enable_packed_vision()` refuses with "call apply() first". The fix still
  *runs*; only the gating breaks. Copy the marker onto any wrapper.
- **The in-process OCR text is wrong unless you applied `patches/04`.** Boxes are
  right in every mode; the transcription is not. See
  [the decode-mode trap](../MEASUREMENTS.md#ocr-text-quality--the-decode-mode-trap).
- **Use a control arm.** Run shipped-vs-shipped as a third arm; if it is not
  exactly 0.0000, the comparison is uncalibrated.
- **Video past ~20 frames returns an empty string with no error.** The
  `{"type": "video"}` message path exceeds the model's 16,384-token context and
  `generate()` returns `""` — status ok, zero boxes, indistinguishable from
  "found nothing". Decode to frames and run each as an image, which is what
  NVIDIA's own Space does.
