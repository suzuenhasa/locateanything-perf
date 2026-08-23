# Runtime scripts for `locateanything-perf`

All confirmed working on an RTX 3090 (24 GB), torch 2.11.0+cu128,
transformers 4.57.1, `nvidia/LocateAnything-3B` in bf16.

Every script takes `--model` (or `LA_MODEL`); nothing is hardcoded to a machine.
`locateanything_fix` must be importable -- `pip install -e .` or
`PYTHONPATH=/path/to/locateanything-perf`.

## Scripts

| script | what it does |
|---|---|
| `setup.sh` | venv + torch + pinned deps + `pip install -e` the patch + model download. `./setup.sh [BASE_DIR] [PATCH_REPO_DIR]` |
| `verify_patch.py` | patch is live on the real code path, and numerically equivalent to the original. Exit 0/1, so it works as a CI gate |
| `serve.py` | resident engine: load, patch, compile and warm once, then serve pages warm |
| `tile_ocr.py` | tile a page, OCR the tiles as one batch, stitch the boxes back |
| `crossbox.py` | one command per machine, output directly comparable: the kernel probe (no weights needed) plus the end-to-end run |
| `la_common.py` | prompt templates, output parsing, resize maths. Imported by the two above; not run directly |

That is the whole runtime, plus `crossbox.py` -- which is measurement, but it is
the measurement anyone verifying the claim will want to run first, so it ships.
The 28 scripts that produced the rest of the numbers -- A/B sweeps, decode-budget
probes, the task matrix, the SGLang bench, the video pipeline -- are evidence
tooling, not tooling for using the fix, and stay out of the repo. Nothing here
imports them; the handful of helpers they shared with `serve.py` and `tile_ocr.py`
is now `la_common.py`.

## Typical use

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
bash locateanything-perf/scripts/setup.sh          # BASE defaults to the repo's parent

source venv/bin/activate
export PYTHONPATH="$PWD/locateanything-perf"
export LA_MODEL="$PWD/model"                       # or just use the hub id

python locateanything-perf/scripts/verify_patch.py --model "$LA_MODEL"

# a directory of pages, model stays resident
python locateanything-perf/scripts/serve.py --bench ./inbox --task OCR

# one dense page, tiled
python locateanything-perf/scripts/tile_ocr.py --image ./page.jpg --grid 3x3
```

## Serving pages warm (`serve.py`)

```bash
python serve.py --bench ./inbox --compile --warmup 6 --temperature 0
python serve.py --bench ./inbox --no-compile           --temperature 0   # control
python serve.py --bench ./inbox --compile --warmup 6 --strict            # staging
```

| | s/page | ms/forward | startup |
|---|---|---|---|
| eager | 5.61 | 50.59 | 7 s |
| compiled | **4.68** | **39.82** | 178 s |

Break-even is ~185 pages. For fewer than that in one process, pass `--no-compile`.

Three flags exist because without them the benchmark lies:

- **`--temperature 0`.** Sampling makes the same page emit 39-112 boxes. No A/B
  on wall seconds survives that. Greedy is bit-reproducible: identical box and
  forward counts on every round.
- **`--tile-mode raw`** (the default). Tiles used to be padded to a canonical
  448x448 for "shape stability", which was wrong -- the vision tower runs
  outside the compiled region, so the padding stabilised nothing and cost 30%
  linear resolution. Raw crops are the same speed and find 12-18% more regions.
  `pad` is kept only for comparison.
- **`--strict`.** Sets dynamo's `fail_on_recompile` after warmup, so a shape
  warmup missed raises instead of costing a live request 100 seconds.

`--warmup N` feeds *heterogeneous* synthetic pages -- each tile region a
different line count, across three aspect ratios. That matters: the dimension
that varies is the batch, and it varies during decode as rows retire at
`im_end`, so uniform warmup pages never trace B = 5,4,3,2,1. With density-only
warmup the first real request still spent 101.7 s tracing; with this, recompiles
are zero from round 0.

Watch `gave_up`, not `graphs`. `counters["stats"]["unique_graphs"]` freezes when
a code object hits `cache_size_limit` and then runs eager forever, so a delta of
zero is indistinguishable from total failure.


## Tiling a page (`tile_ocr.py`)

```bash
python tile_ocr.py --image ./page.jpg --grid 3x3 --modes whole,tiled-batch
python tile_ocr.py --image ./pages/  --grid 2x3 --patches 10000 --overlap 48
```

Tiling is the single largest lever on dense pages, and it is not a resolution
effect -- it is a region-count effect. The model emits one decode block per
region it finds, so latency tracks how many regions land in one forward stack,
not how many pixels went in. Splitting a page into a grid lets six small stacks
run as one batch instead of one long serial stack.

| A4 page, 2x3 | s | boxes |
|---|---:|---:|
| whole | 9.85 | 45 |
| tiled-batch | **7.04** | **84** |

Faster *and* it finds 87% more regions, because each tile gets more of the
25,600-patch budget spent on its own text. On the densest pages in the corpus
the gap reaches 2.9x (14.20 s whole, 4.94 s tiled).

`--overlap 48` and `--iou 0.55` exist because boxes straddling a tile seam
otherwise arrive twice, once truncated. Raise the overlap for pages with wide
tables; the dedupe is IoU over the stitched-back page coordinates, so it costs
nothing at inference time.

### Ask for patches, not a short side

`--short-side` is a trap above 1024. The processor rescales anything over
`in_token_limit` (25,600 patches, `image_processing_locateanything.py:52`), and
the short side that saturates that budget depends entirely on aspect ratio:

| image | aspect | short side at 25,600 patches |
|---|---|---|
| a 3.7:1 results table | 1500x407 | **1166 px** |
| a 1.7:1 photo | 2048x1206 | 1718 px |
| a 3:4 page | 1200x1600 | 1939 px |

Ask for `--short-side 1680,2240` on the table and you get the same clamped run
twice, in a table that looks fine. `--patches 5476,10000,14400,25600` puts every
aspect ratio on the same rung and lines up with the top-level README's rows.

## Use plurals in your query

A singular noun returns exactly one instance per frame no matter how many are
present. Measured over 30 frames that each contained two kittens:

| query | boxes per frame |
|---|---|
| `kitten` | 1.00 — one box every frame, alternating between the two animals |
| `kittens` | 1.93 |
| `cats` | 2.00, but with a 1 and a 3 in there |
| `all the kittens` | **2.00 on every frame** |

`all the ...` matches the model card's own prompt form ("Locate all the instances
that matches the following description: ..."), which is presumably why it is the
most stable. Easy to misread as the model failing to detect something.


## Reproducing the claim on another card (`crossbox.py`)

```bash
python crossbox.py                                  # kernel probe only -- needs just torch
python crossbox.py --model "$LA_MODEL" --out box.json   # also end-to-end
```

One command per machine, emitting the same measurements in the same shape, so a
3090 result and an H100 result sit side by side without asterisks. It records
driver, torch, CUDA and compute capability alongside every number, because "does
this reproduce on newer torch" and "does 80 GB hide it" are both live questions
that need the stack pinned to the figure.

Two things it does deliberately, both of which are easy to get wrong:

- **Equal patch counts, not equal pixel sizes.** The defect scales with patches^2,
  so "a 1680px image" is different work at different aspect ratios. Every arm is
  defined by a target patch count, and the script reports the count it achieved.
- **Peak memory *above* resident weights.** Absolute peak conflates the model
  (identical everywhere) with the activation cost (the thing being measured), and
  cards with more VRAM let the allocator behave differently.

Every memory figure matched across a 3090, an H100 PCIe and an A100 80GB,
including the OOM boundary at 39.06 GiB. That byte-identical agreement across
three cards is the check that the protocol measures what it claims to.

## Traps

Four ways to get a clean-looking result table that is entirely fake.

- **Patching the model directory does nothing on its own.** `trust_remote_code`
  executes a copy under `HF_HOME/modules/transformers_modules/`. Clear it, or the
  edit never runs and you get a perfectly clean null result.
- **`apply()` mutates module state.** Anything running a stock arm and a fixed
  arm in one process must `revert()` between them, or arm 2 inherits arm 1.
- **`do_sample=False` never terminates.** Greedy never emits the End block on
  this model's MTP paths and loops `<box><0><0><1000><1000></box>` to
  `max_new_tokens`. Every image then takes an identical "time to reach the cap".
  `serve.py`'s `--temperature 0` handles this correctly; naive `do_sample=False`
  does not.
- **Use a control arm.** Run shipped-vs-shipped as a third arm; if it is not
  exactly 0.0000, the comparison is uncalibrated.


## Findings that belong in the repo's own README

**1. The packed multi-image branch is no longer dead.** The NOTE at
`locateanything_fix.py:125-130` says nothing reaches it. True by default — but
`enable_packed_vision()` is exactly the caller the last sentence anticipates.
Measured with `archive/scripts/segcheck.py`, 8 images through `batch_utils`:

```
packed OFF   segments-per-call: {1: 216}    216 calls (27 layers x 8 images), all single-sequence
packed ON    segments-per-call: {8: 27}     27 calls (one per layer), 8 packed segments each
```

The 216 figure reproduces the existing note's 135 (27 x 5) at a different batch
size. So the two fixes are coupled: `enable_packed_vision()` makes the segmented
branch live, and the segmented branch is what makes packing safe.

**2. `enable_packed_vision()` throughput, first measurements.** 120 video frames
at 576x1024 (3,108 patches), batch engine, sdpa:

| config | fps | s/frame | peak MB |
|---|---:|---:|---:|
| b1-stock | 1.55 | 0.647 | 9,488 |
| b1 sdpa+logits | 2.80 | 0.357 | 8,099 |
| b4 sdpa+logits | 3.82 | 0.262 | 9,378 |
| b8 sdpa+logits | 3.92 | 0.255 | 10,689 |
| b8 +packed | 4.09 | 0.244 | 10,548 |
| b16 sdpa+logits | 4.07 | 0.246 | 11,236 |
| b16 +packed | **4.37** | **0.229** | 11,239 |

Packing is worth +4.3% at batch 8 and +7.4% at batch 16 on top of the SDPA fix.
Real, but small next to the SDPA fix's own 1.81x. Batching flattens past 4, which
says token decoding — not vision encoding — is the remaining bottleneck.

**3. Greedy decoding silently destroys benchmarks.** `do_sample=False` never emits
the End block on the MTP paths (`hybrid`/`fast`) and loops
`<box><0><0><1000><1000></box>` to `max_new_tokens`. Every image then takes an
identical "time to emit the token cap", which looks like a clean, plausible
result table and is entirely fake. `slow` (pure AR) is unaffected. This is the
same class of hazard as the `revert()` note about A/B contamination and deserves
a line next to it.

**4. `is_applied()` is marker-identity based.** Any wrapper around
`VL_VISION_ATTENTION_FUNCTIONS["sdpa"]` — profiler, logger, counter — drops the
`_locateanything_sdpa_4d` attribute, so `is_applied()` goes False and
`enable_packed_vision()` refuses with "call apply() first". The fix still *runs*
(the wrapper delegates to it); only the gating breaks. Copy the marker onto any
wrapper — `archive/scripts/segcheck.py` shows the pattern. The loud failure is good design; it
just needs documenting.

**5. `requirements-model.txt` understates decord.** The note says decord/lmdb/
opencv exist only so the import scan passes and "the code path never uses them".
That holds for images, not video: the model defaults to the `torchvision` video
backend, and torchvision >= 0.19 removed `io.read_video`, so video decode fails
outright with `module 'torchvision.io' has no attribute 'read_video'`. Video needs
`decord` plus `process_vision_info(..., video_reader_backend="decord")`.

**6. Unrelated to the patch, worth knowing:** feeding a clip through the
`{"type": "video"}` message path past ~20 frames exceeds the model's 16,384-token
context and `generate()` returns an **empty string with no error** — status ok,
zero boxes, indistinguishable from "found nothing". NVIDIA's own Space never uses
that path; it decodes to frames and runs each as an image.
