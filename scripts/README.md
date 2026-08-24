# Runtime scripts for `locateanything-perf`

All confirmed working on an RTX 3090 (24 GB), `nvidia/LocateAnything-3B` in
bf16.

Every script takes `--model` (or `LA_MODEL`); nothing is hardcoded to a machine.
`locateanything_fix` must be importable -- `pip install -e .` or
`PYTHONPATH=/path/to/locateanything-perf`.

## Versions

```
MINIMUMS:  torch >= 2.0     transformers >= 4.51     python >= 3.9
MAXIMUMS:  none
```

Both minimums are upstream's, not this repo's, and there is no upper bound. If
your stack clears them, nothing here asks you to change it.

Tested end to end, same page, **byte-identical output at every point** (28
boxes, 1,621 ref chars):

| torch | CUDA | transformers |
|---|---|---|
| 2.6.0+cu124 | 12.4 | 4.51.3 |
| 2.11.0+cu130 | 13.0 | 5.12.1 |
| 2.13.0+cu132 | 13.2 | 5.15.1 |

- **torch 2.0** is where `scaled_dot_product_attention` lands -- the one torch
  API the checkpoint cannot do without, and transformers' own declared floor.
  The checkpoint asserts no torch version at all.
- **transformers 4.51** is where `models.qwen3` appears, which
  `configuration_locateanything.py` and `modeling_locateanything.py` import at
  module level for a branch this checkpoint never takes (its `architectures` is
  `Qwen2ForCausalLM`). Lazy that import and the next wall is 4.45, at
  `processing_utils.Unpack`.

`patches/05-transformers5-compat.patch` is what removes the ceiling. The
checkpoint's code assumes transformers 4.x in six places; five raise on 5.x and
the sixth does not -- the rotary embedding's buffers are `persistent=False`, so
they are absent from the checkpoint and 5.x materialises them from meta without
writing to them (`inv_freq[:3] == [1.19e-26, 0.0, 0.0]`, should start at 1.0).
`cos` goes non-finite, layer 0 is NaN on the first forward, and the model loads
with every weight matching the safetensors and emits token soup. Applied
unconditionally by `setup.sh` for that reason. Every hunk is guarded, so 4.x
behaves exactly as it did before.

## Scripts

| script | what it does |
|---|---|
| `setup.sh` | checks the box, reuses or builds an environment, downloads the checkpoint, applies the patches, and refuses to claim success unless `verify_patch.py` passes |
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

## Setup

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
cd locateanything-perf
bash scripts/setup.sh
```

It checks compute capability, VRAM, disk, DNS and outbound HTTPS, and the
driver's CUDA version before installing anything, then ends by running
`verify_patch.py` and refusing to claim success unless it prints
`PATCH_VERIFIED`. Afterwards `source <BASE>/env.sh` sets `PYTHONPATH`,
`LA_MODEL` and `HF_HOME`.

| flag | what it does |
|---|---|
| *(none)* | environment, checkpoint, `patches/05`, verify |
| `--check` | verify an existing install, change nothing. Exits non-zero on any problem, so it works as a CI gate |
| `--fix-decode` | also apply `patches/04`, which makes in-process transcription correct. See the OCR section below |
| `--sglang` | patch an SGLang **you already have** with `patches/02` (version-gated) and `patches/06` |
| `<sshhost>` | copy this checkout to a remote machine over ssh and run there |

### It installs as little as it can

- **torch already present** -> used as-is at whatever version, no wheels
  downloaded. It only installs torch if there is none.
- **transformers at or above 4.51** -> left alone at whatever version. The only
  version it will ever change is one below the floor, and it upgrades to the
  minimum, not the latest, after saying why.
- **an interpreter that already has both** -> reused rather than building a venv
  beside the checkout and downloading several GB next to a working install.
  `LA_NO_SYSTEM_PY=1` to opt out.
- **SGLang** -> never installed. Installing it into a working environment drags
  its own pinned torch over yours. `--sglang` patches one that is already
  importable; if there is none, setup says so and stops.

What it does have to fetch on a bare machine: the 7.3 GB checkpoint, and
whichever of `accelerate peft einops timm decord lmdb opencv-python-headless`
are missing.

| variable | for |
|---|---|
| `LA_PY` | use this interpreter |
| `LA_BASE` | where the venv, model and `env.sh` go |
| `LA_MODEL` | a checkpoint you already have — skips the 7.3 GB download |
| `LA_HF_HOME` | share an existing HF cache |
| `TORCH_INDEX` | override the wheel index chosen from your driver's CUDA version |
| `LA_SGLVENV` | build a separate SGLang venv here |
| `LA_NO_SYSTEM_PY` | never reuse an interpreter off `PATH` |
| `LA_UNPINNED` | take the checkpoint's current HEAD instead of the pinned revision |
| `LA_SKIP_VERIFY` | skip the final `verify_patch.py` |

## Typical use

```bash
source <BASE>/env.sh

python scripts/verify_patch.py --model "$LA_MODEL"

# a directory of pages, model stays resident
python scripts/serve.py --bench ./inbox --task OCR

# one dense page, tiled
python scripts/tile_ocr.py --image ./page.jpg --grid 3x3
```

## Serving pages warm (`serve.py`)

```bash
python serve.py --bench ./inbox --compile --warmup 6 --temperature 0
python serve.py --bench ./inbox --no-compile           --temperature 0   # control
python serve.py --bench ./inbox --compile --warmup 6 --strict            # staging
```

Nine page scans, 2x3 tiles, `--temperature 0`, with `patches/04` applied so the
text is correct:

| | s/page | ms/forward | startup |
|---|---|---|---|
| eager | 8.92 | 53.75 | 7.2 s |
| compiled | **6.70** | **40.34** | 387.7 s cold / 181.5 s warm |

`ms/forward` is the invariant unit and `patches/04` does not move it -- 53.75
against 53.15 measured before the patch. Correct text is not a slower engine, it
is more forwards.

**`--compile` is paid on every process start, not once per machine.** It traces
in memory; what survives is the on-disk inductor cache (~94 MB at
`/tmp/torchinductor_$USER`), which halves the trace but does not remove it:

| | startup | saves 2.22 s/page, so pays back at |
|---|---:|---:|
| first compile on a machine (cold cache) | 387.7 s | ~171 pages |
| every later process start (warm) | 181.5 s | ~78 pages |
| `--no-compile` | 7.2 s | n/a |

If a process handles fewer than ~78 pages before exiting, `--no-compile` wins
outright -- compile is for a long-lived `serve.py`, not a run that starts and
stops. And if `/tmp` is not persistent where you are running (containers,
scratch filesystems), point `TORCHINDUCTOR_CACHE_DIR` somewhere that is, or
every start pays the cold price.

Compile startup is a property of the machine, so measure it rather than trusting
the number above. Holding hardware and torch fixed and changing only
transformers, cold in both cases: 320.0 s / 26 graphs on 4.51.3 against
387.7 s / 42 graphs on 5.12.1.

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

Measured over nine page scans -- book pages, a magazine spread, an invoice --
at 10,000 patches:

| 3x3 | s/page | 9 pages | regions |
|---|---:|---:|---:|
| whole | 17.86 | 160.7 | 347 |
| tiled, batched | **10.14** | **91.3** | 869 |

Faster *and* about 2.5x the regions, because each tile gets the full
25,600-patch budget spent on its own text. On one dense page with all three arms:
whole 16.43 s, tiled one-at-a-time 35.23 s, tiled batched 12.53 s -- so batching
the tiles rather than running them in sequence is worth **2.81x on its own**.
That is the whole speed win, and it is why `tiled-seq` exists only as a control.

(These are with `patches/04` applied. Before it the same corpus ran at 7.98 and
5.06 s/page, but that was the model speculating six tokens of prose per forward
and never checking them -- right boxes, mush words. Correct text costs about 2x;
the ratio tiling buys is unaffected.)

The direction is corpus-dependent and the counts say which case you are in. On
pages the whole-page pass already reads competently, tiling costs about the same
wall clock and roughly doubles the regions. On a page it cannot read -- a dense
newspaper front page where the whole-page pass returns five boxes -- tiling costs
10x the time and returns 30-60x the regions. Wall clock alone will call that a
regression; it is the whole-page arm giving up.

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

## OCR text quality: the boxes are right, the text may not be

`tile_ocr.py` and `serve.py` both call the batch engine, which is hardcoded to
`generation_mode="hybrid"` -- six tokens per forward through the model's
parallel box decoder. That is right for coordinates, where the next six tokens
are nearly determined by `<x0><y0><x1><y1>`, and wrong for prose, where they are
not and the speculation is not rejected. The result is a systematic stutter.

Same scanned book page, same prompt, one line of it:

| decode path | seconds | what it read |
|---|---:|---|
| `slow` (pure AR) | 19.24 | `that have taken their procession flight` |
| `hybrid` | 5.14 | `that the taken theirionalional located flight` |
| `fast` | 5.10 | byte-identical to `hybrid` |
| SGLang | 5.32 | `that have taken their procession flight` |

Two things worth knowing before trusting an OCR number from this repo:

- **`hybrid` did not fall back.** It is documented as "MTP first, fall back to AR
  on error, switch back on box_end". On this page it produced output
  byte-identical to `fast` -- it never fell back once.
- **Region counts cannot see this.** All four paths return 28 regions and within
  25 characters of each other. A benchmark that counts boxes scores the garbled
  run and the clean run identically, which is why this went unnoticed.

SGLang has no parallel box decoder (`grep -c n_future` its `locate_anything.py`:
zero) and gets its speed from batching across requests instead, so it cannot
stutter. It matched `slow` exactly -- 28 regions, 1,618 chars against 1,619 --
in 5.32s.

Detection, grounding, pointing and GUI grounding are unaffected: their output is
coordinates, which is what the parallel decoder is for. Only transcription is
hit.

### Three ways to be correct

| | text | boxes | needs |
|---|---|---|---|
| locating only | n/a | correct | nothing beyond `apply()` |
| some OCR | correct, ~17.9 s/page | correct | `setup.sh --fix-decode` |
| OCR in volume | correct, ~1.35 s/page at 9 concurrent | correct | an SGLang install, then `setup.sh --sglang` |

### How much slower is OCR without SGLang

Nine **distinct** page scans, whole-page OCR, warm server. Distinct matters:
sending the same page N times lets SGLang's radix cache serve the shared
prefill, which inflates the result.

| | 9 pages | per page |
|---|---:|---:|
| in-process, `patches/04` applied | 160.7 s | 17.90 s |
| SGLang, one page at a time | 54.7 s | 6.08 s |
| SGLang, 3 concurrent | 22.5 s | 2.50 s |
| SGLang, 6 concurrent | 12.9 s | 1.44 s |
| **SGLang, 9 concurrent** | **12.1 s** | **1.35 s** |

Concurrency saturates around six -- past that the card is the limit, not the
scheduler. End to end that is **13x**, from two gaps that compound: **2.9x per
request**, because the model's decode loop runs at 21% of this card's
memory-bandwidth roofline (31.5 ms/token) against SGLang's 73% (9.0 ms/token);
and **4.5x across requests**, because the in-process engine serves pages one
after another. It batches tiles *within* a page -- that is the 2.81x above --
but it cannot overlap *pages*. That is the part you do not have without SGLang.

`setup.sh` reports which of these paths you are on whether or not you pass
`--sglang`, because nothing errors without it; pages are simply an order of
magnitude slower and there is no reason to suspect it.

### The two SGLang patches, one of which is version-dependent

**`patches/06` is required.** `sglang/srt/models/kimi_vl_moonvit.py` is a
verbatim vendoring of the checkpoint's `modeling_vit.py`, so it carries the same
3-D SDPA defect `patches/01` fixes -- and `locateanything_fix.apply()` cannot
reach it, because the server is a different process running a different module.
Unpatched, the server loads clean, reports healthy, and dies on the first
full-resolution page trying to allocate 10.62 GiB, taking the whole server with
it. Launch with `LA_VIT_FASTMASK=1` or the patch is inert.

**`patches/02` depends on your SGLang version, and both directions fail
silently.** It renames the checkpoint's vision tensors (`wqkv`/`wo`) to the names
older SGLang used (`attn.qkv_proj`/`attn.proj`). Newer SGLang renamed its *own*
modules to match the checkpoint instead, so there the patch renames them **away**
from the correct names -- and it still applies cleanly, which is the trap.
Either way all 54 vision-attention tensors miss `params_dict` and stay at random
init, the tower still loads its MLPs and norms, **the server comes up healthy**,
and every box is the whole image `<0><0><1000><1000>`. Measured on sglang
0.5.16: patched, 108 parameters did not receive weights; reverted, 0.

`setup.sh --sglang` detects which way your build goes rather than assuming, and
`--check` fails if `patches/02` is applied to one that does not want it.


## Traps

Six ways to get a clean-looking result that is entirely fake.

- **A model that loads is not a model that works.** On transformers 5.x without
  `patches/05`, the rotary buffers come back as uninitialised memory. Nothing
  raises: the model loads, every weight matches the safetensors byte for byte,
  and the output is token soup. `verify_patch.py` covers the vision attention
  numerically; it is the only check that runs the real code path.
- **An SGLang server that reports healthy is not a server that sees the image.**
  Get `patches/02` wrong in either direction and 54 vision tensors stay at random
  init while the tower still loads its MLPs and norms. Every box comes back as
  the whole image. Grep the launch log for `not found in params_dict` and
  `did not receive weights`; both should be zero.

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
