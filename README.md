LocateAnything-3b Performance Tweaks
------------------------------------------

I was using nvidia LocateAnything, but I didn't feel it was performing the best, and I couldn't exactly afford an H100 all the time. With this fix you can
get it running larger batches and way bigger resolutions with a couple tweaks on a 3090 at least, so hopefully this helps someone out too. It should also work on a 5090, but I will post those results later. 

![stock OOMs, fixed finds five cats](assets/panel_oom.jpg)

## the fixes

| | what | gain |
|---|---|---|
| **4-D sdpa** | vision encoder hands pytorch 3-D tensors, so every fused kernel is declined and the attention matrix is materialised in fp32 | everywhere |
| **logits slice** | wired up already, but only for the magi path — sdpa bypasses it and projects every prompt position through a 152,681-wide head | batch ≥ 4 |
| **vision cache** | re-encodes the image once per question | repeat queries |
| **packed vision** | several images share one ViT forward, off by default | +4-7% batched |
| **video decode** | not a speed fix — video is broken on torchvision ≥ 0.19 | video |


## the numbers

one image, 3090:

| image | patches | stock | fixed | real flash-attn |
|---|---|---|---|---|
| 1024px | 5,476 | 1.63s / 4499 MB | **0.62s / 1397 MB** | 0.62s / 1397 MB |
| 1400px | 10,000 | 4.62s / 14491 MB | **1.08s / 2456 MB** | 1.08s / 2456 MB |
| 1680px | 14,400 | **OOM** | **1.74s / 3486 MB** | 1.73s / 3486 MB |
| 2240px | 25,600 | **OOM** | **4.02s / 6134 MB** | 3.97s / 6134 MB |

byte-identical to real flash-attn, within 1% on time.

16 mixed-resolution photos, query "cats":

| | stock | fixed |
|---|---:|---:|
| completed | 8/16 | **16/16** |
| OOM | 8 | **0** |
| on the 8 both finished | 12.80s | **6.96s** |
| worst peak | 16,004 MB | **8,738 MB** |

![same two kittens, 2.96x faster](assets/panel_speed.jpg)

h100, batched on 2K photos:

| batch | stock | + 4-D | + logits |
|---|---|---|---|
| 8 | **0.16 img/s** | 0.52 | **1.13** |
| 16 | **OOM** | **OOM** | 0.62 |
| 32 | **OOM** | **OOM** | 0.69 |

25,600 patches is `in_token_limit` from the model's own config. it OOMs on both
cards as shipped.

AP delta −0.0011 / −0.0021 against a shipped-vs-shipped control of 0.0000.

### these are all Detection prompts

Every number above is `Locate all the instances that matches ...`, which emits a
handful of boxes. The demo has four other task types and the model card
documents three more, and the speedup on them depends almost entirely on how
much the model has to say: 4.7x on GUI grounding, 3.6x on pointing, **1.4x on
OCR**, because a dense page spends its time in ~400 decode steps that none of
these fixes touch — and in the decode mode the demo actually uses for OCR, which
measurably reads better, it is ~1.02x. What the fixes buy on OCR is headroom,
not clock: stock OOMs on every cell above 10,000 patches, fixed completes all of
them.

Each of those is measured per task and split into vision / lm-prefill / decode.

### reproduced

Re-ran [`scripts/crossbox.py`](scripts/crossbox.py) on 2026-08-20 on a fresh
H100 PCIe, torch 2.11.0+cu128, no
flash-attn — a different machine from the August run:

| patches | stock (Aug) | stock (Aug 20) | fixed (Aug) | fixed (Aug 20) |
|---|---|---|---|---|
| 5,476 | 2.86s / **4491 MB** | 3.24s / **4491 MB** | 0.52s / **1389 MB** | 0.72s / **1389 MB** |
| 10,000 | 3.73s / **14483 MB** | 4.53s / **14483 MB** | 0.61s / **2448 MB** | 0.75s / **2448 MB** |
| 14,400 | 5.79s / **29707 MB** | 6.63s / **29707 MB** | 1.12s / **3478 MB** | 0.87s / **3478 MB** |
| 25,600 | **OOM** 39.06 GiB | **OOM** 39.06 GiB | 4.02s / **6127 MB** | 3.92s / **6127 MB** |

Also run on an **A100 80GB (sm_80)** the same day:

| patches | stock | fixed | speedup |
|---|---|---|---|
| 5,476 | 1.42s / **4491 MB** | 0.26s / **1388 MB** | 5.5x |
| 10,000 | 2.45s / **14483 MB** | 0.41s / **2449 MB** | 6.0x |
| 14,400 | 4.86s / **29707 MB** | 0.62s / **3478 MB** | 7.8x |
| 25,600 | **OOM** 39.06 GiB | 1.36s / **6127 MB** | — |


## video demo

![kitten detection](assets/video_demo.gif)

1920x1080, detected at 10fps and held across a 30fps render. 134 frames in 23.5s.

## quickstart/install

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
cd locateanything-perf
bash scripts/setup.sh            # checks the box, installs, downloads, verifies
```

`setup.sh` checks the card, the driver's CUDA version, disk, and every pinned
dependency before it installs anything, then ends by running `verify_patch.py`
and refusing to claim success unless it prints `PATCH_VERIFIED`.

```bash
bash scripts/setup.sh --check    # verify an existing install, change nothing
bash scripts/setup.sh <sshhost>  # copy this repo to a remote box and install there
bash scripts/setup.sh --sglang   # ALSO build the SGLang serving venv — OCR only, see below
```

`--sglang` is **opt-in and off by default**. It is a second ~9 GB venv and it is
only worth it if you are reading text. Detection, grounding, pointing and GUI
grounding do not need it and are unaffected by it.

needs `transformers==4.57.1`. **do not install flash-attn** — transformers picks
`flash_attention_2` for the LM, qwen2 doesn't implement it, first forward dies.

## how to use

```python
import torch, locateanything_fix
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL = "nvidia/LocateAnything-3B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16,
                                  trust_remote_code=True).to("cuda").eval()

locateanything_fix.apply()        # must be after from_pretrained
```

opt-in, by workload:

```python
locateanything_fix.enable_vision_cache(model, maxsize=4)  # many queries, one image
locateanything_fix.enable_logits_slice(model)             # batch >= 4
locateanything_fix.enable_packed_vision()                 # needs apply() first
locateanything_fix.enable_video_decode()                  # video
```

| | 4-D | logits | cache | packed |
|---|---|---|---|---|
| one image, one query | **yes** | — | — | — |
| one image, many queries | **yes** | — | **yes** | — |
| many images, batched | **yes** | **yes** | — | **yes** |

keep the model resident — load is 4.04s and the first inference is 2x steady
(1.19s vs 0.59s).

## running it

```bash
python scripts/verify_patch.py --model /path/to/LocateAnything-3B   # confirm the fix is live
python scripts/serve.py       --model /path/to/LocateAnything-3B --bench ./images
python scripts/tile_ocr.py    --model /path/to/LocateAnything-3B --image ./page.jpg
```

`serve.py` keeps the model resident and torch.compiled across requests; `tile_ocr.py`
splits a page into a grid, runs the tiles as one batch, and dedupes the boxes.
Flags in [scripts/README.md](scripts/README.md).

## measured, on this repo, on one 3090

Clean install from this checkout, RTX 3090 24 GB, torch 2.11.0+cu128,
transformers 4.57.1, bf16, no flash-attn. Corpus: 16 photographs and 9 page
scans — book pages, a magazine spread, an invoice.

### locate — 16 photographs, one prompt, five resolutions

| patches | pixels (typical) | mean/photo | instances found |
|---|---|---:|---:|
| 2,500 | 606x808 | 0.39s | 54 |
| 5,476 | 897x1196 | 0.64s | 55 |
| 10,000 | 1212x1616 | 1.04s | 55 |
| 14,400 | 1455x1940 | 1.57s | 55 |
| 25,600 | 1940x2586 | 3.55s | 55 |

**Resolution does not buy recall on this task.** The same 55 instances across 16
photographs at every rung, 9x the time. One photo gains one instance between the
first two rungs and nothing changes after that. The reason to have the headroom
is that the shipped code cannot run the middle of that table at all — see below.

### the fix, three ways

Same photo, same prompt. `sdpa-only` is `apply()` alone; `fixed` adds
`enable_logits_slice(keep=6)`.

| patches | stock | sdpa-only | fixed |
|---|---|---|---|
| 5,632 | 1.65s / 4,808 MB | 0.79s / 1,494 MB | 0.75s / 372 MB |
| 10,320 | 4.21s / 15,526 MB | 1.80s / 2,635 MB | 1.45s / 596 MB |
| 14,484 | **OOM** | 2.11s / 3,635 MB | 1.91s / 862 MB |
| 25,840 | **OOM** | 4.79s / 6,421 MB | 3.83s / 1,394 MB |
| 40,460 | **OOM** | 7.89s / 10,001 MB | 8.15s / 1,816 MB |
| 66,272 | **OOM** | 18.38s / 15,773 MB | 18.54s / 3,298 MB |

Across all eight prompt templates at 10,000 patches the SDPA change alone is
**1.9-3.8x faster (mean 3.2x) and 5.4-6.0x less activation memory**; with
`logits_slice` the memory figure becomes 18.5-28.7x. The two remove different
costs — the attention change is quadratic in patches, `logits_slice` is linear
in tokens times the 152,681-token vocabulary — so below ~10k patches the first
dominates and above it the second does. Neither alone gets you high resolution.

Stock's failure is exact. Every OOM matches a dense tensor over
`S = (2*ceil(w/28)) * (2*ceil(h/28))`, the processor's real patch grid: the bool
mask at 1 byte, torch's bf16 copy of it at 2, and the fp32 attention score
matrix `[16,S,S]` at 64. Whichever first exceeds free VRAM is what you see —
fitted against all nine OOM rows, worst error 0.05%, no free parameters.

### reading text

| | per page | 9 pages | regions |
|---|---:|---:|---:|
| whole page | 7.98s | 71.8s | 423 |
| tiled 3x3, one at a time | 14.00s | 126.0s | 858 |
| **tiled 3x3, batched** | **5.06s** | **45.5s** | 836 |

Tiling and batching is **faster than a single whole-page pass and finds about
twice as much**. Batching the nine tiles rather than running them in sequence is
worth 2.77x on its own.

Serving the same nine pages with the model resident (`serve.py`, eager):
5.03s/page warm, 53.15 ms/forward, startup amortising to 0.37s/request.
`--compile` is 1.33x per forward but costs 177s of startup, so it pays back at
about 81 pages in one process.

### OCR text quality needs SGLang

The box counts above are right in every mode. **The transcribed text is not.**

This model decodes six tokens per forward through its parallel box decoder. For
coordinates (`<x0><y0><x1><y1>`) the next six tokens are nearly determined and
that works. For prose they are not, the speculation is not rejected, and the
output stutters. Measured on a scanned book page, same page, same prompt:

| decode path | seconds | transcription |
|---|---:|---|
| `generation_mode="slow"` (pure AR) | 19.24 | `...that have taken their procession flight` |
| `generation_mode="hybrid"` | 5.14 | `...that the taken theirionalional located flight` |
| `generation_mode="fast"` | 5.10 | byte-identical to `hybrid` |
| **SGLang** | **5.32** | `...that have taken their procession flight` |

`hybrid` is documented to fall back to AR on error; on this page it never did,
producing output byte-identical to `fast`. `slow` is correct and 3.7x slower.

SGLang has no parallel box decoder at all — decode is one token per forward — so
it cannot stutter, and it gets its speed from batching across requests instead.
It transcribed the page identically to `slow` (28 regions, 1,618 chars against
1,619) in 5.32s, and nine concurrent requests — the shape of a 3x3 tiled page —
cost 7.43s against 5.32s for one, at 1,137 tok/s and zero degenerate outputs by
18 concurrent.

**So: use SGLang if you are reading text, and skip it otherwise.** Boxes,
points, grounding and GUI grounding are identical across all decode paths and
need nothing beyond `apply()`.

SGLang needs `patches/02-sglang-locateanything-vision-weights.patch`, which is
not upstream. Without it the vision tower loads 54 attention tensors at random
init, the server comes up healthy, and every box is the whole image.
`setup.sh --sglang` applies it and verifies it took.
