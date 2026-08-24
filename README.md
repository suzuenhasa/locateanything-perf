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

| image | patches | stock | fixed |
|---|---|---|---|
| 1024px | 5,476 | 1.63s / 4499 MB | **0.62s / 1397 MB** |
| 1400px | 10,000 | 4.62s / 14491 MB | **1.08s / 2456 MB** |
| 1680px | 14,400 | **OOM** | **1.74s / 3486 MB** |
| 2240px | 25,600 | **OOM** | **4.02s / 6134 MB** |

Byte-identical to real flash-attn, within 1% on time. 25,600 patches is
`in_token_limit` from the model's own config — it OOMs as shipped.

16 mixed-resolution photos, query "cats":

| | stock | fixed |
|---|---:|---:|
| completed | 8/16 | **16/16** |
| OOM | 8 | **0** |
| on the 8 both finished | 12.80s | **6.96s** |
| worst peak | 16,004 MB | **8,738 MB** |

![same two kittens, 2.96x faster](assets/panel_speed.jpg)

Both tables are Detection prompts. Other tasks gain less — 4.7x on GUI
grounding, 3.6x on pointing, 1.4x on OCR.

### reading a page

Nine page scans, whole-page OCR, one 3090, correct text in every row:

| | per page | 9 pages |
|---|---:|---:|
| in-process, whole page | 17.9s | 160.7s |
| in-process, tiled 3x3 batched | 10.1s | 91.3s |
| SGLang, one page at a time | 6.1s | 54.7s |
| **SGLang, 9 concurrent** | **1.35s** | **12.1s** |

SGLang is faster two ways: its decode is about 3x quicker per request, and it
overlaps requests, where the in-process engine reads pages one at a time. It
batches tiles within a page, not across pages. Past six concurrent the card is
the limit.

You only need it for text. Detection, grounding, pointing and GUI grounding are
the same without it.

The rest of the numbers are in [MEASUREMENTS.md](MEASUREMENTS.md).

## video demo

![kitten detection](assets/video_demo.gif)

1920x1080, detected at 10fps and held across a 30fps render. 134 frames in 23.5s.

## quickstart/install

```
MINIMUMS:  torch >= 2.0     transformers >= 4.51     python >= 3.9
MAXIMUMS:  none
```

Both minimums are upstream's, not this repo's. If your stack clears them,
nothing here asks you to change it — `setup.sh` reuses an existing torch,
leaves transformers alone above the floor, and never installs SGLang.

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
cd locateanything-perf
bash scripts/setup.sh            # checks the machine, installs, downloads, verifies
```

It checks the card, driver CUDA, disk and network before installing anything,
then ends by running `verify_patch.py` and refusing to claim success unless it
prints `PATCH_VERIFIED`.

```bash
bash scripts/setup.sh --check       # verify an existing install, change nothing
bash scripts/setup.sh --fix-decode  # also apply patches/04 — correct in-process OCR text
bash scripts/setup.sh --sglang      # patch an SGLang you already have
bash scripts/setup.sh <sshhost>     # install on a remote machine over ssh
```

Tested on torch 2.6 through 2.13 and transformers 4.51 through 5.15.1, with
byte-identical output. `patches/05` is what removes the ceiling and is applied
unconditionally — without it, transformers 5.x loads the model with
uninitialised rotary buffers and emits token soup with no error at all.

**Do not install flash-attn** — transformers picks `flash_attention_2` for the
LM, qwen2 doesn't implement it, first forward dies.

Setup flags and environment variables: [scripts/README.md](scripts/README.md).

## typical use

`setup.sh` writes `env.sh`, which sets `PYTHONPATH`, `LA_MODEL` and `HF_HOME`:

```bash
source <BASE>/env.sh

python scripts/verify_patch.py --model "$LA_MODEL"    # confirm the fix is live

# a directory of pages, model stays resident
python scripts/serve.py --bench ./pages --task OCR

# one dense page, tiled 3x3 and run as one batch
python scripts/tile_ocr.py --image ./page.jpg --grid 3x3

# locate things in photographs
python scripts/tile_ocr.py --image ./photos --modes whole \
       --task Detection --category "cats"
```

`serve.py` keeps the model resident across requests; `tile_ocr.py` splits a page
into a grid, runs the tiles as one batch and dedupes the boxes. Flags for both
in [scripts/README.md](scripts/README.md).

## how to use

Or in your own code:

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

## where things are

| | |
|---|---|
| [MEASUREMENTS.md](MEASUREMENTS.md) | every number, the hardware it came from, and how to reproduce it |
| [scripts/README.md](scripts/README.md) | setup flags, script flags, traps |
| [patches/](patches/) | the six upstream patches, each with what it fixes and how it was verified |
