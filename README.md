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

### reproduced six months later

Re-ran `crossbox.py` on 2026-08-20 on a fresh H100 PCIe, torch 2.11.0+cu128, no
flash-attn — a different machine from the August run:

| patches | stock (Aug) | stock (Aug 20) | fixed (Aug) | fixed (Aug 20) |
|---|---|---|---|---|
| 5,476 | 2.86s / **4491 MB** | 3.24s / **4491 MB** | 0.52s / **1389 MB** | 0.72s / **1389 MB** |
| 10,000 | 3.73s / **14483 MB** | 4.53s / **14483 MB** | 0.61s / **2448 MB** | 0.75s / **2448 MB** |
| 14,400 | 5.79s / **29707 MB** | 6.63s / **29707 MB** | 1.12s / **3478 MB** | 0.87s / **3478 MB** |
| 25,600 | **OOM** 39.06 GiB | **OOM** 39.06 GiB | 4.02s / **6127 MB** | 3.92s / **6127 MB** |

**Every memory figure is byte-identical**, on different hardware six months
apart, including the OOM boundary. The kernel probe matches too — 898x at 14,400
patches both times. Wall-clock differs (this box is PCIe, the August one was
likely SXM) and today's speedups are better: 4.5x / 6.0x / 7.6x.

## video demo

![kitten detection](assets/video_demo.gif)

1920x1080, detected at 10fps and held across a 30fps render. 134 frames in 23.5s.

## quickstart/install

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
cd locateanything-perf
bash scripts/setup.sh
python scripts/verify_patch.py
```

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

## a/b testing script

```bash
python scripts/ab_sweep.py --images ./kitty --queries "cats" --out ./results
```

panels, csv, json, summary. flags and gotchas in [scripts/README.md](scripts/README.md).
