# Benchmark & harness scripts for `locateanything-perf`

All confirmed working on an RTX 3090 (24 GB), torch 2.11.0+cu128,
transformers 4.57.1, `nvidia/LocateAnything-3B` in bf16.

Every script takes `--model` (or `BASE_DIR`); nothing is hardcoded to a machine.
`locateanything_fix` must be importable — `pip install -e .` or
`PYTHONPATH=/path/to/locateanything-perf`.

## Scripts

| script | what it does |
|---|---|
| `setup.sh` | venv + torch + pinned deps + `pip install -e` the patch + model download. `./setup.sh [BASE_DIR] [PATCH_REPO_DIR]` |
| `verify_patch.py` | patch is live on the real code path, and numerically equivalent to the original. Exit 0/1, so it works as a CI gate |
| `ab_sweep.py` | the main A/B harness: stock vs fixed over a directory of images x N queries -> panels, CSV, summary |
| `batchbench.py` | throughput sweep across batch sizes and fix combinations in the batch engine |
| `batch_video.py` | batched detection over a frame directory; writes `ab_sweep`-shaped JSONL |
| `mkvideo.py` | draws per-frame boxes and re-encodes to mp4; supports detect-low / render-high via `--hold` |
| `segcheck.py` | proves whether the packed multi-image branch of `_sdpa_attention_4d` actually executes |

## Typical use

```bash
git clone https://github.com/suzuenhasa/locateanything-perf.git
bash locateanything-perf/scripts/setup.sh          # BASE defaults to the repo's parent

source venv/bin/activate
export PYTHONPATH="$PWD/locateanything-perf"
export LA_MODEL="$PWD/model"                       # or just use the hub id

python locateanything-perf/scripts/verify_patch.py --model "$LA_MODEL"

# images: stock vs fixed
python ab_sweep.py --images ./inbox --queries "cat,kitten" --out ./results

# video: extract -> batched detect -> render (detect at 5fps, hold x6, render 30fps)
ffmpeg -i clip.mp4 -vf "fps=5,scale=384:-2" -q:v 3 frames/f%04d.jpg
python batch_video.py --frames ./frames --query "black cat" --batch-size 16 \
                      --out ./raw.jsonl
python mkvideo.py --frames ./frames --render-frames ./frames_full \
                  --results ./raw.jsonl --query "black cat" --fps 30 --hold 6 --out out.mp4
```

## ab_sweep.py flags

| flag | default | what it does |
|---|---|---|
| `--images DIR` | *required* | folder of images (jpg/png/webp/jfif/bmp/gif/tif) |
| `--queries "a,b,c"` | `cat` | comma-separated things to locate |
| `--out DIR` | `./ab_results` | output directory |
| `--model PATH` | `$LA_MODEL`, else the hub | local path or HF id |
| `--arms` | `stock,fixed` | which arms to run |
| `--fixes` | `sdpa,logits,cache` | which fixes the fixed arm uses — **A/B one at a time** |
| `--limit N` | all | only the N smallest images, for a quick check |
| `--frame-threshold` | `0.90` | a box covering more than this is a whole-frame non-answer |
| `--no-render` | off | data only, skip the panels |
| `--seed` / `--max-new-tokens` / `--temperature` / `--top-p` | 1234 / 2048 / 0.7 / 0.9 | generation |

Outputs `panels/*.jpg`, `full_table.csv`, `results.json` (with box geometry),
`summary.md` and `raw_*.jsonl`. It streams to JSONL and resumes, so an OOM that
kills the process is retried rather than losing the sweep.

## Three things that look like quirks and are not

- **Each arm runs in its own subprocess.** `apply()` mutates module state, so a
  fresh process is the only honest way to measure the stock arm, and it gives a
  clean peak-memory number per arm.
- **The loop is image-major** (all queries for one image, then the next) so the
  vision cache can hit. Query-major evicts before reuse and scores zero hits,
  which makes the cache look worthless.
- **`do_sample` stays True.** Greedy never emits the End block on this model's MTP
  paths and loops `<box><0><0><1000><1000></box>` to the token cap. Every image
  then takes an identical "time to reach the cap", which looks like a clean result
  table and is entirely fake.

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

Note also that `mkvideo.py` labels boxes by their index within that frame, so
"kitten 1" is not a stable identity across frames — there is no tracker here.

## Video, end to end

Detect on a downscaled low-fps pass, render on the full-rate one:

```bash
mkdir -p frames frames_full
ffmpeg -i clip.mp4 -vf "fps=10,scale=-2:384" -q:v 3 frames/f%04d.jpg   # detected
ffmpeg -i clip.mp4 -vf "fps=30" -q:v 2 frames_full/f%04d.jpg           # drawn on

python batch_video.py --frames ./frames --query "all the kittens" \
                      --batch-size 16 --out ./raw.jsonl
python mkvideo.py --frames ./frames --render-frames ./frames_full \
                  --results ./raw.jsonl --query "all the kittens" \
                  --fps 30 --hold 3 --out out.mp4
```

`--hold` is output fps / detect fps, so 30/10 = 3. Detecting every frame at 30fps
is mostly wasted; things do not move that fast.

Measured on a 1920x1080 540-frame clip, 3090, all fixes on:

| | fps=10 / hold 3 | fps=5 / hold 6 |
|---|---|---|
| detect frames | 134 | 67 |
| detect wall | 23.5s | **11.9s** |
| peak | 9,511 MB | **8,948 MB** |
| boxes/frame | 1.96 | 1.94 |

Halving the detection rate cost almost nothing on this footage, but `--hold 6`
holds a box for 200ms, so fast motion will show visible lag.


## Findings that belong in the repo's own README

**1. The packed multi-image branch is no longer dead.** The NOTE at
`locateanything_fix.py:125-130` says nothing reaches it. True by default — but
`enable_packed_vision()` is exactly the caller the last sentence anticipates.
Measured with `segcheck.py`, 8 images through `batch_utils`:

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
wrapper — `segcheck.py` shows the pattern. The loud failure is good design; it
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
