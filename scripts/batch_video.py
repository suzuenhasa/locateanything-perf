#!/usr/bin/env python3
"""Batched per-frame detection over a frame directory, using the LocateAnything
batch engine with all four locateanything_fix fixes.

    python batch_video.py --frames ./frames30 --query "black cat" \
                          --batch-size 16 --out ./video30/raw_batch.jsonl

Writes the same JSONL shape ab_sweep.py produces, so mkvideo.py can render it.
Per-frame `seconds` is the batch wall clock divided by the batch size -- an
average, not an individual measurement, because a batch is one fused call.
"""
import argparse, json, os, re, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--query", default="black cat")
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--model", default=os.environ.get("LA_MODEL", "nvidia/LocateAnything-3B"))
ap.add_argument("--model-dir", default=os.environ.get("LA_MODEL", "nvidia/LocateAnything-3B"),
                help="dir containing batch_utils/ (added to sys.path)")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--frame-threshold", type=float, default=0.90)
ap.add_argument("--no-packed", action="store_true")
a = ap.parse_args()

sys.path.insert(0, a.model_dir)
os.environ["LA_FLASH_MODEL"] = a.model
os.environ["LA_FLASH_ATTN"] = "sdpa"
os.environ["LA_FLASH_VISION_ATTN"] = "auto"
os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"

import torch
from batch_utils import generate_batch_hybrid, load
from batch_utils.hybrid_runtime import load_pil

tok, proc, model = load()
import locateanything_fix
applied = []
locateanything_fix.apply(verbose=False); applied.append("sdpa")
try:
    locateanything_fix.enable_logits_slice(model, keep=6, verbose=False); applied.append("logits")
except Exception as e:
    print("logits_slice unavailable:", e)
if not a.no_packed:
    try:
        locateanything_fix.enable_packed_vision(verbose=False); applied.append("packed")
    except Exception as e:
        print("packed_vision unavailable:", e)
print(f"fixes: {applied}  batch={a.batch_size}")

files = sorted(Path(a.frames).iterdir())
files = [p for p in files if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"}]
if a.limit: files = files[:a.limit]
BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

fh = open(a.out, "w", buffering=1)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); T0 = time.perf_counter()
n_done = 0
for s in range(0, len(files), a.batch_size):
    chunk = files[s:s+a.batch_size]
    pairs = [(load_pil(str(p)), a.query) for p in chunk]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    texts = generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9,
                                  repetition_penalty=1.1,
                                  max_new_tokens=a.max_new_tokens, scheduler="pipeline")
    torch.cuda.synchronize(); dt = (time.perf_counter()-t0)/max(1, len(chunk))
    for p, txt, (pil, _) in zip(chunk, texts, pairs):
        W, H = pil.size
        boxes = [[int(g[0])/1000*W, int(g[1])/1000*H, int(g[2])/1000*W, int(g[3])/1000*H]
                 for g in BOX.findall(str(txt))]
        usable = [b for b in boxes
                  if (b[2]-b[0])*(b[3]-b[1])/(W*H) <= a.frame_threshold]
        fh.write(json.dumps({"image": p.name, "query": a.query, "arm": "batch",
                             "w": W, "h": H, "px": W*H, "status": "ok",
                             "seconds": round(dt, 4), "n_boxes": len(boxes),
                             "n_usable": len(usable), "boxes": boxes, "usable": usable,
                             "answer": str(txt)})+"\n")
    n_done += len(chunk)
    print(f"  {n_done}/{len(files)}", end="\r", flush=True)
torch.cuda.synchronize(); TOT = time.perf_counter()-T0
fh.close()
print(f"\n{n_done} frames in {TOT:.2f}s = {n_done/TOT:.2f} fps "
      f"({TOT/n_done:.3f}s/frame)  peak {torch.cuda.max_memory_allocated()/2**20:,.0f} MB")
print("->", a.out)
