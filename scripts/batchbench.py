#!/usr/bin/env python3
"""Batch throughput sweep: what does each fix buy in the batch engine?

    python batchbench.py --model /path/to/model --frames ./frames \
                         --configs b1-stock,b1-fix,b8-fix,b8-fix-packed

Config names are <batch>-<fixes>: b8-fix-packed = batch 8, sdpa+logits+packed.
Each config runs in its own process (the fixes mutate module state).
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/LocateAnything-3B")
ap.add_argument("--model-dir", default="")
ap.add_argument("--frames", required=True)
ap.add_argument("--limit", type=int, default=120)
ap.add_argument("--query", default="cat")
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--configs", default="b1-stock,b1-fix,b4-fix,b8-fix,b8-fix-packed,b16-fix,b16-fix-packed")
ap.add_argument("--out", default="")
ap.add_argument("--_config", help=argparse.SUPPRESS)
a = ap.parse_args()

if not a._config:
    rows = []
    for c in [x.strip() for x in a.configs.split(",") if x.strip()]:
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--_config", c])
    if a.out and Path(a.out).exists():
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        print(f"\n{'config':18s} {'fps':>6s} {'s/frame':>8s} {'peak MB':>9s}")
        for r in rows:
            print(f"{r['config']:18s} {r['fps']:>6.2f} {r['sec_per_frame']:>8.4f} {r['peak_mb']:>9,.0f}")
    sys.exit(0)

cfg = a._config
bs = int(cfg.split("-")[0][1:])
use_fix, use_packed = "fix" in cfg, "packed" in cfg

sys.path.insert(0, a.model_dir or a.model)
os.environ["LA_FLASH_MODEL"] = a.model
os.environ["LA_FLASH_ATTN"] = "sdpa"
os.environ["LA_FLASH_VISION_ATTN"] = "auto"
os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"

import torch
from batch_utils import generate_batch_hybrid, load
from batch_utils.hybrid_runtime import load_pil

tok, proc, model = load()
applied = []
if use_fix:
    import locateanything_fix
    locateanything_fix.apply(verbose=False); applied.append("sdpa")
    try:
        locateanything_fix.enable_logits_slice(model, keep=6, verbose=False); applied.append("logits")
    except Exception as e:
        print("  logits_slice unavailable:", e)
    if use_packed:
        try:
            locateanything_fix.enable_packed_vision(verbose=False); applied.append("packed")
        except Exception as e:
            print("  packed_vision FAILED:", type(e).__name__, e)

files = sorted(p for p in Path(a.frames).iterdir()
               if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"})[:a.limit]
pairs = [(load_pil(str(p)), a.query) for p in files]
print(f"[{cfg}] batch={bs} fixes={applied or 'none'} frames={len(files)}", flush=True)

gen = lambda ch: generate_batch_hybrid(ch, temperature=0.7, top_p=0.9,
                                       repetition_penalty=1.1,
                                       max_new_tokens=a.max_new_tokens, scheduler="pipeline")
try: gen(pairs[:min(bs, len(pairs))])        # warmup, untimed
except Exception as e: print("  warmup failed:", type(e).__name__, str(e)[:110])

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); t0 = time.perf_counter()
outs, err = [], None
try:
    for s in range(0, len(pairs), bs):
        outs.extend(gen(pairs[s:s+bs]))
except torch.cuda.OutOfMemoryError as e:
    err = "OOM: " + str(e).split("\n")[0][:110]
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:130]}"
torch.cuda.synchronize(); dt = time.perf_counter() - t0

BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
rec = {"config": cfg, "batch": bs, "fixes": applied, "frames": len(files),
       "done": len(outs), "seconds": round(dt, 2),
       "fps": round(len(outs)/dt, 2) if dt and outs else 0,
       "sec_per_frame": round(dt/len(outs), 4) if outs else None,
       "peak_mb": round(torch.cuda.max_memory_allocated()/2**20, 1),
       "boxes": sum(len(BOX.findall(str(o))) for o in outs), "error": err}
print(f"[{cfg}] {rec['done']}/{len(files)} in {rec['seconds']}s -> {rec['fps']} fps  "
      f"{rec['sec_per_frame']}s/frame  peak {rec['peak_mb']:,.0f}MB  boxes={rec['boxes']}"
      + (f"  ERR {err}" if err else ""), flush=True)
if a.out:
    with open(a.out, "a") as fh: fh.write(json.dumps(rec)+"\n")
