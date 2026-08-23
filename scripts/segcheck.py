#!/usr/bin/env python3
"""Does enable_packed_vision() actually reach the segmented (packed multi-image)
branch of _sdpa_attention_4d?

    python segcheck.py --model /path/to/model --frames ./frames --n 8

Prints a histogram of packed segments per vision-attention call, with packing
off and on. Off: {1: layers x images}. On: {images: layers}.
"""
import argparse, collections, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/LocateAnything-3B")
ap.add_argument("--model-dir", default="", help="dir containing batch_utils/ (default: --model)")
ap.add_argument("--frames", required=True, help="directory of images")
ap.add_argument("--n", type=int, default=8, help="how many images to pack")
ap.add_argument("--query", default="cat")
a = ap.parse_args()

sys.path.insert(0, a.model_dir or a.model)
os.environ["LA_FLASH_MODEL"] = a.model
os.environ["LA_FLASH_ATTN"] = "sdpa"
os.environ["LA_FLASH_VISION_ATTN"] = "auto"
os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"

from batch_utils import generate_batch_hybrid, load
from batch_utils.hybrid_runtime import load_pil
import locateanything_fix

tok, proc, model = load()
MARK = "_locateanything_sdpa_4d"

def instrument():
    """Count vision-attention calls by packed-segment count.

    NB: the marker is copied onto the wrapper. Without it is_applied() goes
    False and enable_packed_vision() refuses with 'call apply() first'.
    """
    vit = locateanything_fix._find_vit_module()
    inner = vit.VL_VISION_ATTENTION_FUNCTIONS["sdpa"]
    stats = collections.Counter()
    def counting(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
        stats[0 if q_cu_seqlens is None else max(0, len(q_cu_seqlens)-1)] += 1
        return inner(q, k, v, q_cu_seqlens, k_cu_seqlens)
    if hasattr(inner, MARK):
        setattr(counting, MARK, getattr(inner, MARK))
    vit.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = counting
    return stats

files = sorted(p for p in Path(a.frames).iterdir()
               if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"})[:a.n]
pairs = [(load_pil(str(p)), a.query) for p in files]
run = lambda: generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9,
                                    repetition_penalty=1.1, max_new_tokens=256,
                                    scheduler="pipeline")

locateanything_fix.apply(verbose=False)
locateanything_fix.enable_logits_slice(model, keep=6, verbose=False)

s_off = instrument(); run()
off = dict(s_off)                      # snapshot: the wrapper keeps counting below
print(f"packed OFF  segments-per-call: {off}")

locateanything_fix.enable_packed_vision(verbose=True)
s_on = instrument(); before = dict(s_on); run()
on = {k: v - before.get(k, 0) for k, v in s_on.items() if v - before.get(k, 0) > 0}
print(f"packed ON   segments-per-call: {on}")

multi_off = sum(v for k, v in off.items() if k > 1)
multi_on  = sum(v for k, v in on.items()  if k > 1)
print(f"\ncalls hitting the segmented branch (>1 sequence): OFF={multi_off} ON={multi_on}")
print("=> segmented branch is", "LIVE with packing" if multi_on and not multi_off
      else ("live even without packing" if multi_off else "STILL DEAD"))
