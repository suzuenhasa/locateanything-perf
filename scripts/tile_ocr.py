#!/usr/bin/env python3
"""Tile a page, OCR the tiles as one batch, stitch the boxes back.

Box emission is sequential: `forward_steps = 20.2 + 2.63 x boxes` (R^2 0.91),
about 90 ms per text region in hybrid mode. Nothing parallelises across boxes --
each one is generated conditioned on the ones before it. But images DO
parallelise, because a batch-1 decode streams all 6.175 GB of weights to produce
~5 tokens while a batch-8 decode streams the same 6.175 GB to produce ~40.

So: cut the page into tiles, hand them to the batch engine at once, and the
sequential per-region cost is paid concurrently instead of one after another.

The comparison is deliberately pixel-exact. The page is resized ONCE to a patch
budget, and the tiles are crops of that same resized image, so every arm sees
the same pixels and the same text. The only difference is how the work is split.

    whole        one generate call on the page
    tiled-seq    the tiles, one call each, in sequence   (isolates tiling itself)
    tiled-batch  the tiles, one batched call             (adds the parallelism)

`tiled-seq` is the control: it should be no faster than `whole`, because the
region count is unchanged. Any real gain has to show up only in `tiled-batch`.

Boxes come back normalised 0-1000 per tile; they are remapped to page space and
deduplicated by IoU across the overlap margin.

    python scripts/tile_ocr.py --image page.png --grid 3x3 --patches 10000 \
        --modes whole,tiled-seq,tiled-batch --out ./results/tiles
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from la_common import build_prompt, parse_mixed_results, resize_short_side, short_for_patches  # noqa: E402
from la_common import parse_out_info  # noqa: E402


def tile_boxes(W, H, nx, ny, overlap):
    """Crop rectangles covering the page, expanded by `overlap` px each side."""
    out = []
    tw, th = W / nx, H / ny
    for j in range(ny):
        for i in range(nx):
            x0 = max(0, int(round(i * tw)) - overlap)
            y0 = max(0, int(round(j * th)) - overlap)
            x1 = min(W, int(round((i + 1) * tw)) + overlap)
            y1 = min(H, int(round((j + 1) * th)) + overlap)
            out.append((x0, y0, x1, y1))
    return out


def to_page(det, rect, W, H):
    """Tile-normalised 0-1000 -> page-normalised 0-1000."""
    x0, y0, x1, y1 = rect
    tw, th = x1 - x0, y1 - y0
    c = det["coords"]
    if len(c) == 4:
        px = [c[0] / 1000 * tw + x0, c[1] / 1000 * th + y0,
              c[2] / 1000 * tw + x0, c[3] / 1000 * th + y0]
    elif len(c) == 2:
        px = [c[0] / 1000 * tw + x0, c[1] / 1000 * th + y0]
    else:
        return None
    n = [px[0] / W * 1000, px[1] / H * 1000] + \
        ([px[2] / W * 1000, px[3] / H * 1000] if len(px) == 4 else [])
    return {**det, "coords": n}


def iou(a, b):
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix, iy = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe(dets, thr):
    """Greedy IoU dedupe. Overlapping tiles see the same region twice; identical
    text at the same place is one region, not two."""
    out = []
    for d in dets:
        if any(iou(d["coords"], o["coords"]) > thr and
               d.get("label", "").strip().lower() == o.get("label", "").strip().lower()
               for o in out):
            continue
        if any(iou(d["coords"], o["coords"]) > 0.85 for o in out):
            continue
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="a file, or a directory of them")
    ap.add_argument("--model", default=os.environ.get("LA_MODEL", "nvidia/LocateAnything-3B"))
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--task", default="OCR")
    ap.add_argument("--category", default="all the objects")
    ap.add_argument("--grid", default="3x3")
    ap.add_argument("--patches", type=int, default=10000,
                    help="patch budget for the WHOLE page; tiles are crops of it, "
                         "so every arm sees identical pixels")
    ap.add_argument("--overlap", type=int, default=48, help="tile overlap in px")
    ap.add_argument("--iou", type=float, default=0.55, help="dedupe threshold")
    ap.add_argument("--modes", default="whole,tiled-seq,tiled-batch")
    ap.add_argument("--batch", type=int, default=0, help="0 = all tiles in one batch")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each mode N times. The repo's own README notes the "
                         "first inference is ~2x steady, and a warmup on one tile "
                         "does not warm the whole-page shape, so run 1 and run 2 "
                         "are not the same measurement.")
    ap.add_argument("--out", default="./tile_results")
    a = ap.parse_args()

    nx, ny = (int(v) for v in a.grid.lower().split("x"))
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)

    # the batch engine clamps every input to MAX_DIM=1024 (hybrid_runtime.py:83),
    # which would shrink the whole-page arm but not the tiles and make the
    # comparison meaningless. Raise it and control resolution ourselves.
    sys.path.insert(0, a.model_dir or a.model)
    os.environ["LA_FLASH_MODEL"] = a.model
    os.environ["LA_FLASH_ATTN"] = "sdpa"
    os.environ["LA_FLASH_VISION_ATTN"] = "auto"
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"

    import torch
    from PIL import Image
    import batch_utils.hybrid_runtime as hr
    from batch_utils import generate_batch_hybrid, load

    hr.MAX_DIM = 1 << 30
    prompt = build_prompt(a.task, a.category)
    hr._PROMPT = ""
    query = prompt[:-1] if prompt.endswith(".") else prompt

    tok, proc, model = load()
    import locateanything_fix
    locateanything_fix.apply(verbose=False)
    try:
        locateanything_fix.enable_logits_slice(model, keep=6, verbose=False)
    except Exception as e:
        print("  logits_slice unavailable:", e)

    p = Path(a.image)
    files = ([p] if p.is_file() else
             sorted(q for q in p.iterdir()
                    if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}))
    print(f"prompt={prompt!r}  grid={nx}x{ny}  patches={a.patches}  files={len(files)}\n")

    def gen(pairs):
        return generate_batch_hybrid(pairs, temperature=0.7, top_p=0.9,
                                     repetition_penalty=1.1,
                                     max_new_tokens=a.max_new_tokens,
                                     scheduler="pipeline")

    fh = (outd / "raw_tiles.jsonl").open("a", buffering=1)
    rows = []
    for f in files:
        base = Image.open(f).convert("RGB")
        page = resize_short_side(base, short_for_patches(*base.size, a.patches))
        W, H = page.size
        rects = tile_boxes(W, H, nx, ny, a.overlap)
        crops = [page.crop(r) for r in rects]

        # warm the engine once per file so no arm pays the first-call cost
        try:
            gen([(crops[0], query)])
        except Exception as e:
            print("  warmup failed:", type(e).__name__, str(e)[:100])

        for mode, rep_i in [(m, i) for m in modes for i in range(a.repeats)]:
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            err, dets = None, []
            try:
                if mode == "whole":
                    outs = gen([(page, query)])
                    dets = parse_mixed_results(str(outs[0]), "")
                else:
                    pairs = [(c, query) for c in crops]
                    bs = a.batch or len(pairs)
                    if mode == "tiled-seq":
                        bs = 1
                    outs = []
                    for s in range(0, len(pairs), bs):
                        outs.extend(gen(pairs[s:s + bs]))
                    for rect, o in zip(rects, outs):
                        for d in parse_mixed_results(str(o), ""):
                            m = to_page(d, rect, W, H)
                            if m:
                                dets.append(m)
                    dets = dedupe(dets, a.iou)
            except torch.cuda.OutOfMemoryError as e:
                err = "OOM: " + str(e).split("\n")[0][:120]
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:150]}"
            torch.cuda.synchronize(); dt = time.perf_counter() - t0

            boxes = [d for d in dets if len(d["coords"]) == 4]
            refs = [d.get("label", "") for d in dets if d.get("label")]
            rec = {"image": f.name, "task": a.task, "mode": "hybrid", "arm": mode,
                   "rep": rep_i,
                   "rung": f"p{a.patches}", "grid": a.grid, "status": "error" if err else "ok",
                   "seconds": None if err else round(dt, 3),
                   "w": W, "h": H, "tiles": 1 if mode == "whole" else len(rects),
                   "n_boxes": len(boxes), "n_refs": len(refs),
                   "ref_chars": sum(len(r) for r in refs), "refs": refs[:500],
                   "peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                   "error": err}
            fh.write(json.dumps(rec) + "\n"); rows.append(rec)
            print(f"  {f.name[:26]:26s} {mode:12s} r{rep_i} {rec['status']:5s} "
                  f"{str(rec['seconds']):>8}s  tiles={rec['tiles']:2d} "
                  f"boxes={rec['n_boxes']:4d} refchars={rec['ref_chars']:5d} "
                  f"peak={rec['peak_mb']:,.0f}MB" + (f"  {err}" if err else ""), flush=True)

    # summary
    print()
    print(f"  {'arm':12s} {'n':>2} {'sec mean':>9} {'vs whole':>9} {'boxes':>7} {'ref chars':>10}")
    import statistics
    ok = [r for r in rows if r["status"] == "ok"]
    warm = [r for r in ok if r.get("rep", 0) > 0] or ok
    base_t = statistics.mean([r["seconds"] for r in warm if r["arm"] == "whole"]) \
        if any(r["arm"] == "whole" for r in warm) else None
    for m in modes:
        rs = [r for r in warm if r["arm"] == m]
        if not rs:
            continue
        t = statistics.mean(r["seconds"] for r in rs)
        cold = [r["seconds"] for r in ok if r["arm"] == m and r.get("rep", 0) == 0]
        if cold and a.repeats > 1:
            print(f"    ({m}: cold run {cold[0]:.2f}s vs warm {t:.2f}s = "
                  f"{cold[0]/t:.2f}x)")
        sp = f"{base_t/t:.2f}x" if base_t else "-"
        print(f"  {m:12s} {len(rs):2d} {t:9.2f} {sp:>9} "
              f"{statistics.mean(r['n_boxes'] for r in rs):7.1f} "
              f"{statistics.mean(r['ref_chars'] for r in rs):10.0f}")
    (outd / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {outd}/raw_tiles.jsonl  (scoreable with archive/scripts/ocr_score.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
