#!/usr/bin/env python3
"""A/B sweep for nvidia/LocateAnything-3B: stock vs locateanything_fix.

Point it at a directory of images, give it what you want to find, get back
side-by-side panels, a full CSV, and a summary.

    python ab_sweep.py --images ./inbox --queries "cat,kitten" --out ./results

Runs each arm in its own subprocess: applying the fix mutates module state, so a
fresh process is the only honest way to measure the stock arm and to get clean
peak-memory numbers. Results stream to JSONL, so a hard OOM kill can be resumed
by re-running the same command.
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path

BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".gif", ".tif", ".tiff"}
COLORS = ["#ff3b30","#34c759","#0a84ff","#ffd60a","#bf5af2","#ff9f0a","#5ac8fa",
          "#30d158","#ff6482","#64d2ff","#ffd426","#c77dff"]


def build_args():
    p = argparse.ArgumentParser(
        description="Stock vs fixed A/B sweep for LocateAnything-3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python ab_sweep.py --images ./inbox --queries "cat"
  python ab_sweep.py --images ./photos --queries "person,car,bicycle" --out ./out
  python ab_sweep.py --images ./inbox --queries "cat,kitten" --fixes sdpa
  python ab_sweep.py --images ./inbox --queries "cat" --arms fixed --limit 5
""")
    p.add_argument("--images", required=True, help="directory of images to sweep")
    p.add_argument("--queries", default="cat",
                   help="comma-separated things to find, e.g. \"cat,kitten\" (default: cat)")
    p.add_argument("--out", default="./ab_results", help="output directory")
    p.add_argument("--model", default=os.environ.get("LA_MODEL", "nvidia/LocateAnything-3B"),
                   help="model path or HF id (env: LA_MODEL)")
    p.add_argument("--arms", default="stock,fixed", help="stock,fixed (default: both)")
    p.add_argument("--fixes", default="sdpa,logits,cache",
                   help="which fixes the 'fixed' arm applies (default: all three)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=0, help="only the N smallest images (0 = all)")
    p.add_argument("--frame-threshold", type=float, default=0.90,
                   help="a box covering more than this fraction of the frame is "
                        "counted as a whole-frame non-answer (default: 0.90)")
    p.add_argument("--panel-width", type=int, default=660)
    p.add_argument("--no-render", action="store_true", help="collect data only")
    p.add_argument("--_arm", help=argparse.SUPPRESS)   # internal: run one arm
    return p


def list_images(d, limit=0):
    from PIL import Image
    fs = [p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMG_EXT]
    def px(p):
        try:
            with Image.open(p) as im: return im.size[0] * im.size[1]
        except Exception: return 0
    fs = [p for p in fs if px(p) > 0]
    fs.sort(key=px)
    return fs[:limit] if limit else fs


# --------------------------------------------------------------------------
# worker: one arm, in its own process
# --------------------------------------------------------------------------
def run_arm(a):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    arm, queries = a._arm, [q.strip() for q in a.queries.split(",") if q.strip()]
    out = Path(a.out) / f"raw_{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            if l.strip():
                r = json.loads(l); done.add((r["image"], r["query"]))

    files = list_images(a.images, a.limit)
    todo = [(p, q) for p in files for q in queries if (p.name, q) not in done]
    print(f"[{arm}] {len(files)} images x {len(queries)} queries = "
          f"{len(files)*len(queries)} cells, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("ALL_DONE", flush=True); return 0

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda").eval()
    cache = None
    if arm == "fixed":
        import locateanything_fix
        want = {f.strip() for f in a.fixes.split(",") if f.strip()}
        if "sdpa" in want:
            locateanything_fix.apply(verbose=False)
            locateanything_fix.verify(model)
        if "logits" in want:
            locateanything_fix.enable_logits_slice(model, keep=6, verbose=False)
        if "cache" in want:
            cache = locateanything_fix.enable_vision_cache(model, maxsize=4)
        print(f"[fixed] applied: {sorted(want)}", flush=True)
    else:
        try:
            import locateanything_fix
            assert not locateanything_fix.is_applied(), "stock arm is contaminated"
        except ImportError:
            pass

    @torch.no_grad()
    def infer(img, query):
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        W, H = img.size
        prompt = ("Locate all the instances that matches the following "
                  f"description: {query}.")
        msg = [{"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": prompt}]}]
        text = proc.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        ims, vids = proc.process_vision_info(msg)
        inp = proc(text=[text], images=ims, videos=vids, return_tensors="pt").to("cuda")
        r = model.generate(
            pixel_values=inp["pixel_values"].to(torch.bfloat16),
            input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
            image_grid_hws=inp.get("image_grid_hws"), tokenizer=tok,
            max_new_tokens=a.max_new_tokens, use_cache=True, generation_mode="hybrid",
            # NB: do_sample must stay True. Greedy never emits the End block on the
            # MTP paths and loops <box><0><0><1000><1000></box> to the token cap.
            do_sample=True, temperature=a.temperature, top_p=a.top_p,
            repetition_penalty=1.1, verbose=False)
        ans = r[0] if isinstance(r, tuple) else r
        if isinstance(ans, (list, tuple)): ans = ans[0]
        ans = str(ans)
        boxes = [[int(g[0])/1000*W, int(g[1])/1000*H, int(g[2])/1000*W, int(g[3])/1000*H]
                 for g in BOX_RE.findall(ans)]
        A = W * H
        usable = [b for b in boxes
                  if (b[2]-b[0])*(b[3]-b[1]) / A <= a.frame_threshold]
        g = inp.get("image_grid_hws")
        return boxes, usable, (int(g.prod(-1).sum()) if g is not None else None), ans

    # warm up kernel selection; not timed, and the cache is cleared after so it
    # cannot fake a hit on a measured cell
    try:
        infer(Image.open(files[0]).convert("RGB"), queries[0])
    except Exception as e:
        print(f"[{arm}] warmup failed (continuing): {type(e).__name__}: {e}", flush=True)
    if cache is not None:
        cache.clear(); cache.hits = 0; cache.misses = 0
    torch.cuda.empty_cache()

    fh = out.open("a", buffering=1)
    t_arm = time.perf_counter()
    # image-major: every query for one image before moving on, so the vision
    # cache can actually hit. Query-major would evict before reuse.
    last = None
    for p, q in todo:
        if p != last:
            img = Image.open(p).convert("RGB"); last = p
        W, H = img.size
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        rec = {"image": p.name, "query": q, "arm": arm, "w": W, "h": H, "px": W*H}
        try:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            boxes, usable, patches, ans = infer(img, q)
            torch.cuda.synchronize()
            rec.update(status="ok", seconds=round(time.perf_counter()-t0, 4),
                       n_boxes=len(boxes), n_usable=len(usable), boxes=boxes,
                       usable=usable, patches=patches, answer=ans,
                       terminated=ans.rstrip().endswith("<|im_end|>"))
        except torch.cuda.OutOfMemoryError as e:
            rec.update(status="OOM", seconds=None, error=str(e).split("\n")[0][:160])
            torch.cuda.empty_cache()
        except Exception as e:
            m = str(e)
            rec.update(status="OOM" if "out of memory" in m.lower() else "error",
                       seconds=None, error=m.split("\n")[0][:160])
            torch.cuda.empty_cache()
        rec["peak_mb"] = round(torch.cuda.max_memory_allocated()/2**20, 1)
        if cache is not None:
            rec["cache"] = dict(cache.stats)
        fh.write(json.dumps(rec)+"\n")
        print(f"[{arm}] {p.name} :: {q!r} -> {rec['status']} "
              f"{rec.get('seconds','-')}s {rec.get('n_usable','-')} found", flush=True)
    print(f"[{arm}] ARM WALL CLOCK {time.perf_counter()-t_arm:.2f}s", flush=True)
    if cache is not None:
        print(f"[fixed] cache {cache.stats}", flush=True)
    print("ALL_DONE", flush=True)
    return 0


# --------------------------------------------------------------------------
# render + tabulate
# --------------------------------------------------------------------------
def fnt(sz, bold=True):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", "C:/Windows/Fonts/"):
        try: return ImageFont.truetype(d + n, sz)
        except Exception: pass
    return ImageFont.load_default()


def side_panel(img, rec, label, other, PW):
    from PIL import Image, ImageDraw
    im = img.copy(); sc = PW / im.width
    im = im.resize((PW, max(1, int(im.height*sc))), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    if rec and rec["status"] == "ok":
        for i, b in enumerate(rec["usable"]):
            c = COLORS[i % len(COLORS)]
            d.rectangle([v*sc for v in b], outline=c, width=3)
            t = str(i+1); f = fnt(17); bb = d.textbbox((0,0), t, font=f)
            d.rectangle([b[0]*sc, b[1]*sc, b[0]*sc+(bb[2]-bb[0])+11,
                         b[1]*sc+(bb[3]-bb[1])+10], fill=c)
            d.text((b[0]*sc+5, b[1]*sc+2), t, fill="#000", font=f)
        drop = rec["n_boxes"] - rec["n_usable"]
        l2 = f"{rec['n_usable']} found" + (f"  ({drop} whole-frame dropped)" if drop else "")
        l3 = f"{rec['seconds']:.2f}s   peak {rec['peak_mb']:,.0f} MB"
        if other and other.get("status") == "ok" and rec["seconds"] > 0 \
           and other["seconds"] > rec["seconds"]:
            l3 += f"   {other['seconds']/rec['seconds']:.2f}x faster"
    elif rec and rec["status"] == "OOM":
        d.rectangle([0,0,im.width,im.height], fill=(0,0,0,170))
        d.text((im.width//2, im.height//2), "OOM", fill="#ff453a",
               font=fnt(max(28, PW//8)), anchor="mm")
        e = rec.get("error", "")
        want = e.split("Tried to allocate")[-1].split("GiB")[0].strip() \
               if "Tried to allocate" in e else "?"
        l2, l3 = "OUT OF MEMORY - no result", f"tried to allocate {want} GiB"
    else:
        d.rectangle([0,0,im.width,im.height], fill=(0,0,0,170))
        l2, l3 = ((rec or {}).get("status") or "not run"), ""
    hh = 74
    c = Image.new("RGB", (im.width, im.height+hh), "#0f1216")
    c.paste(im, (0, hh)); hd = ImageDraw.Draw(c)
    hd.text((13, 7),  label, fill="#ffffff", font=fnt(19))
    hd.text((13, 31), l2,    fill="#e8edf3", font=fnt(16))
    hd.text((13, 52), l3,    fill="#93a0ae", font=fnt(14, False))
    return c


def render(a, arms):
    from PIL import Image, ImageDraw
    outd = Path(a.out); pand = outd / "panels"; pand.mkdir(parents=True, exist_ok=True)
    queries = [q.strip() for q in a.queries.split(",") if q.strip()]
    data = {}
    for arm in arms:
        f = outd / f"raw_{arm}.jsonl"
        if f.exists():
            for l in f.open():
                if l.strip():
                    r = json.loads(l); data[(r["image"], r["query"], arm)] = r
    names = sorted({k[0] for k in data},
                   key=lambda n: next(r["px"] for k, r in data.items() if k[0] == n))
    prim = "fixed" if "fixed" in arms else arms[0]
    rows, per_image = [], []
    for n in names:
        # same query on both sides: the one the primary arm resolves best
        cand = [(q, data.get((n, q, prim))) for q in queries]
        best = max(cand, key=lambda t: (t[1]["n_usable"]
                   if t[1] and t[1]["status"] == "ok" else -1))[0]
        s, f = data.get((n, best, "stock")), data.get((n, best, "fixed"))
        ref = f or s
        src = Path(a.images) / n
        if not a.no_render and ref:
            img = Image.open(src).convert("RGB"); W, H = img.size
            panels, labels = [], []
            if "stock" in arms: panels.append((s, "STOCK - no fix"))
            if "fixed" in arms: panels.append((f, f"FIXED - {a.fixes}"))
            built = [side_panel(img, r, lab, (f if lab.startswith("STOCK") else s), a.panel_width)
                     for r, lab in panels]
            bh = 62
            sheet = Image.new("RGB",
                              (sum(b.width for b in built) + 10*(len(built)-1),
                               max(b.height for b in built) + bh), "#0f1216")
            bd = ImageDraw.Draw(sheet)
            bd.text((13, 8), n, fill="#ffffff", font=fnt(21))
            bd.text((13, 36),
                    f"{W}x{H}   {W*H/1e6:.2f} MP   {ref.get('patches') or 0:,} vision patches"
                    f"   |   query: \"{best}\"   |   seed {a.seed}, hybrid, bf16",
                    fill="#93a0ae", font=fnt(15, False))
            x = 0
            for b in built:
                sheet.paste(b, (x, bh)); x += b.width + 10
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                           for ch in n.rsplit(".", 1)[0])[:40]
            sheet.save(pand / f"{safe}.jpg", "JPEG", quality=87, optimize=True)
        per_image.append({
            "image": n, "w": ref["w"], "h": ref["h"], "mp": round(ref["px"]/1e6, 2),
            "patches": ref.get("patches"), "best_query": best,
            "stock_status": (s or {}).get("status"), "fixed_status": (f or {}).get("status"),
            "stock_s": (s or {}).get("seconds"), "fixed_s": (f or {}).get("seconds"),
            "stock_mb": (s or {}).get("peak_mb"), "fixed_mb": (f or {}).get("peak_mb"),
            "stock_found": (s or {}).get("n_usable"), "fixed_found": (f or {}).get("n_usable")})
        for arm in arms:
            for q in queries:
                r = data.get((n, q, arm))
                if not r: continue
                rows.append({"image": n, "width": r["w"], "height": r["h"],
                             "megapixels": round(r["px"]/1e6, 2), "patches": r.get("patches"),
                             "query": q, "arm": arm, "status": r["status"],
                             "seconds": r.get("seconds"), "peak_mb": r.get("peak_mb"),
                             "boxes_raw": r.get("n_boxes"), "boxes_usable": r.get("n_usable"),
                             "cache_hits": (r.get("cache") or {}).get("hits"),
                             "error": r.get("error", "")})
    if rows:
        with (outd/"full_table.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    json.dump({"per_image": per_image, "cells": rows},
              (outd/"results.json").open("w"), indent=1)
    write_summary(a, arms, rows, per_image, outd)
    n_panels = 0 if a.no_render else len(names)
    print(f"\nwrote {n_panels} panels, {len(rows)} rows -> {outd}")


def write_summary(a, arms, rows, per_image, outd):
    S = [r for r in rows if r["arm"] == "stock"]; F = [r for r in rows if r["arm"] == "fixed"]
    Sok = [r for r in S if r["status"] == "ok"]; Fok = [r for r in F if r["status"] == "ok"]
    L = [f"# LocateAnything-3B A/B sweep\n",
         f"`{len(per_image)}` images x `{len(a.queries.split(','))}` queries "
         f"= **{len(per_image)*len(a.queries.split(','))} cells per arm**\n",
         "## Run\n", "| | |", "|---|---|",
         f"| images | `{a.images}` |", f"| queries | `{a.queries}` |",
         f"| model | `{a.model}` |", f"| fixes applied | `{a.fixes}` |",
         f"| seed | {a.seed} (identical in both arms) |",
         f"| generation | hybrid, do_sample=True, temp={a.temperature}, "
         f"top_p={a.top_p}, max_new_tokens={a.max_new_tokens} |",
         f"| whole-frame threshold | {a.frame_threshold:.2f} |", ""]
    if S and F:
        both = [(next(x for x in S if x["image"] == r["image"] and x["query"] == r["query"]), r)
                for r in Fok
                if any(x["image"] == r["image"] and x["query"] == r["query"]
                       and x["status"] == "ok" for x in S)]
        bS = sum(x["seconds"] for x, _ in both); bF = sum(y["seconds"] for _, y in both)
        L += ["## Headline\n", "| metric | stock | fixed |", "|---|---:|---:|",
              f"| completed | {len(Sok)}/{len(S)} | **{len(Fok)}/{len(F)}** |",
              f"| OOM | {sum(1 for r in S if r['status']=='OOM')} | "
              f"**{sum(1 for r in F if r['status']=='OOM')}** |",
              f"| wall clock (own cells) | {sum(r['seconds'] for r in Sok):.2f}s | "
              f"{sum(r['seconds'] for r in Fok):.2f}s |"]
        if both and bF:
            L.append(f"| wall clock on the {len(both)} cells both finished | {bS:.2f}s | "
                     f"**{bF:.2f}s** ({bS/bF:.2f}x) |")
        if Sok and Fok:
            L.append(f"| worst peak memory | {max(r['peak_mb'] for r in Sok):,.0f} MB | "
                     f"**{max(r['peak_mb'] for r in Fok):,.0f} MB** |")
        hits = [r["cache_hits"] for r in F if r["cache_hits"] is not None]
        if hits:
            L.append(f"| vision encodes | {len(F)} | **{len(F)-max(hits)}** ({max(hits)} cache hits) |")
        L.append("")
    L += ["## Per image (best-resolving query, same query both arms)\n",
          "| image | resolution | MP | patches | query | stock | fixed | speedup | found |",
          "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in per_image:
        st = "**OOM**" if r["stock_status"] == "OOM" else (
             f"{r['stock_s']:.2f}s" if r["stock_s"] is not None else "—")
        sp = (f"{r['stock_s']/r['fixed_s']:.2f}x"
              if r["stock_s"] and r["fixed_s"] else "—")
        fd = r["fixed_found"] if r["fixed_found"] is not None else r["stock_found"]
        L.append(f"| `{r['image']}` | {r['w']}x{r['h']} | {r['mp']:.2f} | "
                 f"{r['patches'] or 0:,} | \"{r['best_query']}\" | {st} | "
                 f"{r['fixed_s']:.2f}s | {sp} | {fd} |"
                 if r["fixed_s"] is not None else
                 f"| `{r['image']}` | {r['w']}x{r['h']} | {r['mp']:.2f} | "
                 f"{r['patches'] or 0:,} | \"{r['best_query']}\" | {st} | — | — | {fd} |")
    L += ["", "## Per query\n", "| query | arm | found | mean s | whole-frame collapses |",
          "|---|---|---:|---:|---:|"]
    for q in [x.strip() for x in a.queries.split(",") if x.strip()]:
        for arm in arms:
            rs = [r for r in rows if r["query"] == q and r["arm"] == arm and r["status"] == "ok"]
            if not rs: continue
            L.append(f"| \"{q}\" | {arm} | {sum(r['boxes_usable'] for r in rs)} | "
                     f"{sum(r['seconds'] for r in rs)/len(rs):.2f} | "
                     f"{sum(r['boxes_raw']-r['boxes_usable'] for r in rs)} |")
    L += ["", "## Notes\n",
          f"- \"found\" counts boxes covering <= {a.frame_threshold:.0%} of the frame; "
          "whole-frame output is treated as a non-answer.",
          "- Sampling is required: greedy decoding never terminates on this model's MTP paths.",
          "- The vision cache only hits because the loop is image-major.",
          "- The fixes change speed and memory, not what the model knows.", ""]
    (outd/"summary.md").write_text("\n".join(L))


def main():
    a = build_args().parse_args()
    if a._arm:
        return run_arm(a)
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    imgs = list_images(a.images, a.limit)
    if not imgs:
        sys.exit(f"no images found in {a.images}")
    qs = [q.strip() for q in a.queries.split(",") if q.strip()]
    print(f"{len(imgs)} images x {len(qs)} queries x {len(arms)} arms "
          f"= {len(imgs)*len(qs)*len(arms)} inferences\n")
    for arm in arms:
        for attempt in range(1, 7):          # OOM can kill the process; resume
            cmd = [sys.executable, __file__, *sys.argv[1:], "--_arm", arm]
            r = subprocess.run(cmd)
            n = sum(1 for _ in (Path(a.out)/f"raw_{arm}.jsonl").open()) \
                if (Path(a.out)/f"raw_{arm}.jsonl").exists() else 0
            if n >= len(imgs)*len(qs):
                break
            print(f"[{arm}] {n}/{len(imgs)*len(qs)} done, restarting (attempt {attempt})")
    render(a, arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
