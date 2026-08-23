#!/usr/bin/env python3
"""Resident LocateAnything engine: pay the expensive setup once, serve warm.

Everything this repo has measured points at one deployment shape:

    model load                     5 s    once
    locateanything_fix.apply()    ~0 s    once, process-global
    torch.compile + warmup       178 s    once, worth 1.20-1.27x
    tiling + batching              -      per request, worth 2-8x

Only the last is per-request, so a long-lived process amortises the rest.
Break-even against eager is ~185 pages: compile saves 0.93 s/page and costs
171 s more at startup. Below that, run --no-compile.

Two things had to be got right before any of this was measurable.

RECOMPILATION. The shape that varies is the BATCH dimension, and it varies
during decode: rows retire as their tile hits im_end, so B walks 6,5,4,3,2,1.
It is NOT the KV length (dynamic=True makes that symbolic) and it is NOT the
vision dimensions -- the vision tower runs outside the compiled region and the
language model only ever sees the resulting token count. Warmup therefore feeds
HETEROGENEOUS synthetic pages, each tile region with a different amount of text,
across three aspect ratios. That takes recompiles-in-live-requests to zero,
round 0 included; density-only warmup left a 101.7 s trace inside the first
real request.

MEASUREMENT. Sampling makes the same page emit 39-112 boxes, which drowns any
A/B on wall seconds -- use --temperature 0. And compile changes float
accumulation, so even greedy the two arms emit different output; the invariant
unit is ms per forward, reported here alongside wall time.

    python scripts/serve.py --bench ./inbox --compile   --warmup 6 --temperature 0
    python scripts/serve.py --bench ./inbox --no-compile             --temperature 0
    python scripts/serve.py --bench ./inbox --compile --warmup 6 --strict
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from la_common import build_prompt, parse_mixed_results, resize_short_side, short_for_patches  # noqa: E402
from tile_ocr import tile_boxes, to_page, dedupe                               # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class Engine:
    """Load once, compile once, serve many."""

    def __init__(self, model_path, compile=True, tile_px=(448, 448), overlap=48,
                 max_new_tokens=1024, verbose=True, tile_mode="raw",
                 hostsync=True):
        self.tile_px = tile_px
        self.tile_mode = tile_mode
        self.overlap = overlap
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.stats = {"requests": 0, "tiles": 0, "recompiles_at_start": 0}

        # setdefault would let a stale inherited LA_FLASH_ATTN win, and
        # LA_FLASH_ATTN=la_flash routes language_model_forward through
        # _direct_base_forward -- a reimplementation of the layer loop that
        # calls neither lm.forward nor base.forward. The compiled wrapper would
        # then never be invoked: no error, no stall, recompiles stuck at 0, and
        # a service running at 1.0x while every health signal says warm. So
        # assign, do not default, and assert afterwards.
        os.environ["LA_FLASH_MODEL"] = model_path
        os.environ["LA_FLASH_ATTN"] = os.environ.get("LA_ATTN_OVERRIDE", "sdpa")
        os.environ["LA_FLASH_VISION_ATTN"] = "auto"
        os.environ["LA_FLASH_HYBRID_SCHEDULER"] = "pipeline"
        sys.path.insert(0, model_path)

        import torch
        self.torch = torch
        t0 = time.perf_counter()
        import batch_utils.hybrid_runtime as hr
        from batch_utils import generate_batch_hybrid, load
        self.hr, self._gen = hr, generate_batch_hybrid
        # the engine clamps every input to 1024 on the long edge and hardcodes
        # the detection template; both are module-level, so override here rather
        # than editing the model dir (trust_remote_code re-copies that anyway)
        hr.MAX_DIM = 1 << 30
        hr._PROMPT = ""
        if getattr(hr, "ATTN_MODE", "sdpa") != "sdpa":
            raise RuntimeError(
                f"attention route is {hr.ATTN_MODE!r}, not 'sdpa'. The compiled "
                f"wrapper on language_model.forward is bypassed by the "
                f"_direct_base_forward path, so compile would be a silent no-op.")
        self.tok, self.proc, self.model = load()
        t_load = time.perf_counter() - t0

        import locateanything_fix
        locateanything_fix.apply(verbose=False)
        try:
            locateanything_fix.enable_logits_slice(self.model, keep=6, verbose=False)
        except Exception as e:
            self._say(f"logits_slice unavailable: {e}")
        # the batch engine re-reads every row's prompt off the GPU on every
        # decode step; the prompt never changes. See archive/scripts/cpu_sync_probe.py.
        if hostsync:
            try:
                locateanything_fix.enable_batch_hostsync_fix(verbose=False)
            except Exception as e:
                self._say(f"hostsync fix unavailable: {e}")

        req = getattr(self.model, "_la_flash_requested_attn", hr.ATTN_MODE)
        if req in {"magi", "la_flash"}:
            raise RuntimeError(
                f"model._la_flash_requested_attn={req!r}: language_model_forward "
                f"routes to _direct_base_forward, which calls neither lm.forward "
                f"nor base.forward. torch.compile would be a silent no-op.")
        self._say(f"attention route: hr.ATTN_MODE={hr.ATTN_MODE} model={req}")

        self.compiled = False
        t_compile = 0.0
        if compile:
            t1 = time.perf_counter()
            self._configure_dynamo()
            lm = self.model.language_model
            self._orig_lm_forward = lm.forward
            lm.forward = torch.compile(lm.forward, dynamic=True)
            self.compiled = True
            t_compile = time.perf_counter() - t1
        self._say(f"loaded in {t_load:.1f}s"
                  + (f", compile wrapper installed in {t_compile:.1f}s "
                     f"(tracing happens on first request)" if compile else ", eager"))

    def _say(self, m):
        if self.verbose:
            print(f"[engine] {m}", flush=True)

    def _configure_dynamo(self):
        """Everything that must be set BEFORE the first trace."""
        import torch._dynamo as dynamo
        # The measured graph count for this service is ~31 across ~10 code
        # objects. The default per-code-object limit is 8; a code object that
        # hits it is silently abandoned to eager forever. Give real headroom.
        dynamo.config.cache_size_limit = 64
        dynamo.config.accumulated_cache_size_limit = 512
        # Duck shaping guesses that two dims which happen to be equal on the
        # first trace are ALWAYS equal, and writes that into the guard.
        # _pack_stock_kv_rows produces [B, num_kv_heads, kmax, head_dim] and
        # Qwen2.5-3B is GQA 16:2, so every request that drains to its last two
        # tiles hits B == num_kv_heads == 2 and bakes in a false equality that
        # then recompiles on every other B.
        try:
            import torch.fx.experimental._config as fx_cfg
            fx_cfg.use_duck_shape = False
        except Exception as e:
            self._say(f"could not disable duck shaping: {e}")

    def _dynamo_health(self):
        """Health, not activity.

        `counters["stats"]["unique_graphs"]` is the obvious metric and it is a
        trap: when a code object hits the recompile limit the counter FREEZES
        and the function runs eager from then on, so a delta of 0 means either
        "warm and stable" or "dynamo permanently gave up", with nothing to tell
        them apart. `gave_up` is the one that actually matters.
        """
        try:
            import torch._dynamo as dynamo
            c = dynamo.utils.counters
            return {
                "graphs": int(c["stats"].get("unique_graphs", 0)),
                "frames": int(c["frames"].get("total", 0)),
                "gave_up": int(c["unimplemented"].get("recompile_limit reached", 0)),
            }
        except Exception:
            return {"graphs": 0, "frames": 0, "gave_up": 0}

    def _recompiles(self):
        return self._dynamo_health()["graphs"]

    def warmup(self, rounds=3, tiles=8, task="OCR", category="text", verbose=True):
        """Trace the compiled graphs before the first real request.

        The first version of this varied text DENSITY only, on one page size.
        It did not work: after 241 s of warmup, the first real request on
        page2_test.png still spent 101.7 s tracing five more graphs. The reason
        is that the shape that actually varies is the BATCH dimension, and it
        varies during decode, not between requests. Rows retire as their tile
        hits im_end, so B walks 6,5,4,3,2,1 -- but a synthetic page with uniform
        text makes every tile finish at the same step, so B goes 6,6,6,...,0 and
        the intermediate values are never traced.

        So warmup pages must be HETEROGENEOUS: each tile region gets a different
        amount of text, which staggers the retirements and walks B through every
        value. Aspect ratio is varied too, because grid_for returns a different
        grid (and therefore a different starting B) for portrait vs landscape.
        """
        from PIL import Image, ImageDraw
        t0 = time.perf_counter()
        shapes = [(900, 1300), (1400, 900), (1100, 1100)]
        for i in range(rounds):
            W, H = shapes[i % len(shapes)]
            img = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(img)
            grid = self.grid_for(img, tiles)
            nx, ny = grid
            # a different line count per tile region, including an empty one,
            # so the rows retire at different steps
            for ti, (x0, y0, x1, y1) in enumerate(tile_boxes(W, H, nx, ny, self.overlap)):
                nl = (ti * 7) % 23          # 0, 7, 14, 21, 5, 12, ...
                for k in range(nl):
                    yy = y0 + 8 + k * max(12, (y1 - y0 - 16) // max(1, nl))
                    if yy > y1 - 14:
                        break
                    d.text((x0 + 8, yy),
                           f"warm {ti}-{k} invoice total amount due ref 12345",
                           fill="black")
            try:
                r = self.run(img, task=task, category=category, tiles=tiles,
                             temperature=0.0)
                if verbose:
                    self._say(f"warmup {i+1}/{rounds}: {r['seconds']:6.1f}s  "
                              f"{W}x{H} grid {r['grid']}  {r['n_boxes']:3d} boxes  "
                              f"{r['recompiles_delta']:2d} new graphs")
            except Exception as e:
                self._say(f"warmup {i+1} failed: {type(e).__name__}: {str(e)[:120]}")
        h = self._dynamo_health()
        self._say(f"warmup complete in {time.perf_counter()-t0:.1f}s "
                  f"({h['graphs']} graphs, gave_up={h['gave_up']})")

    def strict(self, on=True):
        """Turn any further recompile into a loud failure.

        Use in staging: replay real traffic with this on and you find out
        whether warmup actually covered the shape space, instead of finding out
        in production as a 100 s request. In production prefer eager_on_recompile,
        which degrades that request instead of stalling it.
        """
        try:
            import torch._dynamo as dynamo
            dynamo.set_stance("fail_on_recompile" if on else "default")
            self._say(f"dynamo stance: {'fail_on_recompile' if on else 'default'}")
        except Exception as e:
            self._say(f"could not set stance: {e}")

    def degrade_on_recompile(self):
        try:
            import torch._dynamo as dynamo
            dynamo.set_stance("eager_on_recompile")
            self._say("dynamo stance: eager_on_recompile")
        except Exception as e:
            self._say(f"could not set stance: {e}")

    def _tiles(self, img, grid):
        """Crop to a grid.

        An earlier version padded every tile to one canonical 448x448 so the
        compiled graph would see identical vision dimensions. That was wasted:
        the vision tower runs OUTSIDE the compiled region -- engine_hybrid runs
        it separately and hands `visual_features` to the language model, so the
        only thing the compiled forward ever sees is the resulting token count,
        which is an ordinary dynamic sequence-length dimension. Padding bought
        no shape stability at all, and on the inbox pages it cost 30% linear
        resolution (636x603 crops squeezed into 448x448 -- half the area) on a
        model whose whole point here is that it can take the full resolution.

        `tile_mode="pad"` keeps the old behaviour for comparison.
        """
        from PIL import Image
        W, H = img.size
        nx, ny = grid
        rects = tile_boxes(W, H, nx, ny, self.overlap)
        out = []
        for r in rects:
            c = img.crop(r)
            if self.tile_mode == "raw":
                out.append((c, r, c.size, c.size))
                continue
            tw, th = self.tile_px
            s = min(tw / c.size[0], th / c.size[1])
            c = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))),
                         Image.BILINEAR)
            canvas = Image.new("RGB", (tw, th), (255, 255, 255))
            canvas.paste(c, (0, 0))
            out.append((canvas, r, c.size, canvas.size))
        return out

    def grid_for(self, img, target_tiles=8):
        """Pick a grid whose tiles are roughly square, given the page aspect."""
        W, H = img.size
        best, bestscore = (1, 1), 1e9
        for nx in range(1, target_tiles + 1):
            for ny in range(1, target_tiles + 1):
                if nx * ny > target_tiles or nx * ny < max(2, target_tiles // 2):
                    continue
                ar = (W / nx) / (H / ny)
                score = abs(ar - 1.0) + 0.05 * abs(nx * ny - target_tiles)
                if score < bestscore:
                    best, bestscore = (nx, ny), score
        return best

    def run(self, image, task="OCR", category="text", grid=None, tiles=8,
            page_patches=10000, iou=0.55, temperature=0.7):
        from PIL import Image
        torch = self.torch
        t0 = time.perf_counter()
        h0 = self._dynamo_health()

        base = image if hasattr(image, "size") else Image.open(image).convert("RGB")
        if base.mode != "RGB":
            base = base.convert("RGB")
        page = resize_short_side(base, short_for_patches(*base.size, page_patches))
        W, H = page.size
        grid = grid or self.grid_for(page, tiles)
        packed = self._tiles(page, grid)

        prompt = build_prompt(task, category)
        query = prompt[:-1] if prompt.endswith(".") else prompt
        pairs = [(t[0], query) for t in packed]

        outs = self._gen(pairs, temperature=temperature, top_p=0.9,
                         repetition_penalty=1.1,
                         max_new_tokens=self.max_new_tokens, scheduler="pipeline")
        try:
            from batch_utils.engine_hybrid import get_last_hybrid_stats
            hs = get_last_hybrid_stats() or {}
        except Exception:
            hs = {}

        dets = []
        for (canvas, rect, used, canvas_px), o in zip(packed, outs):
            tw, th = canvas_px
            uw, uh = used
            for d in parse_mixed_results(str(o), ""):
                c = d["coords"]
                # canvas coords -> used-region coords -> page coords
                if len(c) == 4:
                    c = [c[0] * tw / uw, c[1] * th / uh, c[2] * tw / uw, c[3] * th / uh]
                elif len(c) == 2:
                    c = [c[0] * tw / uw, c[1] * th / uh]
                else:
                    continue
                if any(v > 1000 or v < 0 for v in c):      # spilled into the padding
                    continue
                m = to_page({**d, "coords": c}, rect, W, H)
                if m:
                    dets.append(m)
        dets = dedupe(dets, iou)

        dt = time.perf_counter() - t0
        self.stats["requests"] += 1
        self.stats["tiles"] += len(packed)
        h1 = self._dynamo_health()
        fwd = (int(hs.get("mtp_forwards", 0)) + int(hs.get("ar_forwards", 0))
               + int(hs.get("prompt_prefill_forwards", 0)))
        return {
            "seconds": round(dt, 3), "grid": f"{grid[0]}x{grid[1]}",
            "tiles": len(packed), "page_px": [W, H], "task": task,
            "n_boxes": sum(1 for d in dets if len(d["coords"]) == 4),
            "n_points": sum(1 for d in dets if len(d["coords"]) == 2),
            # wall seconds cannot compare compiled vs eager: output length
            # varies. Forwards is the invariant denominator.
            "forwards": fwd,
            "mtp_forwards": int(hs.get("mtp_forwards", 0)),
            "ar_forwards": int(hs.get("ar_forwards", 0)),
            "ms_per_forward": round(dt * 1000.0 / fwd, 3) if fwd else None,
            "recompiles_delta": h1["graphs"] - h0["graphs"],
            "gave_up": h1["gave_up"] - h0["gave_up"],
            "detections": [{"coords": [round(v, 1) for v in d["coords"]],
                            "label": d.get("label", "")} for d in dets],
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("LA_MODEL", "nvidia/LocateAnything-3B"))
    ap.add_argument("--bench", default="", help="directory of images to process in sequence")
    ap.add_argument("--task", default="OCR")
    ap.add_argument("--category", default="text")
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--tile-px", default="448x448")
    ap.add_argument("--tile-mode", choices=["raw", "pad"], default="raw",
                    help="raw: feed the crop at full resolution (default). "
                         "pad: squeeze into --tile-px. Padding was originally "
                         "for shape stability, but the vision tower is outside "
                         "the compiled region so it bought nothing.")
    ap.add_argument("--no-hostsync", dest="hostsync", action="store_false",
                    default=True, help="disable the host-sync fixes (control arm)")
    ap.add_argument("--strict", action="store_true",
                    help="after warmup, make any further recompile raise")
    ap.add_argument("--page-patches", type=int, default=10000)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=0,
                    help="synthetic pages to run at startup so dynamo finishes "
                         "tracing before the first real request. 3 is enough.")
    ap.add_argument("--rounds", type=int, default=2,
                    help="passes over the directory; round 2+ is the warm number")
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy. Sampling makes output length vary ~3x on "
                         "the same page, which drowns any A/B on wall seconds.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tw, th = (int(v) for v in a.tile_px.lower().split("x"))
    t_start = time.perf_counter()
    eng = Engine(a.model, compile=a.compile, tile_px=(tw, th),
                 max_new_tokens=a.max_new_tokens, tile_mode=a.tile_mode,
                 hostsync=a.hostsync)
    if a.warmup:
        eng.warmup(a.warmup, tiles=a.tiles, task=a.task, category=a.category)
    if a.strict:
        eng.strict(True)
    t_startup = time.perf_counter() - t_start

    if not a.bench:
        print("nothing to do: pass --bench DIR")
        return 0

    files = sorted(p for p in Path(a.bench).iterdir() if p.suffix.lower() in IMG_EXT)
    print(f"\n[engine] startup {t_startup:.1f}s  compile={a.compile}  "
          f"tile_mode={a.tile_mode}  {len(files)} images x {a.rounds} rounds\n")
    print(f"  {'round':>5} {'image':24s} {'grid':>5} {'sec':>8} {'boxes':>6} "
          f"{'fwd':>6} {'ms/fwd':>8} {'graphs':>7} {'gaveup':>7}")
    rows = []
    for rd in range(a.rounds):
        for f in files:
            r = eng.run(str(f), task=a.task, category=a.category,
                        tiles=a.tiles, page_patches=a.page_patches,
                        temperature=a.temperature)
            r["round"], r["image"] = rd, f.name
            rows.append(r)
            mpf = r["ms_per_forward"]
            print(f"  {rd:5d} {f.name[:24]:24s} {r['grid']:>5} {r['seconds']:8.3f} "
                  f"{r['n_boxes']:6d} {r['forwards']:6d} "
                  f"{(f'{mpf:8.2f}' if mpf else '       -')} "
                  f"{r['recompiles_delta']:7d} {r['gave_up']:7d}")
    import statistics
    for rd in range(a.rounds):
        sel = [r for r in rows if r["round"] == rd]
        if not sel:
            continue
        sec = sum(r["seconds"] for r in sel)
        fwd = sum(r["forwards"] for r in sel)
        print(f"  round {rd}: mean {statistics.mean([r['seconds'] for r in sel]):.3f}s"
              f"   {sec*1000/fwd:.2f} ms/forward over {fwd} forwards"
              if fwd else f"  round {rd}: mean "
              f"{statistics.mean([r['seconds'] for r in sel]):.3f}s")
    warm = [r for r in rows if r["round"] > 0]
    if warm and sum(r["forwards"] for r in warm):
        s_ = sum(r["seconds"] for r in warm); f_ = sum(r["forwards"] for r in warm)
        print(f"\n  WARM (round>=1): {s_*1000/f_:.2f} ms/forward  "
              f"[{f_} forwards, {s_:.1f}s]   compile={a.compile}")
    h = eng._dynamo_health()
    print(f"  dynamo: {h['graphs']} graphs, {h['frames']} frames, "
          f"gave_up={h['gave_up']}"
          + ("   <-- COMPILE DISABLED ITSELF" if h["gave_up"] else ""))
    print(f"\n  startup {t_startup:.1f}s amortises over "
          f"{len(files)*a.rounds} requests = "
          f"{t_startup/max(1,len(files)*a.rounds):.2f}s/request")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
