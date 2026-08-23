"""One command, any GPU: produce a directly comparable record of the defect.

Run this on each machine you want to compare. It emits the same measurements in
the same shape, tagged with everything needed to interpret them, so a 3090 result
and an H100 result can be put side by side without asterisks.

    python crossbox.py                      # kernel probe only — needs just torch
    python crossbox.py --model /path/to/LocateAnything-3B    # also end-to-end

WHY IT IS BUILT THIS WAY

Comparing GPUs invites two mistakes and this script forecloses both.

1. **Compare equal patch counts, not equal pixel sizes.** The defect scales with
   patches², so "a 1680px image" means different work at different aspect ratios.
   Every arm here is defined by a target patch count and the script reports the
   count it actually achieved.

2. **Report peak memory ABOVE resident weights.** Absolute peak conflates the
   model (identical everywhere) with the activation cost (what we are measuring),
   and cards with more VRAM let the allocator behave differently.

It also records driver, torch, CUDA and compute capability, because "does this
reproduce on newer torch" and "does 80 GB hide it" are both live questions and
both need the stack pinned to the number.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


def machine_record():
    props = torch.cuda.get_device_properties(0)
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        drv = "?"
    try:
        import flash_attn
        fa = flash_attn.__version__
    except ImportError:
        fa = None
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_mb": round(props.total_memory / 2**20),
        "sm_count": props.multi_processor_count,
        "driver": drv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "flash_attn": fa,
    }


def probe_kernels(heads=16, head_dim=72, patch_counts=(5476, 10000, 14400, 25600)):
    """The defect in isolation: 3-D vs 4-D SDPA. No weights, no model."""
    rows = []
    for S in patch_counts:
        torch.manual_seed(0)
        try:
            q = torch.randn(S, heads, head_dim, device="cuda", dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
        except torch.OutOfMemoryError:
            rows.append({"patches": S, "error": "OOM allocating inputs"})
            torch.cuda.empty_cache()
            continue
        base = torch.cuda.memory_allocated()

        def run(four_d):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            qq, kk, vv = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            if four_d:
                qq, kk, vv = qq.unsqueeze(0), kk.unsqueeze(0), vv.unsqueeze(0)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                F.scaled_dot_product_attention(qq, kk, vv, None, dropout_p=0.0)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0,
                    (torch.cuda.max_memory_allocated() - base) / 2**20)

        row = {"patches": S}
        for label, four_d in (("three_d", False), ("four_d", True)):
            try:
                run(four_d)                       # warm
                dt, pk = run(four_d)
                row[label] = {"secs": dt, "peak_mb": pk}
            except torch.OutOfMemoryError as e:
                msg = str(e)
                amt = (msg.split("Tried to allocate")[1].split("of which")[0].strip()
                       if "Tried to allocate" in msg else "?")
                row[label] = {"error": f"OOM {amt}"}
                torch.cuda.empty_cache()
        rows.append(row)
        del q, k, v
        torch.cuda.empty_cache()
    return rows


def end_to_end(model_path, patch_targets=(5476, 10000, 14400, 25600)):
    """The same thing through the real model, if weights are available."""
    from PIL import Image, ImageDraw
    from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import locateanything_fix as fix

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Pin the LLM: if flash_attn is importable transformers auto-selects
    # flash_attention_2, which modeling_qwen2.py does not implement.
    try:
        cfg.text_config._attn_implementation = "sdpa"
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, config=cfg, dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda").eval()
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 2**20

    # synthetic scene: no downloads, no third-party imagery, identical everywhere
    def scene(px):
        im = Image.new("RGB", (px, px), (235, 235, 240))
        d = ImageDraw.Draw(im)
        for (x, y, w, h, c) in [(0.10, 0.35, 0.28, 0.50, (196, 132, 74)),
                                (0.40, 0.10, 0.30, 0.78, (208, 150, 90)),
                                (0.72, 0.42, 0.24, 0.44, (184, 120, 66))]:
            d.ellipse([x * px, y * px, (x + w) * px, (y + h) * px], fill=c)
        return im

    Q = "Locate all the instances that matches the following description: cat."
    rows = []
    for target in patch_targets:
        px = int((target ** 0.5) * 14)            # square image giving ~target patches
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": scene(px)}, {"type": "text", "text": Q}]}]
        text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ims, vids = proc.process_vision_info(msgs)
        inp = proc(text=[text], images=ims, videos=vids, return_tensors="pt").to("cuda")
        actual = int(inp["pixel_values"].shape[0])

        def once():
            torch.manual_seed(1234)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16),
                               input_ids=inp["input_ids"],
                               attention_mask=inp["attention_mask"],
                               image_grid_hws=inp.get("image_grid_hws"), tokenizer=tok,
                               max_new_tokens=48, use_cache=True,
                               generation_mode="hybrid", do_sample=True,
                               temperature=0.7, top_p=0.9, repetition_penalty=1.1,
                               verbose=False)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0,
                    torch.cuda.max_memory_allocated() / 2**20 - weights_mb)

        row = {"requested_patches": target, "actual_patches": actual, "px": px}
        for label, apply_fix in (("stock", False), ("fixed", True)):
            # Set the arm explicitly every iteration. apply() mutates a
            # module-level dict, so simply not calling it is NOT the same as the
            # stock path once a previous row has applied it.
            if apply_fix:
                if not fix.is_applied():
                    fix.apply(verbose=False)
            else:
                fix.revert(verbose=False)
            assert fix.is_applied() == apply_fix, "arm did not take"
            try:
                dt, pk = once()
                row[label] = {"secs": dt, "peak_above_weights_mb": pk}
            except torch.OutOfMemoryError as e:
                msg = str(e)
                amt = (msg.split("Tried to allocate")[1].split("of which")[0].strip()
                       if "Tried to allocate" in msg else "?")
                row[label] = {"error": f"OOM {amt}"}
                torch.cuda.empty_cache()
        rows.append(row)
        del inp
    return {"weights_mb": weights_mb, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to LocateAnything-3B; omit for kernel probe only")
    ap.add_argument("--out", default="crossbox.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA")
        return 1

    rec = machine_record()
    print("machine")
    for k, v in rec.items():
        print(f"  {k:<20} {v}")
    if rec["flash_attn"]:
        print("\n  WARNING: flash_attn is installed. It changes which kernel the ViT\n"
              "  takes AND flips the batch engine into packing mode, so the 'stock'\n"
              "  arm will not reflect a normal install. Uninstall before comparing.")
    print()

    print("kernel probe — 3-D vs 4-D SDPA, no model")
    print(f"  {'patches':>9}{'3-D':>22}{'4-D':>22}{'ratio':>10}")
    probe = probe_kernels()
    for r in probe:
        if "error" in r:
            print(f"  {r['patches']:>9}  {r['error']}")
            continue
        def fmt(d):
            return d["error"] if "error" in d else f"{d['secs']*1000:.1f}ms/{d['peak_mb']:.0f}MB"
        ratio = ("-" if ("error" in r["three_d"] or "error" in r["four_d"])
                 else f"{r['three_d']['peak_mb'] / max(r['four_d']['peak_mb'], 1e-9):.0f}x")
        print(f"  {r['patches']:>9}{fmt(r['three_d']):>22}{fmt(r['four_d']):>22}{ratio:>10}")

    out = {"machine": rec, "kernel_probe": probe}

    if args.model:
        print("\nend-to-end through the model")
        e2e = end_to_end(args.model)
        out["end_to_end"] = e2e
        print(f"  weights resident {e2e['weights_mb']:.0f} MB")
        print(f"  {'patches':>9}{'stock':>24}{'fixed':>24}")
        for r in e2e["rows"]:
            def fmt(d):
                return d["error"] if "error" in d else \
                    f"{d['secs']:.2f}s/{d['peak_above_weights_mb']:.0f}MB"
            print(f"  {r['actual_patches']:>9}{fmt(r['stock']):>24}{fmt(r['fixed']):>24}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}   (send this file back for cross-machine comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
