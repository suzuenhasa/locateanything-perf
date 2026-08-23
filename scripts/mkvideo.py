#!/usr/bin/env python3
"""Draw per-frame detections onto extracted frames and re-encode to mp4.

    python mkvideo.py --frames ./frames30 --results ./video30/raw_fixed.jsonl \
                      --query "black cat" --fps 30 --out ./detected.mp4

Frames are independent, so output framerate is limited only by how many frames
you ran detection on -- extract at native fps and run them all for a smooth clip.
"""
import argparse, json, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True, help="directory of extracted frames")
ap.add_argument("--results", required=True, help="raw_*.jsonl from ab_sweep.py")
ap.add_argument("--query", required=True)
ap.add_argument("--fps", type=float, default=30.0, help="output framerate")
ap.add_argument("--out", required=True)
ap.add_argument("--color", default="#00e5ff")
ap.add_argument("--no-hud", action="store_true")
ap.add_argument("--hold", type=int, default=1,
                help="repeat each detection across N output frames "
                     "(detect at 10fps, render at 30fps -> --hold 3)")
ap.add_argument("--render-frames", default="",
                help="full-rate/full-res frames to draw on; boxes are rescaled "
                     "from the detection frame size. defaults to --frames")
a = ap.parse_args()

rows = {}
for l in open(a.results):
    if l.strip():
        r = json.loads(l)
        if r["query"] == a.query:
            rows[r["image"]] = r
if not rows:
    raise SystemExit(f"no results for query {a.query!r} in {a.results}")

def fnt(s, b=True):
    n = "DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", "C:/Windows/Fonts/"):
        try: return ImageFont.truetype(d+n, s)
        except Exception: pass
    return ImageFont.load_default()

det_names = sorted(rows)
rdir = Path(a.render_frames or a.frames)
render_names = sorted(p.name for p in rdir.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
if a.hold > 1:
    # output frame i is covered by detection i // hold
    names = render_names[:len(det_names) * a.hold]
    pick = lambda i: det_names[min(i // a.hold, len(det_names) - 1)]
else:
    names = det_names
    pick = lambda i: det_names[i]
tot = hits = 0
with tempfile.TemporaryDirectory() as td:
    for i, n in enumerate(names):
        r = rows[pick(i)]
        im = Image.open(rdir/n).convert("RGB")
        # detection ran on a possibly smaller frame; map boxes onto this one
        bs = im.width / r.get("w", im.width)
        d = ImageDraw.Draw(im)
        boxes = r.get("usable", []) if r["status"] == "ok" else []
        tot += len(boxes); hits += 1 if boxes else 0
        for j, b0 in enumerate(boxes):
            b = [v * bs for v in b0]
            # The model can emit a box with the corners the wrong way round --
            # measured on this clip, 2 of 138 had x1 < x0 -- and PIL raises
            # "x1 must be greater than or equal to x0" rather than drawing it.
            # ab_sweep's parser sorts each pair; do the same here so one bad box
            # cannot take down a whole render at the last frame.
            b = [min(b[0], b[2]), min(b[1], b[3]),
                 max(b[0], b[2]), max(b[1], b[3])]
            if b[2] - b[0] < 1 or b[3] - b[1] < 1:
                continue
            d.rectangle(b, outline=a.color, width=4)
            lab = f"{a.query} {j+1}"
            f = fnt(20); bb = d.textbbox((0,0), lab, font=f)
            d.rectangle([b[0], b[1]-(bb[3]-bb[1])-10,
                         b[0]+(bb[2]-bb[0])+12, b[1]], fill=a.color)
            d.text((b[0]+6, b[1]-(bb[3]-bb[1])-8), lab, fill="#00201f", font=f)
        if not a.no_hud:
            d.rectangle([0,0,im.width,54], fill="#0f1216")
            d.text((10,5), f'"{a.query}"   frame {i+1}/{len(names)}   t={i/a.fps:.2f}s',
                   fill="#ffffff", font=fnt(19))
            sub = (f"{len(boxes)} detected   {r.get('seconds',0):.3f}s/frame"
                   + (f"   detect@{a.fps/a.hold:.0f}fps held x{a.hold}" if a.hold > 1 else ""))
            d.text((10,29), sub, fill="#93a0ae", font=fnt(14, False))
        im.save(Path(td)/f"o{i:05d}.jpg", quality=90)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(a.fps),
                    "-i",str(Path(td)/"o%05d.jpg"),"-c:v","libx264","-pix_fmt","yuv420p",
                    "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2","-r",str(a.fps),a.out],
                   check=True)
print(f"{len(names)} frames @ {a.fps}fps -> {a.out}")
print(f"  {tot} detections across {hits}/{len(names)} frames "
      f"({hits/len(names)*100:.0f}% of frames had a hit)")
