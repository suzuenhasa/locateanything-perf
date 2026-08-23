"""Shared helpers for the LocateAnything runtime path.

These lived in archive/scripts/task_sweep.py, a 953-line benchmark harness, so serve.py
and tile_ocr.py -- the two things you actually run -- imported a benchmark to
build a prompt string. They belong in a module; this is that module.

Nothing here depends on the harness, the test corpus, or any measurement code.
"""
import math
import re

from PIL import Image


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp", ".gif", ".tif", ".tiff"}

TASK_PROMPTS = {
    # --- in the Space ---
    "Detection":  "Locate all the instances that matches the following description: {cats}.",
    "Grounding":  "Locate all the instances that match the following description: {cats}.",
    "OCR":        "Detect all the text in box format.",
    "GUI":        "Locate the region that matches the following description: {cats}.",
    "Pointing":   "Point to: {cats}.",
    # --- model card only ---
    "GroundingOne": "Locate a single instance that matches the following description: {cats}.",
    "TextGrounding": "Please locate the text referred as {cats}.",
    "Layout":     "Locate all the instances that matches the following description: {cats}.",
}

TASK_ORDER = ["Detection", "Grounding", "OCR", "GUI", "Pointing",
              "GroundingOne", "TextGrounding", "Layout"]

NEEDS_CATEGORY = set(TASK_ORDER) - {"OCR"}

DEFAULT_CATEGORY = "all the objects"

REF_OR_BOX = re.compile(r"(<ref>.*?</ref>)|(<box>.*?</box>)", re.IGNORECASE | re.DOTALL)

NUM = re.compile(r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>")


def build_prompt(task, category):
    """app.py:445, including the '</c>' joiner for multiple categories."""
    if task == "OCR":
        return TASK_PROMPTS["OCR"]
    cats = "</c>".join(c.strip() for c in category.split(",") if c.strip()) or "objects"
    return TASK_PROMPTS[task].format(cats=cats)


def parse_mixed_results(text, category_str=""):
    results = []
    expected = [c.strip().lower() for c in category_str.split("</c>") if c.strip()]
    current_label, found = None, False

    for m in REF_OR_BOX.finditer(text):
        token = m.group(0)
        if token.lower().startswith("<ref>"):
            raw = re.sub(r"</?ref>", "", token, flags=re.IGNORECASE).strip()
            if raw:
                current_label = raw
        else:
            content = re.sub(r"</?box>", "", token, flags=re.IGNORECASE)
            coords = [float(n) for n in NUM.findall(content)]
            if not coords:
                continue
            label = current_label if current_label is not None else (
                expected[0] if expected else "object")
            if len(coords) == 4:
                results.append({"type": "box", "coords": coords, "label": label})
            elif len(coords) == 2:
                results.append({"type": "point", "coords": coords, "label": label})
            found = True

    if found:
        return results

    # fallback: bare <box> runs with no <ref> labels
    for i, part in enumerate(re.split(r"<box>(.*?)</box>", text)):
        if i % 2 == 0:
            continue
        coords = [float(n) for n in NUM.findall(part)]
        label = expected[0] if expected else "object"
        if len(coords) == 4:
            results.append({"type": "box", "coords": coords, "label": label})
        elif len(coords) == 2:
            results.append({"type": "point", "coords": coords, "label": label})
    return results


def resize_short_side(image, short):
    """app.py:143, but WITHOUT the min(short_size, 1024) clamp at app.py:481 --
    running past that clamp is the entire point of this repo."""
    from PIL import Image
    w, h = image.size
    if w <= h:
        nw = short; nh = int(h * (short / w))
    else:
        nh = short; nw = int(w * (short / h))
    return image.resize((nw, nh), Image.BILINEAR)


def short_for_patches(w, h, target, patch_size=14):
    """Short side that lands this aspect ratio on roughly `target` patches."""
    ar = max(w, h) / max(1, min(w, h))
    return max(2 * patch_size, int(round(math.sqrt(target / ar) * patch_size)))


def parse_out_info(out_info):
    r"""The model's own stats line (modeling_locateanything.py:524).

    app.py:328 strips the prefix with `^[Ss]tast?ic\s*[Ii]nfo` -- which does not
    actually match the "Statistic Info," the model emits, so the demo's own
    parser mangles the first key into "Statistic Info, num_tokens". Splitting
    each key on its last comma fixes it without breaking the other fields.
    """
    stats = {}
    if not out_info:
        return stats
    for part in out_info.strip().split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.split(",")[-1].strip()          # drop any "Statistic Info," prefix
        v = v.strip()
        try:
            stats[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            stats[k] = v
    return stats


def parse_rungs(a):
    """-> [(kind, value, label)], kind 's' = short side, 'p' = patch budget.

    A short-side ladder is NOT comparable across images. The processor rescales
    anything above in_token_limit=25,600 patches
    (image_processing_locateanything.py:52-55), and the short side that
    saturates that budget depends entirely on aspect ratio: 1166 px for a 3.7:1
    results table, 1718 px for a 1.7:1 photo, 1939 px for a 3:4 page. Ask for
    1680 and 2240 on the table and you get the same clamped run twice, with a
    tidy-looking table that is measuring one rung.

    A patch ladder puts every image on the same rung, and lines up with the
    README's own 5,476 / 10,000 / 14,400 / 25,600 rows.
    """
    if a.patches.strip():
        return [("p", int(v), f"p{int(v)}")
                for v in a.patches.split(",") if v.strip()]
    out = []
    for v in a.short_side.split(","):
        v = v.strip().lower()
        if not v:
            continue
        if v in ("native", "0", "none"):
            out.append(("s", None, "native"))
        else:
            out.append(("s", int(v), f"s{int(v)}"))
    return out


def prepare_image(base, kind, val):
    if kind == "s":
        return base if val is None else resize_short_side(base, val)
    w, h = base.size
    return resize_short_side(base, short_for_patches(w, h, val))
