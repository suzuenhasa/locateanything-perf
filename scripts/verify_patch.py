#!/usr/bin/env python3
"""Confirm locateanything_fix is live on the path the model actually takes,
and that the swapped attention is numerically equivalent to the original.

    python verify_patch.py --model /path/to/LocateAnything-3B

Worth running before trusting any benchmark: transformers imports custom code
from its own HF_HOME cache, so a patch written to the model directory can look
applied while never executing.
"""
import argparse, sys, torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/LocateAnything-3B")
ap.add_argument("--seq", type=int, default=1024, help="sequence length for the equivalence check")
ap.add_argument("--tol", type=float, default=0.05, help="max abs diff allowed (bf16)")
a = ap.parse_args()

import locateanything_fix

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0))
try:
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True)
except TypeError:
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to("cuda").eval()
print("model loaded:", type(model).__name__)

print("is_applied() before:", locateanything_fix.is_applied())
locateanything_fix.apply()
print("is_applied() after :", locateanything_fix.is_applied())
locateanything_fix.verify(model)

# Same shape the ViT actually uses: [S, heads, D], single packed sequence.
vit = locateanything_fix._find_vit_module()
orig = vit.sdpa_attention
fixed = vit.VL_VISION_ATTENTION_FUNCTIONS["sdpa"]

H = getattr(model.config.vision_config, "num_attention_heads", 16)
D = max(1, getattr(model.config.vision_config, "hidden_size", 1152) // H)
S = a.seq
torch.manual_seed(0)
q, k, v = (torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
with torch.no_grad():
    out_o = orig(q, k, v, cu, cu)
    out_f = fixed(q, k, v, cu, cu)
diff = (out_o.float() - out_f.float()).abs().max().item()
print(f"equivalence: max|orig-fixed| = {diff:.6f}  shape {tuple(out_o.shape)} "
      f"(S={S}, heads={H}, head_dim={D})")

ok = diff < a.tol
print("PATCH_VERIFIED" if ok else f"PATCH_MISMATCH {diff} >= {a.tol}")
sys.exit(0 if ok else 1)
