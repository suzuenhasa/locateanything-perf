"""Runtime fix for nvidia/LocateAnything-3B's vision attention.

Import this and call `apply()` after loading the model. It edits nothing on disk
and needs no fork of the model repo — it swaps the "sdpa" entry in the dynamically
imported module's attention table.

    from transformers import AutoModel, AutoTokenizer, AutoProcessor
    import locateanything_fix

    model = AutoModel.from_pretrained("nvidia/LocateAnything-3B",
                                      dtype=torch.bfloat16,
                                      trust_remote_code=True).to("cuda").eval()
    locateanything_fix.apply()          # <- after the model is loaded, not before

Why after: the custom code is not imported until the first
`from_pretrained(trust_remote_code=True)` call, and transformers imports it from
a copy under `HF_HOME/modules/transformers_modules/`, not from the model
directory. There is nothing to patch until that copy has been imported.

WHAT IT FIXES
-------------
`modeling_vit.sdpa_attention` passes 3-dimensional q/k/v to
`F.scaled_dot_product_attention`. PyTorch requires 4-D for every fused kernel:

    UserWarning: All fused kernels requires query, key and value to be 4
    dimensional, but got Query dim: 3 ...

so flash, mem-efficient and cuDNN are all disqualified and SDPA falls back to the
math backend, which materialises the full [heads, P, P] score matrix in fp32. For
one 1680px image that is a 12.36 GiB allocation; with a batch dimension the same
op peaks at 0.13 GB.

Two changes, both behaviour-preserving:

  1. Add a batch dimension so the fused kernels are eligible.
  2. When there is a single packed sequence, skip the mask entirely. The mask the
     original builds is `torch.zeros([1, S, S], dtype=bool)` filled to True over
     each sequence's own span -- with one sequence that is all-True, so it masks
     nothing, and a dense mask separately rules out the flash kernel. Multi-image
     packing keeps the original block-diagonal mask.

The computation is mathematically identical; only floating-point accumulation
order changes, because a tiled kernel reduces differently from the math backend.

DOES THIS NEED flash-attn? No. It needs nothing beyond torch.
`F.scaled_dot_product_attention` has a flash backend that supports Ampere and
ships inside torch; the model simply never reached it. Measured against a real
flash-attn build (FA2 2.8.3, sm_86): peak memory byte-identical and wall time
within 1% at every resolution from 5,476 to 25,600 patches. This patch reaches
the same kernel, not an approximation of it.

Do NOT install flash-attn to "help". With the wheel importable, transformers
auto-selects `flash_attention_2` for the language model, and modeling_qwen2.py
handles only `magi` and `sdpa` -- so the model card's own loading snippet dies
with NotImplementedError on the first forward. It also flips the batch engine
into a different vision-encoding path, which quietly invalidates any before/after
measurement taken on that machine.
"""

from __future__ import annotations

import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F

__all__ = ["apply", "revert", "is_applied", "verify", "enable_vision_cache",
           "enable_packed_vision", "enable_logits_slice", "disable_logits_slice",
           "enable_video_decode", "disable_video_decode", "VisionCache"]

_MARK = "_locateanything_sdpa_4d"


def _find_vit_module():
    """The dynamically imported modeling_vit, or None if the model isn't loaded."""
    for name, mod in list(sys.modules.items()):
        if "transformers_modules" in name and name.endswith("modeling_vit"):
            if hasattr(mod, "VL_VISION_ATTENTION_FUNCTIONS"):
                return mod
    return None


def _is_single_sequence(cu_seqlens, seq_length) -> bool:
    """True only for the conventional single-sequence form.

    `None` means nothing was packed -- upstream crashes on this (`len(None)`);
    the shape implies one sequence, so we treat it as one.

    Otherwise we require exactly the canonical [0, S]. A malformed or shorter
    cu_seqlens is deliberately NOT treated as single: upstream's loop
    `for i in range(1, len(cu_seqlens))` would leave the mask all-False there,
    and quietly substituting full attention would change behaviour on bad input
    instead of surfacing it. Those fall through to the mask path, as upstream.
    """
    if cu_seqlens is None:
        return True
    if len(cu_seqlens) != 2:
        return False
    return int(cu_seqlens[0]) == 0 and int(cu_seqlens[1]) == seq_length


def _sdpa_attention_4d(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    """Drop-in replacement for modeling_vit.sdpa_attention."""
    seq_length = q.shape[0]

    # [S, heads, D] -> [heads, S, D]; the batch dim added below is what makes the
    # fused kernels eligible at all.
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    if _is_single_sequence(q_cu_seqlens, seq_length):
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None, dropout_p=0.0
        ).squeeze(0)
    else:
        # Packed multi-image. Upstream builds a dense [1, S, S] bool mask, which
        # costs S^2 bytes for a matrix that is block-diagonal by construction.
        # Attending each segment separately is the same computation (a query
        # never attends outside its own sequence either way) with no mask tensor
        # at all, and every segment still gets a fused kernel. Measured on a
        # 3x6400 pack: 1098 MB -> 127 MB, outputs agreeing to under one bf16 ulp.
        #
        # Dead BY DEFAULT, live once you ask for it. Without enable_packed_vision()
        # the engine encodes each image separately and never packs: cu_seqlens is
        # [0, S] on every call (measured at 135 = 27 layers x 5 images, and again
        # at 216 = 27 x 8).
        #
        # enable_packed_vision() is the caller this signature anticipates. With it
        # on, that same 8-image run collapses to 27 calls carrying 8 packed
        # segments each -- scripts/segcheck.py measures exactly this. The two are
        # coupled: packing makes this branch live, and this branch is what makes
        # packing safe, since upstream refuses to pack without flash precisely
        # because the dense-mask path costs (sum S_i)^2 where this costs
        # sum(S_i^2). Worth +4.3% at batch 8 and +7.4% at batch 16.
        bounds = [int(x) for x in q_cu_seqlens]
        parts = []
        for i in range(1, len(bounds)):
            a, b = bounds[i - 1], bounds[i]
            if b <= a:
                continue
            parts.append(F.scaled_dot_product_attention(
                q[:, a:b].unsqueeze(0), k[:, a:b].unsqueeze(0),
                v[:, a:b].unsqueeze(0), None, dropout_p=0.0).squeeze(0))
        out = torch.cat(parts, dim=1) if parts else q.new_zeros(q.shape)

    return out.transpose(0, 1).reshape(seq_length, -1)


class VisionCache:
    """Reuse vision features across calls about the same image.

    `modeling_locateanything.py:209` calls `extract_feature()` on every
    `generate()`, so asking four questions about one image encodes it four times.
    This wraps that method and returns the previous result when the pixels match.

    Only worth enabling for repeat queries on the same image (an interactive
    session, a multi-task pipeline). A batch job over distinct images never hits
    the cache and pays the digest cost for nothing.

    Keying: a content digest — shape, dtype, and two reductions over the tensor —
    NOT the tensor's address. `data_ptr()` looks like an obvious key and is
    unsafe: the allocator reuses freed addresses, so a later image can land on a
    dead entry's pointer and silently receive another image's features. That is
    fine inside a benchmark that holds every input alive, and a real bug in a
    long-running process.

    The digest is not cryptographic. Two genuinely different images matching on
    shape, dtype, sum and sum-of-squares simultaneously is not credible in
    practice, but if you need a hard guarantee, pass your own key:

        cache.next_key = "path/to/image.jpg"   # consumed by the next call
    """

    def __init__(self, model, maxsize: int = 4, digest: str = "fast"):
        if digest not in ("fast", "exact"):
            raise ValueError("digest must be 'fast' or 'exact'")
        self.model = model
        self.maxsize = maxsize
        self.digest_mode = digest
        self._store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._original = model.extract_feature
        self.hits = 0
        self.misses = 0
        self.next_key = None
        self._warned = False
        model.extract_feature = self._wrapped

    def _digest(self, t: torch.Tensor):
        if self.next_key is not None:
            k, self.next_key = self.next_key, None
            return ("explicit", k)
        if self.digest_mode == "exact":
            # Cryptographic digest of the actual bytes: a device-to-host copy
            # plus a hash, tens of ms against an encode of hundreds, and the
            # collision question disappears.
            import hashlib
            b = t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
            return ("blake2b", tuple(t.shape), str(t.dtype),
                    hashlib.blake2b(b, digest_size=16).hexdigest())
        with torch.no_grad():
            f = t.float()
            return (tuple(t.shape), str(t.dtype),
                    round(f.sum().item(), 4), round(f.square().sum().item(), 4))

    @staticmethod
    def _copy(x):
        """Detach/clone tensors anywhere in the returned structure.

        extract_feature returns whatever the vision tower returns -- a tensor for
        some configs, a list of per-image tensors for others. Handing the same
        object back on every hit would let one call's downstream slicing or
        in-place work corrupt the next.
        """
        if torch.is_tensor(x):
            return x.detach().clone()
        if isinstance(x, (list, tuple)):
            return type(x)(VisionCache._copy(i) for i in x)
        return None                       # unrecognised: signal "do not cache"

    def _wrapped(self, pixel_values, image_grid_hws=None, *a, **kw):
        key = self._digest(pixel_values)
        hit = self._store.get(key)
        if hit is not None:
            self._store.move_to_end(key)
            self.hits += 1
            return self._copy(hit)
        self.misses += 1
        out = self._original(pixel_values, image_grid_hws, *a, **kw)
        stored = self._copy(out)
        if stored is None:
            # Loud, once. A cache that silently stores nothing looks exactly like
            # a cache that works and is one of the easier ways to fool yourself.
            if not self._warned:
                self._warned = True
                print(f"locateanything_fix: vision cache disabled -- extract_feature "
                      f"returned {type(out).__name__}, which this cache cannot copy "
                      f"safely. Nothing will be cached.")
            return out
        self._store[key] = stored
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return out

    def clear(self):
        self._store.clear()

    def disable(self):
        """Restore the original method and drop the cache."""
        self.model.extract_feature = self._original
        self.clear()
        if getattr(self.model, "_locateanything_vision_cache", None) is self:
            self.model._locateanything_vision_cache = None

    @property
    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disable()
        return False

    def __repr__(self):
        return (f"VisionCache(hits={self.hits}, misses={self.misses}, "
                f"entries={len(self._store)}/{self.maxsize})")


def enable_vision_cache(model, maxsize: int = 4, digest: str = "fast") -> VisionCache:
    """Cache vision features on `model`. Returns the cache, for stats/disable.

        cache = locateanything_fix.enable_vision_cache(model)
        ...                       # several generate() calls on one image
        print(cache.stats)        # {'hits': 3, 'misses': 1, 'entries': 1}

    Also usable as a context manager, restoring the original method and freeing
    the cached features on exit:

        with locateanything_fix.enable_vision_cache(model) as cache:
            ...

    `digest="exact"` hashes the pixel bytes rather than using the cheap
    shape+sum digest, removing the collision question at the cost of a
    device-to-host copy per call.

    Each entry holds one image's encoder output on the GPU — tens of MB at high
    resolution — so `maxsize` is a memory knob, not just a hit-rate one.
    """
    if getattr(model, "_locateanything_vision_cache", None) is not None:
        return model._locateanything_vision_cache
    cache = VisionCache(model, maxsize=maxsize, digest=digest)
    model._locateanything_vision_cache = cache
    return cache


class _SlicedLMHead(torch.nn.Module):
    """lm_head that only projects the rows the caller will actually read.

    `modeling_qwen2.py:1495` projects every prompt position through a
    152,681-wide head and upcasts all of it to fp32, while the callers keep only
    the tail:

        modeling_locateanything.py:432   outputs.logits[:, -1:, :]      (AR)
        modeling_locateanything.py:415   outputs.logits[:, -n_future:, :]  (MTP, default 6)

    At batch 1 this is invisible -- the vision encoder sets the peak. In the
    batch engine it is the binding constraint: at batch 8, ~13k tokens x 152,681
    x 4 B is a 7 GiB allocation, and it OOMs a 24 GB card. Measured at batch 4,
    peak 14428 MB -> 10652 MB; at batch 8, OOM -> works.

    The repo already contains this fix (hybrid_runtime.py:1287 passes
    logits_slice) but only applies it when attention is magi/la_flash. The
    default sdpa path goes through `lm(**kwargs)` and never gets it.

    Only slices with grad disabled, so a training forward -- which needs every
    position for the loss -- is untouched.
    """

    def __init__(self, inner, keep: int):
        super().__init__()
        self.inner = inner
        self.keep = keep

    def forward(self, hidden_states):
        if (not torch.is_grad_enabled()) and hidden_states.dim() == 3 \
                and hidden_states.shape[1] > self.keep:
            hidden_states = hidden_states[:, -self.keep:, :]
        return self.inner(hidden_states)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


def enable_logits_slice(model, keep: int = 6, verbose: bool = True) -> bool:
    """Stop projecting prompt positions whose logits are immediately discarded.

        locateanything_fix.enable_logits_slice(model)

    `keep` must be >= the largest tail any caller reads. This model's MTP path
    reads the last `n_future_tokens` (default 6) and the AR path the last 1, so 6
    is the safe minimum; raise it if you change n_future_tokens.
    """
    lm = getattr(model, "language_model", None)
    if lm is None or not hasattr(lm, "lm_head"):
        raise RuntimeError("no language_model.lm_head to wrap")
    if isinstance(lm.lm_head, _SlicedLMHead):
        return False
    lm.lm_head = _SlicedLMHead(lm.lm_head, keep)
    if verbose:
        print(f"locateanything_fix: lm_head projects the last {keep} positions only")
    return True


def disable_logits_slice(model) -> bool:
    lm = getattr(model, "language_model", None)
    if lm is not None and isinstance(getattr(lm, "lm_head", None), _SlicedLMHead):
        lm.lm_head = lm.lm_head.inner
        return True
    return False


def enable_packed_vision(verbose: bool = True) -> bool:
    """Let the batch engine encode several images in ONE vision forward.

    `batch_utils/hybrid_runtime.py` already knows how to pack images -- MoonViT's
    cu_seqlens path is block-diagonal by image -- but it refuses unless flash_attn
    is installed. Its own docstring says why:

        "If the vision blocks are on sdpa/eager, OR the flash wheel is absent
         (multihead_attention falls back to the dense-mask sdpa path), packing is
         O(S^2) N^2 -> caller must stay per-image."

    That premise is false once `apply()` is in effect: the segmented SDPA path
    costs sum(S_i^2), not (sum S_i)^2, which is exactly the block-diagonal cost
    flash achieves. So the gate can be opened, and packing becomes available
    without the flash wheel.

    Requires apply() first -- without it, opening this gate would restore the
    very O(S^2) blow-up the gate exists to prevent.
    """
    if not is_applied():
        raise RuntimeError("call apply() first: packing is only safe once the "
                           "segmented SDPA path is active")
    try:
        from batch_utils import hybrid_runtime
    except ImportError as e:
        raise RuntimeError(
            "batch_utils is not importable; add the model directory to sys.path"
        ) from e

    if getattr(hybrid_runtime._vision_is_flash, "_locateanything_packed", False):
        return False

    def _packed_ok():
        return True
    _packed_ok._locateanything_packed = True
    hybrid_runtime._orig_vision_is_flash = hybrid_runtime._vision_is_flash
    hybrid_runtime._vision_is_flash = _packed_ok
    if verbose:
        print(f"locateanything_fix: packed vision encoding enabled "
              f"(micro-batch {hybrid_runtime.VISION_ENCODE_BATCH_SIZE})")
    return True


def _find_processing_module():
    """The dynamically imported processing_locateanything, or None."""
    for name, mod in list(sys.modules.items()):
        if "transformers_modules" in name and name.endswith("processing_locateanything"):
            if hasattr(mod, "VIDEO_READER_BACKENDS"):
                return mod
    return None


def enable_video_decode(verbose: bool = True) -> bool:
    """Make video input decode at all on torchvision >= 0.19.

    `process_vision_info()` defaults to `video_reader_backend="torchvision"`, and
    `_read_video_torchvision` calls `torchvision.io.read_video`, which was removed
    in 0.19. Every video request dies with

        module 'torchvision.io' has no attribute 'read_video'

    and the error is swallowed into a log line that says it is falling back to
    torchvision -- which is what just failed. The decord reader in the same module
    works; nothing selects it unless the caller passes the backend explicitly.

    This points the "torchvision" entry at the decord reader when, and only when,
    torchvision cannot actually do the job. No new dependency: decord is already
    in requirements-model.txt.

    Call it any time after the processor's custom code is imported (i.e. after the
    first AutoProcessor.from_pretrained(..., trust_remote_code=True)).

    Verified, not assumed: decord's output was compared against ffmpeg on seven
    probes of a 540-frame CFR 30fps H.264 clip -- frame 0, frame 1, 50, 137, 269,
    400 and the last frame. Every one was bit-exact (mean|diff| 0.000, max|diff| 0,
    0.00% of pixels off by >8), and frame count and fps matched. So no off-by-one,
    no seek drift, no colour-space error.

    That test covers the easy case, and the limit is worth stating. decord is
    abandoned -- 0.6.0 is the last release, from 2022, with no py3.13 wheels -- and
    it has a reputation for seek inaccuracy on variable-frame-rate and some
    long-GOP files. A VFR phone capture or a file with broken timestamps is where
    it historically drifts, and the probe above would not have caught that.

    decord is still the right choice *here*, for three reasons that are about this
    patch rather than about decoders in general:

      1. it is already registered in the model's own VIDEO_READER_BACKENDS, so
         this is a pointer swap with no new code path to validate;
      2. it is already in requirements-model.txt, so this adds no dependency,
         which is the constraint the rest of this module holds to;
      3. the model's own get_video_reader_backend() prefers it when available, so
         this makes the default match what the model's author already treated as
         correct rather than substituting a different judgement.

    torchcodec is the forward-looking answer -- torchvision removed read_video
    precisely because decoding moved there. But wiring it in means a new
    dependency and a reader implementation that does not exist in this module.
    That is a feature, and it belongs in the model repo, not in a compatibility
    repair.
    """
    mod = _find_processing_module()
    if mod is None:
        raise RuntimeError(
            "processing_locateanything is not imported yet. Load the processor "
            "first:\n    AutoProcessor.from_pretrained(..., trust_remote_code=True)")

    try:
        from torchvision import io as _tvio
        if hasattr(_tvio, "read_video"):
            if verbose:
                print("locateanything_fix: torchvision.io.read_video present; "
                      "video backend left alone")
            return False
    except ImportError:
        pass

    decord_reader = mod.VIDEO_READER_BACKENDS.get("decord")
    if decord_reader is None:
        raise RuntimeError("no decord reader in VIDEO_READER_BACKENDS; pip install decord")
    try:
        import decord  # noqa: F401
    except ImportError as e:
        raise RuntimeError("decord is not installed; video cannot be decoded") from e

    if getattr(mod.VIDEO_READER_BACKENDS.get("torchvision"), "_locateanything_video", False):
        if verbose:
            print("locateanything_fix: video decode already redirected")
        return False

    def _redirected(ele):
        return decord_reader(ele)
    _redirected._locateanything_video = True
    mod._orig_read_video_torchvision = mod.VIDEO_READER_BACKENDS["torchvision"]
    mod.VIDEO_READER_BACKENDS["torchvision"] = _redirected
    if hasattr(mod, "get_video_reader_backend"):
        try:
            mod.get_video_reader_backend.cache_clear()
        except Exception:
            pass
    if verbose:
        print("locateanything_fix: video decode redirected to decord "
              "(torchvision.io.read_video was removed in torchvision 0.19)")
    return True


def disable_video_decode(verbose: bool = True) -> bool:
    """Put the original torchvision video reader back.

    Mirrors `disable_logits_slice`. `enable_video_decode` stashes the original at
    `mod._orig_read_video_torchvision`, and the same A/B-in-one-process hazard
    that `revert()` exists for applies here: this mutates a module-level dict.
    """
    mod = _find_processing_module()
    if mod is None:
        return False
    original = getattr(mod, "_orig_read_video_torchvision", None)
    current = mod.VIDEO_READER_BACKENDS.get("torchvision")
    if original is None or not getattr(current, "_locateanything_video", False):
        return False
    mod.VIDEO_READER_BACKENDS["torchvision"] = original
    if hasattr(mod, "get_video_reader_backend"):
        try:
            mod.get_video_reader_backend.cache_clear()
        except Exception:
            pass
    if verbose:
        print("locateanything_fix: video decode restored to torchvision")
    return True


def is_applied() -> bool:
    """Marker-identity check, with one sharp edge.

    Wrapping VL_VISION_ATTENTION_FUNCTIONS["sdpa"] for any reason -- profiler,
    logger, call counter -- yields a new object without the marker, so this
    reports False and enable_packed_vision() refuses with "call apply() first"
    while the fix is in fact still executing underneath the wrapper. Copy the
    marker onto the wrapper (scripts/segcheck.py shows the pattern) if you need
    to instrument it.
    """
    mod = _find_vit_module()
    if mod is None:
        return False
    return getattr(mod.VL_VISION_ATTENTION_FUNCTIONS.get("sdpa"), _MARK, False)


def apply(verbose: bool = True, model=None, cache_vision: bool = False,
          cache_size: int = 4) -> bool:
    """Swap in the 4-D SDPA implementation. Returns True if it was applied.

    `cache_vision=True` additionally enables the vision-feature cache, which
    needs the model instance. The two are independent: the SDPA fix helps every
    workload, the cache only helps repeat queries about one image.

    The other knobs -- enable_logits_slice, enable_packed_vision,
    enable_video_decode -- are deliberately NOT kwargs here. That is not an
    oversight to tidy up later.

    Their preconditions differ. apply() needs the model's modeling_vit imported
    and raises if it is not. enable_video_decode() needs the processor's
    processing_locateanything, loaded by AutoProcessor rather than AutoModel: a
    different object from a different call. Bundling them means apply() either
    raises for a reason unrelated to attention, or swallows that failure quietly
    -- and silent no-ops are the exact hazard the rest of this module refuses.

    They are also different subsystems: apply() swaps the vision attention
    implementation, enable_video_decode() repairs a broken video decoder. If
    anything here is inconsistent it is cache_vision= itself, one of five knobs
    reachable through apply() against four standalone. The fix for that would be
    retiring these kwargs in favour of enable_vision_cache(), not adding three
    more.

    Raises RuntimeError if the model's custom code has not been imported yet,
    because silently doing nothing is exactly the failure mode that makes a
    benchmark look like a clean negative result.
    """
    mod = _find_vit_module()
    if mod is None:
        raise RuntimeError(
            "modeling_vit is not imported yet. Load the model first:\n"
            "    model = AutoModel.from_pretrained(..., trust_remote_code=True)\n"
            "    locateanything_fix.apply()"
        )
    if is_applied():
        if verbose:
            print("locateanything_fix: already applied")
        return False

    setattr(_sdpa_attention_4d, _MARK, True)
    mod.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_attention_4d

    if verbose:
        flash = getattr(mod, "flash_attn_varlen_func", None)
        print("locateanything_fix: applied to VL_VISION_ATTENTION_FUNCTIONS['sdpa']"
              + ("  (note: flash_attn is installed, so the ViT may use it instead "
                 "and this patch will be inert)" if flash is not None else ""))

    if cache_vision:
        if model is None:
            raise ValueError("cache_vision=True needs the model: "
                             "apply(model=model, cache_vision=True)")
        enable_vision_cache(model, maxsize=cache_size)
        if verbose:
            print(f"locateanything_fix: vision cache enabled (maxsize={cache_size}); "
                  "only helps repeat queries on the same image")
    return True


def revert(verbose: bool = True) -> bool:
    """Put the original sdpa_attention back.

    Needed by any A/B that runs both arms in one process: `apply()` mutates a
    module-level dict, so without this the 'stock' arm silently inherits the fix
    from a previous iteration and the two arms report identical numbers.
    """
    mod = _find_vit_module()
    if mod is None or not is_applied():
        return False
    original = getattr(mod, "sdpa_attention", None)
    if original is None:
        raise RuntimeError("cannot find the original sdpa_attention to restore")
    mod.VL_VISION_ATTENTION_FUNCTIONS["sdpa"] = original
    if verbose:
        print("locateanything_fix: reverted to the original sdpa_attention")
    return True


def verify(model=None) -> None:
    """Confirm the patch is live on the path the model will actually take.

    Worth calling in any benchmark. The equivalent check caught a measurement
    where the patch had been written to the model directory but never executed,
    because transformers imports custom code from its own cache.
    """
    mod = _find_vit_module()
    if mod is None:
        raise RuntimeError("modeling_vit not imported")
    fn = mod.VL_VISION_ATTENTION_FUNCTIONS.get("sdpa")
    if not getattr(fn, _MARK, False):
        raise RuntimeError("locateanything_fix is NOT live on the sdpa path")
    impl = None
    if model is not None:
        vt = getattr(model, "vision_tower", None) or getattr(model, "vision_model", None)
        impl = getattr(vt, "attn_implementation", None) or getattr(
            getattr(vt, "config", None), "_attn_implementation", None)
    print(f"locateanything_fix: live"
          + (f"; vision attn implementation = {impl}" if impl else ""))
