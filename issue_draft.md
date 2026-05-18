# Proposal: add `reconstruct_patches` op + `ReconstructPatches{2,3}D` layers

## Summary

Add the missing inverse of `keras.ops.image.extract_patches` to core Keras, in
both op form (`keras.ops.image.reconstruct_patches`,
`keras.ops.image.reconstruct_patches_3d`) and Layer form
(`keras.layers.ReconstructPatches2D`, `keras.layers.ReconstructPatches3D`),
for the non-overlapping case (`strides == size`).

This addresses the unresolved request in #20046 (fold/unfold). PyTorch ships
`torch.nn.Fold` / `torch.nn.Unfold` as first-class modules. Keras currently
has `extract_patches` (the equivalent of `Unfold`) but no inverse — the
"reconstruct from patches" half is missing entirely.

## Motivation

`extract_patches` (and `extract_patches_3d`) shipped recently and is widely
useful — ViT patch tokenization, patch-based autoencoders, sliding-window
inference for medical imaging, etc. The inverse operation comes up in the
same scenarios:

- **Autoencoders** that compress in patch space and need to reconstruct the
  original image/volume.
- **ViT decoders** that produce patch tokens which must be folded back to a
  pixel grid.
- **Sliding-window inference** where per-patch predictions need to be
  stitched into a full-resolution output.
- **3D medical imaging** segmentation pipelines that operate on patches due
  to GPU memory constraints and reassemble the full volume for downstream
  processing.

Today, users have to hand-roll the inverse with `keras.ops.reshape +
transpose + pad + slice`. Getting it right under `padding="same"` with
non-divisible spatial dimensions is error-prone — and there's no canonical
implementation to point people at.

## Proposed API

```python
keras.ops.image.reconstruct_patches(
    patches,                # (gH, gW, pH*pW*C) or (N, gH, gW, pH*pW*C) for 2D
                            # (gD, gH, gW, pD*pH*pW*C) or batched for 3D
    size,                   # (pH, pW) or (pD, pH, pW), matches extract_patches
    output_size,            # (H, W) or (D, H, W), the original spatial shape
    strides=None,           # must equal size (non-overlapping); default = size
    padding="valid",        # "same" | "valid", semantics mirror extract_patches
    data_format=None,
)

# Dedicated 3D variant (mirroring extract_patches_3d):
keras.ops.image.reconstruct_patches_3d(patches, size, output_size, ...)

# Layer wrappers (mirroring the Conv2DTranspose convention of taking the
# output shape at __init__, not as a second input tensor):
keras.layers.ReconstructPatches2D(size, output_size, padding=..., data_format=...)
keras.layers.ReconstructPatches3D(size, output_size, padding=..., data_format=...)
```

Naming and argument names mirror the forward op exactly so the two compose
cleanly. `output_size` is required (not inferred) because under
`padding="same"`, the forward op is lossy in the spatial dimension — `D=25,
size=16` and `D=32, size=16` both produce `gD=2`, so the original `D` must be
provided explicitly.

## Initial scope

To keep the first PR reviewable:

- **Non-overlapping only** (`strides == size`, `dilation_rate == 1`).
  Overlap-add reconstruction (true `Fold` equivalent for arbitrary strides)
  is a natural follow-up using transposed-conv with identity kernel, but is
  intentionally out of scope for v1.
- **`channels_last` only**. `channels_first` raises `NotImplementedError`
  with a clear message; would be added before merge if reviewers request it.

The forward `extract_patches` op also shipped initially with a focused
scope, so this matches precedent.

## Working prototype + evidence

Branch with the proposed implementation:
**https://github.com/brianmanderson/keras/tree/feat/reconstruct-patches**

Files changed: 7 modified, 4 added; +454 / -0.
- `keras/src/ops/image.py` — adds `ReconstructPatches` Operation,
  `reconstruct_patches`, `reconstruct_patches_3d`, helpers.
- `keras/src/layers/reshaping/reconstruct_patches2d.py` (new) — Layer wrapper.
- `keras/src/layers/reshaping/reconstruct_patches3d.py` (new) — Layer wrapper.
- `keras/src/ops/image_test.py` — symbolic-shape tests.
- `keras/src/layers/reshaping/reconstruct_patches{2,3}d_test.py` (new) —
  roundtrip + serialization + error-handling tests.
- `__init__.py` registrations in `keras/src/layers/`, `keras/api/layers/`,
  `keras/api/_tf_keras/keras/layers/`, `keras/api/ops/image/`,
  `keras/api/_tf_keras/keras/ops/image/`.

**Tests pass on all three backends** (TF 2.21, JAX 0.10, PyTorch 2.12 — 25
new tests × 3 backends = 75 green). The headline test is a parameterized
roundtrip:

```python
patches = ops.image.extract_patches(x, size=size, padding=padding)
recon = layers.ReconstructPatches3D(
    size=size, output_size=(D, H, W), padding=padding,
)(patches)
assert_allclose(recon, x, atol=1e-6)
```

across 6 shape combinations including the deliberately non-divisible
`(D=25, H=59, W=55)` case that exercises asymmetric `"same"` padding.

## Discussion points / known objections

The most likely reviewer pushback is **"users can compose this from
`keras.ops.reshape` + `transpose` + `pad` + `slice` themselves"**. Three
counterpoints:

1. The composition is non-trivial under `padding="same"` with non-divisible
   input dimensions — the cropping must exactly match how
   `backend.nn.conv("same")` distributed the padding on the forward pass
   (`pad_before = total // 2`, verified to be identical across TF/JAX/PyTorch
   backends). Easy to get wrong; the roundtrip then fails silently with a
   shift of one or more pixels.
2. The forward op is a first-class API. Asymmetric APIs (ship the forward
   but not the inverse) are a usability gap.
3. PyTorch's precedent: `torch.nn.Fold` and `torch.nn.Unfold` are
   symmetric, first-class modules. This proposal brings parity.

If maintainers prefer to land just the op without the Layer wrapper, that's
also acceptable — the op is the load-bearing piece and the Layer is sugar.

Happy to split into two PRs (ops first, layers as follow-up) if that's the
preferred review cadence.
