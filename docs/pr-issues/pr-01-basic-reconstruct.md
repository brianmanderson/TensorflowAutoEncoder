# Issue draft for PR 1: basic `reconstruct_patches` op + layers

**File this issue at:** https://github.com/keras-team/keras/issues/new
**Title:** `Add reconstruct_patches op + ReconstructPatches{2,3}D layers (inverse of extract_patches)`

---

## Summary

Add the missing inverse of `keras.ops.image.extract_patches` to core Keras, in both op form
(`keras.ops.image.reconstruct_patches`, `keras.ops.image.reconstruct_patches_3d`) and Layer form
(`keras.layers.ReconstructPatches2D`, `keras.layers.ReconstructPatches3D`).

This PR is intentionally scoped to the basic non-overlapping case (`strides == size`, `channels_last`,
`padding="valid"`/`"same"`). Overlap, `channels_first`, `dilation_rate`, `reduction` modes, and other
features are planned as incremental follow-up PRs.

## Motivation

`extract_patches` (and `extract_patches_3d`) ship in Keras as forward-only ops. PyTorch has
`torch.nn.Unfold` and `torch.nn.Fold` as symmetric pair — Keras has the unfold half but no fold half.
This asymmetry comes up repeatedly in:

- **Patch-based autoencoders** that compress in patch space and need to reconstruct the original.
- **ViT decoders** that produce patch tokens which must be folded back to a pixel grid.
- **Sliding-window inference** where per-patch predictions need to be stitched into a full-resolution output.
- **3D medical-imaging** segmentation pipelines that operate on patches due to GPU-memory constraints
  and reassemble the full volume for downstream processing.

Reference: this addresses the gap that motivated #20046 (fold/unfold), which was closed without an
inverse op landing.

## Proposed API

```python
keras.ops.image.reconstruct_patches(
    patches,            # shape from extract_patches: (B, gH, gW, pH*pW*C) for 2D
    size,               # (pH, pW) — matches extract_patches' `size`
    output_size,        # (H, W) — original spatial shape before extraction
    strides=None,       # defaults to `size` (non-overlapping)
    padding="valid",    # "valid" | "same", matches extract_patches semantics
    data_format=None,
)

keras.ops.image.reconstruct_patches_3d(patches, size, output_size, ...)

keras.layers.ReconstructPatches2D(size, output_size, padding=..., data_format=..., name=...)
keras.layers.ReconstructPatches3D(size, output_size, padding=..., data_format=..., name=...)
```

Naming and argument names match the forward op exactly so the two compose cleanly. `output_size` is
required (not inferred) because under `padding="same"` the forward op is lossy in spatial dimension
(input `D=25` and `D=32` with patch 16 both produce `gD=2`), so the original `D` must be provided
explicitly. (A future PR can add `output_size=None` auto-infer for `padding="valid"`.)

## Implementation approach

Pure reshape / transpose / slice — no new ops, no math beyond layout. Specifically:

1. Reshape patches `(B, gH, gW, pH*pW*C)` → `(B, gH, gW, pH, pW, C)`
2. Transpose to `(B, gH, pH, gW, pW, C)`
3. Reshape to `(B, gH*pH, gW*pW, C)` — the padded canvas the forward conv would have seen
4. For `padding="same"`: slice off the padding using the same `pad_before = total // 2` convention
   that `backend.nn.conv("same")` uses on the forward pass (verified consistent across TF/JAX/torch).
5. For `padding="valid"`: no slicing.

This is the inverse of the conv-with-identity-kernel that `_extract_patches_*` does internally, but
implemented without `conv_transpose` for the non-overlap case — much faster (no math, just data
movement). Overlapping reconstruction (deferred to a follow-up PR) needs `conv_transpose` and is a
fundamentally different code path.

## Working branch

**Branch:** https://github.com/brianmanderson/keras/tree/pr/01-basic-reconstruct
**Diff:** 11 files changed, +851 lines (no deletions)

- `keras/src/ops/image.py` — adds `ReconstructPatches` Operation, `reconstruct_patches`,
  `reconstruct_patches_3d`, internal helpers
- `keras/src/layers/reshaping/reconstruct_patches{2,3}d.py` — Layer wrappers (new files)
- `keras/src/layers/reshaping/reconstruct_patches{2,3}d_test.py` — Layer tests (new files)
- `keras/src/ops/image_test.py` — symbolic-shape tests added next to the existing
  `test_extract_patches{,_3d}`
- `keras/src/layers/__init__.py` and the two `keras/api/.../layers/__init__.py` files — registrations
- `keras/api/{,_tf_keras/}ops/image/__init__.py` — op registrations

## Test status

All tests pass on TensorFlow, JAX, and PyTorch backends locally:

- 11 layer tests on `ReconstructPatches2D` covering valid + same padding, asymmetric patches,
  non-divisible inputs, get_config round-trip, error paths
- 11 layer tests on `ReconstructPatches3D` (same shape for 3D)
- 4 symbolic-shape tests on the ops in `image_test.py`

The killer test is `test_extract_then_reconstruct_roundtrip` — for several shape combinations
(including non-divisible ones forcing the `padding="same"` crop path), it asserts
`reconstruct(extract(x)) == x` exactly. This is the strongest argument the layer should exist; the
crop math is easy to get wrong by hand under `padding="same"` and our implementation matches the
forward op's padding distribution exactly across all three backends.

## Likely objection and pre-emption

The most common review pushback for "missing inverse" PRs is: *"users can compose this from
`keras.ops.reshape` + `transpose` + `slice` themselves."* Three counters:

1. **`padding="same"` is non-trivial.** The cropping must exactly match how `backend.nn.conv("same")`
   distributed padding on the forward pass. Getting this wrong silently shifts pixels by 1 — the
   roundtrip test catches it; manual code rarely does.
2. **Asymmetric APIs are a usability gap.** Shipping the forward op without an inverse forces every
   user to re-derive the same handful of lines, often incorrectly.
3. **PyTorch precedent.** `torch.nn.Fold` and `torch.nn.Unfold` are a symmetric pair, both first-class.
   This proposal brings Keras to parity.

## Follow-up PRs planned

If this lands, the planned incremental follow-ups (in dependency order) are:

1. Overlap support + `reduction="mean"`/`"sum"` (PyTorch `Fold` parity for overlapping patches)
2. `channels_first` data format
3. `dilation_rate` parameter
4. Auto-infer `output_size` for `padding="valid"` + improved error messages
5. `ExtractPatches2D`/`ExtractPatches3D` Layer wrappers
6. Dual-input dynamic `output_size` mode (for variable-input models)

Each of these has working code in a separate branch already. Mentioning here so reviewers can see
the trajectory without conflating scope into this PR.

---

## Next steps for the maintainer

Wait for a maintainer to ack ("yes, send a PR") before opening the actual PR. If the response is
"close as not planned" — file the same content as a focused PR to `keras-cv` or publish as a standalone
package; the working code remains useful regardless.
