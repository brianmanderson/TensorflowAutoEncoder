# Issue draft for PR 2: overlap reconstruction + `reduction="mean"`/`"sum"`

**File this issue at:** https://github.com/keras-team/keras/issues/new
**Title:** `Add overlap support to reconstruct_patches with reduction='mean'/'sum' (PyTorch Fold parity)`

---

## Summary

Builds on PR 1 (basic `reconstruct_patches` op + `ReconstructPatches{2,3}D` layers) by adding
support for **overlapping patches** (`strides < size` on any axis) and a new `reduction` parameter
that controls how overlapping contributions are combined.

This completes the symmetry with `torch.nn.Fold`: PyTorch's `Fold` *sums* overlapping contributions
by default; our `reduction="mean"` (default) *averages* them, which is the value that exactly
recovers the original input when patches were extracted consistently. Set `reduction="sum"` to
match PyTorch's semantics directly.

## Motivation

PR 1 was scoped to the non-overlapping case (`strides == size`). The much more common real-world
need is:

- **Sliding-window inference**: extract patches with `strides=1`, run per-patch predictions, stitch
  back into a full-resolution output. Currently requires manual `Fold(...) / Fold(ones_like(...))`
  in PyTorch and is even more painful in raw Keras. With this PR: one Layer call with
  `reduction="mean"`.
- **Patch-based super-resolution and denoising** that rely on overlapping patches for smoother
  reconstruction at seams.
- **PyTorch → Keras migrations**: users with existing `torch.nn.Fold` code can pass
  `reduction="sum"` and get the same numerical output.

## Working branch

**Branch:** https://github.com/brianmanderson/keras/tree/pr/02-overlap-reduction
**Diff against PR 1:** 4 files changed, +356 / -54

Stacked on top of PR 1. Once PR 1 merges, this branch rebases trivially onto the new master.

## API change (additive)

```python
keras.ops.image.reconstruct_patches(
    patches, size, output_size,
    strides=None,
    padding="valid",
    data_format=None,
    reduction="mean",          # NEW: "mean" | "sum"
)

keras.ops.image.reconstruct_patches_3d(..., reduction="mean")

keras.layers.ReconstructPatches2D(..., reduction="mean")
keras.layers.ReconstructPatches3D(..., reduction="mean")
```

- **`strides`** now accepts any value `1 <= s <= size` per axis. Gapped strides (`s > size`) are
  rejected with `NotImplementedError("does not support gapped patches")` — gaps cannot be filled.
- **`reduction="mean"`** (default): sum overlapping contributions, divide by per-pixel count.
  Recovers original input exactly when patches were extracted consistently.
- **`reduction="sum"`**: raw sum without averaging. Matches `torch.nn.Fold` semantics.
- Ignored when patches do not overlap (count is 1 everywhere).

## Implementation approach

The non-overlap path (PR 1) is kept as a fast specialization: pure reshape/transpose/slice, no
math. Overlap uses a different code path — `backend.nn.conv_transpose` with an identity kernel:

```python
# In _reconstruct_2d_overlap_cl (and 3D variant):
kernel = backend.numpy.eye(out_dim).reshape(pH, pW, C, out_dim)

output_sum = backend.nn.conv_transpose(
    inputs=patches, kernel=kernel, strides=strides,
    padding=padding, output_padding=op, data_format="channels_last",
)

if reduction == "mean":
    counts = backend.nn.conv_transpose(
        inputs=backend.numpy.ones_like(patches), kernel=kernel, ...,
    )
    return output_sum / backend.numpy.maximum(counts, 1)
return output_sum   # reduction == "sum"
```

For `padding="same"`, `output_padding=None` (let backend infer) and crop the result with the same
`pad_before = total // 2` convention as the non-overlap path. (Explicit `output_padding=(0,0)`
triggers a TF translation bug; `None` works cleanly across all backends.)

Dispatcher in `_reconstruct_patches_2d` / `_reconstruct_patches_3d`:

```python
if _is_nonoverlapping(strides, size):
    x = _reconstruct_2d_nonoverlap_cl(...)
else:
    x = _reconstruct_2d_overlap_cl(...)
```

## Tests

The existing PR 1 tests continue to pass. New tests in this PR:

- `test_gapped_strides_rejected` — `NotImplementedError("gapped")` for `strides > size`.
- `test_overlapping_strides_supported` — gradient-volume roundtrip with `strides < size`.
- `test_reduction_sum_overshoots_at_overlap` — with all-ones input, `reduction="sum"` exceeds 1
  at overlap positions (proves it's actually summing, not averaging).

All 25 tests pass on TensorFlow, JAX, and PyTorch backends.

## Likely objections + pre-emption

| Objection | Counter |
|---|---|
| "Why both `mean` and `sum`?" | PyTorch users expect `sum` (matches `Fold`); inverse-correctness users expect `mean`. Forcing one means everyone else needs a manual count-division wrapper. Both are one parameter away — keep both. |
| "Identity-kernel conv_transpose is expensive." | The non-overlap path is unchanged and still uses pure reshape (~30–100× faster). Users who don't need overlap don't pay overlap cost. Documented in the docstring. |
| "Gapped strides should also be supported with zeros." | No — silently filling gaps with zeros gives wrong values that look correct, which is a worse failure mode than an explicit error. If a user has a real need, they can pre-pad and use `valid`. |

## Follow-up PRs planned

After this lands, the remaining PRs from the trajectory (each in its own branch):

3. `channels_first` data format
4. `dilation_rate` parameter
5. Auto-infer `output_size` for `padding="valid"` + improved error messages
6. `ExtractPatches2D`/`ExtractPatches3D` Layer wrappers
7. Dual-input dynamic `output_size` mode (for variable-input models)
