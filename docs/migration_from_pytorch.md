# Migrating from `torch.nn.Fold` / `torch.nn.Unfold`

This guide maps PyTorch's `Fold` and `Unfold` modules to the corresponding
Keras 3 ops and layers in this repo.

## TL;DR

| PyTorch | Keras |
|---|---|
| `torch.nn.Unfold(kernel_size, dilation, padding, stride)` | `ExtractPatches2D(size=kernel_size, ..., dilation_rate=..., padding=..., strides=...)` |
| `torch.nn.Fold(output_size, kernel_size, dilation, padding, stride)` | `ReconstructPatches2D(size=kernel_size, output_size=output_size, ..., dilation_rate=..., padding=..., strides=..., reduction="sum")` |

The single most important behavioral difference: **PyTorch's `Fold` sums
overlapping contributions; our default is `reduction="mean"`**. Pass
`reduction="sum"` to match PyTorch exactly. Use `reduction="mean"` (default)
when you want `reconstruct(extract(x)) == x` to be an exact identity.

## Output format

PyTorch's `Unfold` produces a 3D tensor:

```
torch:  (B, C * prod(kernel_size), L)   where L is the number of patches
keras:  (B, gH, gW, prod(kernel_size) * C)   for channels_last
```

You may need to reshape/transpose to convert. A `keras.ops.transpose +
ops.reshape` pair handles the conversion in 4 lines.

## Side-by-side examples

### Patch extraction (Unfold → ExtractPatches)

PyTorch:
```python
import torch
import torch.nn as nn

x = torch.randn(2, 3, 16, 16)            # (B, C, H, W)
unfold = nn.Unfold(kernel_size=4, stride=4)
patches = unfold(x)                       # (B, C*16, 16) = (2, 48, 16)
```

Keras (this repo):
```python
import numpy as np
from reconstruct_patches import ExtractPatches2D

x = np.random.rand(2, 16, 16, 3).astype("float32")   # (B, H, W, C)
patches = ExtractPatches2D(size=(4, 4), strides=(4, 4), padding="valid")(x)
# patches.shape == (2, 4, 4, 48)
```

Differences:
- Keras keeps the patch-grid as separate axes; PyTorch flattens them into `L`.
- Keras layout is `channels_last` by default (`B, H, W, C`); PyTorch is
  `channels_first` (`B, C, H, W`). Pass `data_format="channels_first"` to
  match PyTorch (but see the TF-CPU caveat in the README).

### Patch reconstruction (Fold → ReconstructPatches)

PyTorch:
```python
fold = nn.Fold(output_size=(16, 16), kernel_size=4, stride=4)
recon = fold(patches)                     # (2, 3, 16, 16); summed
```

Keras with PyTorch-compatible summing:
```python
from reconstruct_patches import ReconstructPatches2D

recon = ReconstructPatches2D(
    size=(4, 4), output_size=(16, 16),
    strides=(4, 4), padding="valid",
    reduction="sum",   # <- match PyTorch Fold semantics
)(patches)
```

Keras with averaging (recovers original exactly when patches came from one image):
```python
recon = ReconstructPatches2D(
    size=(4, 4), output_size=(16, 16),
    strides=(4, 4), padding="valid",
    # reduction="mean" is the default
)(patches)
# recon ≈ original input that was unfolded
```

### Sliding-window inference (Unfold with stride 1 → Fold with stride 1)

A classic pattern: per-patch inference followed by averaged stitching.
This is exactly what `reduction="mean"` is for.

PyTorch:
```python
unfold = nn.Unfold(kernel_size=patch_size, stride=1)
patches = unfold(image)              # (B, C*kh*kw, L)
# ... per-patch processing produces same-shape `predictions`
fold = nn.Fold(output_size=image.shape[-2:], kernel_size=patch_size, stride=1)
predictions_sum = fold(predictions)
counts = fold(torch.ones_like(predictions))
predictions_avg = predictions_sum / counts.clamp(min=1)
```

Keras (one line for the average — no manual count division):
```python
import keras
from reconstruct_patches import ExtractPatches2D, ReconstructPatches2D

patches = ExtractPatches2D(size=patch_size, strides=1, padding="valid")(image)
# ... per-patch processing produces same-shape `predictions`
predictions_avg = ReconstructPatches2D(
    size=patch_size, output_size=image.shape[1:3],
    strides=(1, 1), padding="valid",
    # reduction="mean" is default — handles the averaging for you
)(predictions)
```

### Dilated patches (Atrous Unfold/Fold)

PyTorch:
```python
unfold = nn.Unfold(kernel_size=3, dilation=2, padding=0, stride=1)
patches = unfold(image)
fold = nn.Fold(output_size=image.shape[-2:], kernel_size=3, dilation=2, stride=1)
recon = fold(patches)
```

Keras:
```python
patches = ExtractPatches2D(size=(3, 3), strides=1, padding="valid")(image)
# Note: ExtractPatches uses the underlying extract_patches op's dilation_rate;
# pass it via the op call if you need explicit control:
from keras import ops
patches = ops.image.extract_patches(image, size=(3, 3), strides=1,
                                     dilation_rate=2, padding="valid")
recon = ReconstructPatches2D(
    size=(3, 3), output_size=image.shape[1:3],
    strides=(1, 1), padding="valid", dilation_rate=2, reduction="sum",
)(patches)
```

## 3D / volumetric data

PyTorch's `Fold`/`Unfold` are 2D-only. This repo provides explicit 3D variants:
`ExtractPatches3D` and `ReconstructPatches3D`. Same API, length-3 `size`,
operates on 5D `(B, D, H, W, C)` tensors (channels_last) or `(B, C, D, H, W)`
(channels_first).

```python
from reconstruct_patches import ExtractPatches3D, ReconstructPatches3D

volume = np.random.rand(1, 8, 16, 16, 2).astype("float32")
patches = ExtractPatches3D(size=(4, 4, 4), padding="valid")(volume)
recon = ReconstructPatches3D(
    size=(4, 4, 4), output_size=(8, 16, 16), padding="valid",
)(patches)
np.testing.assert_allclose(np.array(recon), volume, atol=1e-6)
```

## Variable-input models (`Input(shape=[None, ...])`)

A unique feature of this repo's reconstruct layers, not present in PyTorch's
`Fold`: dual-input mode where output spatial dims are derived from a
reference tensor at call time, enabling a single compiled model to handle
arbitrary input sizes.

```python
inputs = keras.Input(shape=(None, None, None, 1))    # any 3D shape
x = keras.layers.Conv3D(2, 3, padding="same")(inputs)
patches = ExtractPatches3D(size=(4, 4, 4), padding="same")(x)
recon = ReconstructPatches3D(size=(4, 4, 4), padding="same")([patches, x])
#  ^^^^^^^^^^^^^^^^^^^^^^^ output_size = x.shape spatial dims at call time
model = keras.Model(inputs, recon)
```

## Quick cross-reference

| Need | PyTorch | Keras |
|---|---|---|
| Tile non-overlapping patches | `Unfold(k, stride=k)` | `ExtractPatches2D(size=k)` |
| Reconstruct (exact roundtrip) | not built-in — must divide by count | `ReconstructPatches2D(reduction="mean")` (default) |
| Reconstruct (raw sum) | `Fold(...)` | `ReconstructPatches2D(reduction="sum")` |
| Sliding-window predictions average | manual sum + count + divide | `ReconstructPatches2D` (default mean) |
| Atrous patches | `dilation=N` | `dilation_rate=N` |
| 3D volumes | (not built-in) | `ExtractPatches3D` / `ReconstructPatches3D` |
| Variable input size | (not supported) | dual-input `Layer([patches, reference])` |
