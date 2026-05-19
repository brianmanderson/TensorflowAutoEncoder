"""Roundtrip tests for `reconstruct_patches` (+ Layer wrappers).

Modeled on WorkingOnPatchMaker.py's gradient-image + asymmetric-slice pattern.
The strongest evidence the layer should exist is: for any input shape,
`reconstruct(extract(x)) == x`. This file proves that for both padding modes,
both 2D and 3D, across a parametric grid of shapes.

Run under each backend:
    KERAS_BACKEND=tensorflow pytest test_reconstruct_patches.py -v
    KERAS_BACKEND=jax        pytest test_reconstruct_patches.py -v
    KERAS_BACKEND=torch      pytest test_reconstruct_patches.py -v
"""

import numpy as np
import pytest

import keras
from keras import ops

from reconstruct_patches import (
    reconstruct_patches,
    reconstruct_patches_3d,
    ReconstructPatches2D,
    ReconstructPatches3D,
)


# ---------------------------------------------------------------------------
# Helpers (parallel to WorkingOnPatchMaker.return_gradient_image)
# ---------------------------------------------------------------------------

def gradient_volume(depth, height, width, channels=1, batch=1):
    """Smoothly varying 5D tensor — any reshape error will be obvious."""
    d = np.linspace(0, 1, depth)
    h = np.linspace(0, 1, height)
    w = np.linspace(0, 1, width)
    c = np.linspace(0, 1, channels) if channels > 1 else np.array([0.5])
    D, H, W, C = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((D + 2 * H + 3 * W + 4 * C) / 10.0).astype("float32")
    return np.broadcast_to(vol[None, ...], (batch, depth, height, width, channels)).copy()


def gradient_image(height, width, channels=1, batch=1):
    h = np.linspace(0, 1, height)
    w = np.linspace(0, 1, width)
    c = np.linspace(0, 1, channels) if channels > 1 else np.array([0.5])
    H, W, C = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * H + 3 * W + 4 * C) / 9.0).astype("float32")
    return np.broadcast_to(img[None, ...], (batch, height, width, channels)).copy()


# ---------------------------------------------------------------------------
# 3D roundtrip
# ---------------------------------------------------------------------------

# Shapes chosen to cover: divisible (valid path) + non-divisible (same path)
# Non-divisible cases mirror WorkingOnPatchMaker's [:, :25, :59, :55].
RECONSTRUCT_3D_CASES = [
    # (D, H, W, C, patch_size, padding)
    # Note: (16, 32, 32) patches with C=2 build a 32768^2 (~4 GB) identity
    # kernel in keras.ops.image.extract_patches and OOM the 7 GB GitHub CI
    # runner. Using (8, 16, 16) here exercises the same non-divisible / valid
    # code paths with a 64x smaller kernel.
    (16, 32, 32, 1, (8, 16, 16), "valid"),    # divisible
    (16, 32, 32, 2, (8, 16, 16), "valid"),
    (25, 59, 55, 2, (8, 16, 16), "same"),     # user's asymmetric shape
    (24, 48, 48, 1, (8, 16, 16), "valid"),
    (17, 33, 41, 3, (4, 8, 8), "same"),       # all dims non-divisible
    (16, 16, 16, 1, (2, 4, 8), "valid"),      # asymmetric patch
    (5, 7, 11, 1, (3, 5, 7), "same"),         # prime-ish
]


@pytest.mark.parametrize("D,H,W,C,size,padding", RECONSTRUCT_3D_CASES)
def test_roundtrip_3d_extract_reconstruct(D, H, W, C, size, padding):
    x = gradient_volume(D, H, W, channels=C, batch=2)
    x_t = ops.convert_to_tensor(x)

    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    recon = reconstruct_patches_3d(
        patches, size=size, output_size=(D, H, W), padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)

    assert recon_np.shape == x.shape, (
        f"Shape mismatch: input {x.shape}, recon {recon_np.shape}"
    )
    np.testing.assert_allclose(recon_np, x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("D,H,W,C,size,padding", RECONSTRUCT_3D_CASES)
def test_roundtrip_3d_via_layer(D, H, W, C, size, padding):
    x = gradient_volume(D, H, W, channels=C, batch=2)
    x_t = ops.convert_to_tensor(x)

    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    layer = ReconstructPatches3D(
        size=size, output_size=(D, H, W), padding=padding,
    )
    recon = layer(patches)
    recon_np = ops.convert_to_numpy(recon)

    assert recon_np.shape == x.shape
    np.testing.assert_allclose(recon_np, x, rtol=1e-6, atol=1e-6)


def test_roundtrip_3d_unbatched():
    D, H, W, C = 16, 16, 16, 2
    size = (4, 4, 4)
    x = gradient_volume(D, H, W, C, batch=1)[0]  # unbatched
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=size, padding="valid")
    recon = reconstruct_patches_3d(
        patches, size=size, output_size=(D, H, W), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2D roundtrip
# ---------------------------------------------------------------------------

RECONSTRUCT_2D_CASES = [
    # (H, W, C, patch_size, padding)
    (64, 64, 1, (8, 8), "valid"),
    (64, 64, 3, (8, 8), "valid"),
    (59, 55, 3, (8, 8), "same"),
    (32, 48, 1, (4, 8), "valid"),  # asymmetric patch
    (33, 41, 2, (5, 7), "same"),
    (7, 11, 1, (3, 5), "same"),
]


@pytest.mark.parametrize("H,W,C,size,padding", RECONSTRUCT_2D_CASES)
def test_roundtrip_2d_extract_reconstruct(H, W, C, size, padding):
    x = gradient_image(H, W, channels=C, batch=2)
    x_t = ops.convert_to_tensor(x)

    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    recon = reconstruct_patches(
        patches, size=size, output_size=(H, W), padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)

    assert recon_np.shape == x.shape
    np.testing.assert_allclose(recon_np, x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("H,W,C,size,padding", RECONSTRUCT_2D_CASES)
def test_roundtrip_2d_via_layer(H, W, C, size, padding):
    x = gradient_image(H, W, channels=C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    layer = ReconstructPatches2D(
        size=size, output_size=(H, W), padding=padding,
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, rtol=1e-6, atol=1e-6)


def test_roundtrip_2d_unbatched():
    H, W, C = 16, 16, 3
    size = (4, 4)
    x = gradient_image(H, W, C, batch=1)[0]
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=size, padding="valid")
    recon = reconstruct_patches(
        patches, size=size, output_size=(H, W), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, rtol=1e-6)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_gapped_strides_rejected():
    """strides > size is rejected because gaps can't be filled."""
    x = gradient_volume(16, 16, 16, 1, batch=1)
    x_t = ops.convert_to_tensor(x)
    # Forward with stride > size — valid-padding produces some output
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), strides=(8, 8, 8), padding="valid",
    )
    with pytest.raises(NotImplementedError, match="gapped"):
        reconstruct_patches_3d(
            patches, size=(4, 4, 4), output_size=(16, 16, 16),
            strides=(8, 8, 8), padding="valid",
        )


def test_bad_size():
    x = np.zeros((1, 4, 4, 1), dtype="float32")
    with pytest.raises(ValueError, match="length 2 or 3"):
        reconstruct_patches(
            ops.convert_to_tensor(x), size=(2, 2, 2, 2),
            output_size=(4, 4), padding="valid",
        )


def test_bad_padding():
    x = gradient_image(8, 8, 1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    with pytest.raises(ValueError, match="'same' or 'valid'"):
        reconstruct_patches(
            patches, size=(4, 4), output_size=(8, 8), padding="reflect",
        )


# ---------------------------------------------------------------------------
# Layer config round-trip (serialization)
# ---------------------------------------------------------------------------

def test_layer_get_config_3d():
    layer = ReconstructPatches3D(
        size=(2, 3, 4), output_size=(10, 15, 20), padding="same",
    )
    config = layer.get_config()
    restored = ReconstructPatches3D.from_config(config)
    assert restored.size == (2, 3, 4)
    assert restored.output_size == (10, 15, 20)
    assert restored.padding == "same"


def test_layer_get_config_2d():
    layer = ReconstructPatches2D(
        size=(3, 4), output_size=(12, 16), padding="valid",
    )
    config = layer.get_config()
    restored = ReconstructPatches2D.from_config(config)
    assert restored.size == (3, 4)
    assert restored.output_size == (12, 16)
    assert restored.padding == "valid"
