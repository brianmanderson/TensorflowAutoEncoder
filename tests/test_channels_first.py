"""Roundtrip tests for `data_format="channels_first"`.

Verifies that extracting and reconstructing with channels_first inputs
produces the original tensor in channels_first layout. The implementation
transposes to channels_last internally, runs the channels_last path, and
transposes back — these tests prove that round-trips end-to-end.
"""

import numpy as np
import pytest
from keras import ops

from reconstruct_patches import (
    ReconstructPatches2D,
    ReconstructPatches3D,
    reconstruct_patches,
    reconstruct_patches_3d,
)


def gradient_image_cf(H, W, C, batch=1, dtype="float32"):
    """Channels-first gradient: shape (batch, C, H, W)."""
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype(dtype)  # (H, W, C)
    img_cf = np.transpose(img, (2, 0, 1))                    # (C, H, W)
    return np.broadcast_to(img_cf[None, ...], (batch, C, H, W)).copy()


def gradient_volume_cf(D, H, W, C, batch=1, dtype="float32"):
    d = np.linspace(0, 1, D)
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Dg, Hg, Wg, Cg = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((Dg + 2 * Hg + 3 * Wg + 4 * Cg) / 10.0).astype(dtype)  # (D,H,W,C)
    vol_cf = np.transpose(vol, (3, 0, 1, 2))                       # (C,D,H,W)
    return np.broadcast_to(vol_cf[None, ...], (batch, C, D, H, W)).copy()


# ---------------------------------------------------------------------------
# 2D channels_first roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "H,W,C,size,padding", [
        (16, 16, 3, (4, 4), "valid"),
        (32, 32, 1, (8, 8), "valid"),
        (15, 17, 3, (5, 5), "same"),
        (33, 41, 2, (5, 7), "same"),
        (24, 24, 4, (3, 3), "valid"),
        (28, 28, 1, (4, 4), "valid"),  # MNIST-like
        (32, 32, 3, (4, 4), "valid"),  # CIFAR-like
    ],
)
def test_2d_channels_first_roundtrip(H, W, C, size, padding):
    x = gradient_image_cf(H, W, C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, padding=padding, data_format="channels_first",
    )
    recon = reconstruct_patches(
        patches, size=size, output_size=(H, W), padding=padding,
        data_format="channels_first",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape
    np.testing.assert_allclose(recon_np, x, atol=1e-6)


@pytest.mark.parametrize(
    "H,W,C,size,padding", [
        (16, 16, 3, (4, 4), "valid"),
        (15, 17, 3, (5, 5), "same"),
    ],
)
def test_2d_channels_first_via_layer(H, W, C, size, padding):
    x = gradient_image_cf(H, W, C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, padding=padding, data_format="channels_first",
    )
    layer = ReconstructPatches2D(
        size=size, output_size=(H, W), padding=padding,
        data_format="channels_first",
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# 3D channels_first roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "D,H,W,C,size,padding", [
        (8, 16, 16, 2, (4, 4, 4), "valid"),
        (16, 32, 32, 1, (4, 8, 8), "valid"),
        (8, 8, 8, 3, (2, 2, 2), "valid"),
        (25, 59, 55, 2, (16, 32, 32), "same"),   # user's motivating shape
        (17, 33, 41, 3, (4, 8, 8), "same"),
        (1, 32, 32, 2, (4, 8, 8), "same"),       # single-slice
    ],
)
def test_3d_channels_first_roundtrip(D, H, W, C, size, padding):
    x = gradient_volume_cf(D, H, W, C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, padding=padding, data_format="channels_first",
    )
    recon = reconstruct_patches_3d(
        patches, size=size, output_size=(D, H, W), padding=padding,
        data_format="channels_first",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape
    np.testing.assert_allclose(recon_np, x, atol=1e-6)


@pytest.mark.parametrize(
    "D,H,W,C,size,padding", [
        (8, 16, 16, 2, (4, 4, 4), "valid"),
        (25, 59, 55, 2, (16, 32, 32), "same"),
    ],
)
def test_3d_channels_first_via_layer(D, H, W, C, size, padding):
    x = gradient_volume_cf(D, H, W, C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, padding=padding, data_format="channels_first",
    )
    layer = ReconstructPatches3D(
        size=size, output_size=(D, H, W), padding=padding,
        data_format="channels_first",
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Output shape verification (proves we're returning channels_first layout,
# not accidentally channels_last)
# ---------------------------------------------------------------------------

def test_2d_channels_first_output_layout():
    """C should be at axis 1 (not -1) for channels_first."""
    x = np.random.RandomState(0).rand(2, 5, 16, 16).astype("float32")  # C=5
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), padding="valid", data_format="channels_first",
    )
    # extract_patches channels_first 2D: (B, pH*pW*C, gH, gW) = (2, 80, 4, 4)
    assert tuple(patches.shape) == (2, 80, 4, 4)
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        padding="valid", data_format="channels_first",
    )
    assert tuple(recon.shape) == (2, 5, 16, 16)


def test_3d_channels_first_output_layout():
    x = np.random.RandomState(1).rand(1, 4, 8, 16, 16).astype("float32")  # C=4
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid", data_format="channels_first",
    )
    # 3D channels_first: (B, pD*pH*pW*C, gD, gH, gW) = (1, 256, 2, 4, 4)
    assert tuple(patches.shape) == (1, 256, 2, 4, 4)
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 16, 16),
        padding="valid", data_format="channels_first",
    )
    assert tuple(recon.shape) == (1, 4, 8, 16, 16)


# ---------------------------------------------------------------------------
# channels_first + unbatched
# ---------------------------------------------------------------------------

def test_2d_channels_first_unbatched():
    H, W, C = 16, 16, 3
    x_b = gradient_image_cf(H, W, C, batch=1)
    x = x_b[0]  # (C, H, W)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), padding="valid", data_format="channels_first",
    )
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(H, W),
        padding="valid", data_format="channels_first",
    )
    assert tuple(recon.shape) == (C, H, W)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_channels_first_unbatched():
    D, H, W, C = 8, 16, 16, 2
    x_b = gradient_volume_cf(D, H, W, C, batch=1)
    x = x_b[0]  # (C, D, H, W)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid", data_format="channels_first",
    )
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(D, H, W),
        padding="valid", data_format="channels_first",
    )
    assert tuple(recon.shape) == (C, D, H, W)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)
