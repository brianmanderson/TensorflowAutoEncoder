"""Roundtrip tests for overlapping reconstruction (strides < size).

When patches overlap, each output pixel receives contributions from multiple
patches. `reconstruct_patches` sums these contributions then divides by the
overlap count to recover the average. When patches came from a consistent
input, the average is the original — proving `reconstruct(extract(x)) == x`
holds even with overlap.

Implementation uses transposed conv with an identity kernel, which is the
canonical analogue of PyTorch's `Fold` for overlapping patches.
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


def gradient_image(H, W, C=1, batch=1, dtype="float32"):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype(dtype)
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


def gradient_volume(D, H, W, C=1, batch=1, dtype="float32"):
    d = np.linspace(0, 1, D)
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Dg, Hg, Wg, Cg = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((Dg + 2 * Hg + 3 * Wg + 4 * Cg) / 10.0).astype(dtype)
    return np.broadcast_to(vol[None, ...], (batch, D, H, W, C)).copy()


# ---------------------------------------------------------------------------
# 2D overlap grid — varying stride ratios and padding modes
# ---------------------------------------------------------------------------

# Each case: (H, W, C, size, stride, padding)
OVERLAP_CASES_2D = [
    # 50% overlap
    (16, 16, 3, (4, 4), (2, 2), "valid"),
    (32, 32, 3, (8, 8), (4, 4), "valid"),
    (24, 24, 1, (6, 6), (3, 3), "valid"),
    # 75% overlap (stride = size/4)
    (16, 16, 3, (4, 4), (1, 1), "valid"),
    (32, 32, 3, (8, 8), (2, 2), "valid"),
    # Asymmetric stride
    (24, 24, 1, (4, 4), (1, 2), "valid"),
    (24, 24, 1, (4, 6), (2, 3), "valid"),
    # Same-padding overlap
    (17, 19, 3, (4, 4), (2, 2), "same"),
    (15, 17, 3, (5, 5), (3, 3), "same"),
    (33, 41, 2, (5, 7), (3, 5), "same"),
    # Asymmetric patch with overlap
    (24, 32, 3, (4, 8), (2, 4), "valid"),
]


@pytest.mark.parametrize(
    "H,W,C,size,stride,padding", OVERLAP_CASES_2D,
    ids=lambda v: f"{v}" if isinstance(v, tuple) else str(v),
)
def test_2d_overlap_roundtrip(H, W, C, size, stride, padding):
    x = gradient_image(H, W, C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, strides=stride, padding=padding,
    )
    recon = reconstruct_patches(
        patches, size=size, output_size=(H, W),
        strides=stride, padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape, (
        f"Shape mismatch: in {x.shape}, out {recon_np.shape} "
        f"(H={H}, W={W}, size={size}, stride={stride}, {padding})"
    )
    np.testing.assert_allclose(
        recon_np, x, rtol=1e-5, atol=1e-5,
        err_msg=(
            f"Overlap roundtrip failed for H={H}, W={W}, size={size}, "
            f"stride={stride}, {padding}"
        ),
    )


# ---------------------------------------------------------------------------
# 3D overlap grid
# ---------------------------------------------------------------------------

OVERLAP_CASES_3D = [
    # 50% overlap on every axis
    (8, 8, 8, 2, (4, 4, 4), (2, 2, 2), "valid"),
    (16, 16, 16, 1, (4, 4, 4), (2, 2, 2), "valid"),
    (8, 16, 16, 2, (4, 8, 8), (2, 4, 4), "valid"),
    # Single-axis overlap, others non-overlap
    (8, 16, 16, 2, (4, 4, 4), (4, 2, 4), "valid"),
    # Stride 1 (maximum overlap)
    (8, 8, 8, 2, (4, 4, 4), (1, 1, 1), "valid"),
    # Same-padding overlap
    (9, 17, 19, 2, (4, 4, 4), (2, 2, 2), "same"),
    (1, 32, 32, 2, (4, 8, 8), (4, 4, 4), "same"),  # single-slice
]


@pytest.mark.parametrize(
    "D,H,W,C,size,stride,padding", OVERLAP_CASES_3D,
    ids=lambda v: f"{v}" if isinstance(v, tuple) else str(v),
)
def test_3d_overlap_roundtrip(D, H, W, C, size, stride, padding):
    x = gradient_volume(D, H, W, C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, strides=stride, padding=padding,
    )
    recon = reconstruct_patches_3d(
        patches, size=size, output_size=(D, H, W),
        strides=stride, padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape
    np.testing.assert_allclose(
        recon_np, x, rtol=1e-5, atol=1e-5,
        err_msg=(
            f"3D overlap roundtrip failed for D={D}, H={H}, W={W}, "
            f"size={size}, stride={stride}, {padding}"
        ),
    )


# ---------------------------------------------------------------------------
# Layer parity for overlap
# ---------------------------------------------------------------------------

def test_2d_overlap_layer_matches_op():
    x = gradient_image(16, 16, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    recon_op = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    recon_layer = layer(patches)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(recon_op),
        ops.convert_to_numpy(recon_layer),
    )


def test_3d_overlap_layer_matches_op():
    x = gradient_volume(8, 8, 8, C=2, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), strides=2, padding="valid")
    recon_op = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 8, 8),
        strides=(2, 2, 2), padding="valid",
    )
    layer = ReconstructPatches3D(
        size=(4, 4, 4), output_size=(8, 8, 8),
        strides=(2, 2, 2), padding="valid",
    )
    recon_layer = layer(patches)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(recon_op),
        ops.convert_to_numpy(recon_layer),
    )


# ---------------------------------------------------------------------------
# Channels_first + overlap
# ---------------------------------------------------------------------------

def test_2d_overlap_channels_first():
    H, W, C = 16, 16, 3
    x = np.random.RandomState(0).rand(2, C, H, W).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), strides=2, padding="valid",
        data_format="channels_first",
    )
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(H, W),
        strides=(2, 2), padding="valid", data_format="channels_first",
    )
    assert tuple(recon.shape) == (2, C, H, W)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Gapped strides (strides > size) — rejected as unrecoverable
# ---------------------------------------------------------------------------

def test_gapped_strides_2d_rejected():
    with pytest.raises(NotImplementedError, match="gapped"):
        ReconstructPatches2D(
            size=(4, 4), output_size=(32, 32),
            strides=(8, 8), padding="valid",
        )


def test_gapped_strides_3d_rejected():
    with pytest.raises(NotImplementedError, match="gapped"):
        ReconstructPatches3D(
            size=(4, 4, 4), output_size=(32, 32, 32),
            strides=(8, 8, 8), padding="valid",
        )


# ---------------------------------------------------------------------------
# Inconsistent valid output_size with given stride is caught at call time
# ---------------------------------------------------------------------------

def test_valid_inconsistent_output_size():
    """For padding='valid' overlap, output_size must satisfy
    (grid-1)*stride + patch <= output_size < (grid-1)*stride + patch + stride.
    """
    x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    # patches.shape = (1, 7, 7, 48), valid_output = (7-1)*2+4 = 16
    # 20 is in [16, 18) ? No, 20 >= 18, so rejected
    with pytest.raises(ValueError, match="inconsistent"):
        reconstruct_patches(
            patches, size=(4, 4), output_size=(20, 20),
            strides=(2, 2), padding="valid",
        )
