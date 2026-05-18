"""Edge cases and real-world scenarios for patch extract / reconstruct.

Covers:
- Degenerate inputs: input == patch (single patch), 1-wide axes
- Input smaller than patch along an axis (only valid for `padding="same"`)
- Real-world architectures: ViT-style image patching, medical-imaging volumes
- Larger inputs that stress memory / allocation paths
"""

import numpy as np
import pytest
from keras import ops

from reconstruct_patches import reconstruct_patches, reconstruct_patches_3d


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
# Degenerate: input dimension equals patch dimension (single patch per axis)
# ---------------------------------------------------------------------------


def test_2d_input_equals_patch():
    """When input exactly equals patch, extraction produces a single patch."""
    x = gradient_image(4, 4, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    assert patches.shape == (2, 1, 1, 48)
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(4, 4), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_input_equals_patch():
    x = gradient_volume(4, 4, 4, C=2, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid",
    )
    assert patches.shape == (2, 1, 1, 1, 128)
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(4, 4, 4), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# 1x1 / 1x1x1 patches (every pixel becomes a patch)
# ---------------------------------------------------------------------------


def test_2d_patch_size_one():
    x = gradient_image(8, 8, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(1, 1), padding="valid")
    assert patches.shape == (2, 8, 8, 3)
    recon = reconstruct_patches(
        patches, size=(1, 1), output_size=(8, 8), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_patch_size_one():
    x = gradient_volume(4, 8, 8, C=2, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(1, 1, 1), padding="valid",
    )
    assert patches.shape == (1, 4, 8, 8, 2)
    recon = reconstruct_patches_3d(
        patches, size=(1, 1, 1), output_size=(4, 8, 8), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Input smaller than patch on one axis (only "same" can handle this)
# ---------------------------------------------------------------------------


def test_2d_input_smaller_than_patch_axis():
    """H=2 with patch_h=4: must pad to 4, becomes 1 row of patches."""
    x = gradient_image(2, 32, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="same")
    # gH = ceil(2/4) = 1, gW = ceil(32/4) = 8
    assert patches.shape == (2, 1, 8, 48)
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(2, 32), padding="same",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_input_smaller_than_patch_axis():
    """D=1 with patch_d=4: must pad. Single-slice medical-imaging case."""
    x = gradient_volume(1, 32, 32, C=2, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 8, 8), padding="same",
    )
    # gD=1, gH=4, gW=4, flat = 4*8*8*2 = 512
    assert patches.shape == (1, 1, 4, 4, 512)
    recon = reconstruct_patches_3d(
        patches, size=(4, 8, 8), output_size=(1, 32, 32), padding="same",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Real-world: ViT-style image patching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_in,patch", [
        (224, 16),    # ViT-Base/16
        (224, 14),    # ViT-Huge/14
        (256, 32),    # ViT-Large/32
        (384, 16),
    ],
)
def test_vit_style_patching(size_in, patch):
    x = gradient_image(size_in, size_in, C=3, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(patch, patch), padding="valid",
    )
    recon = reconstruct_patches(
        patches, size=(patch, patch),
        output_size=(size_in, size_in), padding="valid",
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, rtol=1e-6, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Real-world: medical imaging volumes (3D, asymmetric, non-divisible)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,H,W,pD,pH,pW", [
        (64, 256, 256, 16, 32, 32),    # typical CT/MR chunk
        (48, 192, 192, 16, 32, 32),
        (32, 128, 128, 8, 16, 16),
        # Non-divisible shapes (the original motivating use case)
        (25, 59, 55, 16, 32, 32),
        (40, 200, 180, 16, 32, 32),
    ],
)
def test_medical_imaging_volumes(D, H, W, pD, pH, pW):
    x = gradient_volume(D, H, W, C=1, batch=1)
    x_t = ops.convert_to_tensor(x)
    padding = "valid" if (D % pD == 0 and H % pH == 0 and W % pW == 0) else "same"
    patches = ops.image.extract_patches(
        x_t, size=(pD, pH, pW), padding=padding,
    )
    recon = reconstruct_patches_3d(
        patches, size=(pD, pH, pW),
        output_size=(D, H, W), padding=padding,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, rtol=1e-6, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Real-world: small image classification (MNIST/CIFAR-ish)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size_in,patch,C", [
        (28, 4, 1),       # MNIST-ish
        (32, 4, 3),       # CIFAR-ish
        (32, 8, 3),
        (64, 8, 3),
    ],
)
def test_small_image_classification(size_in, patch, C):
    x = gradient_image(size_in, size_in, C=C, batch=4)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(patch, patch), padding="valid",
    )
    recon = reconstruct_patches(
        patches, size=(patch, patch),
        output_size=(size_in, size_in), padding="valid",
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Asymmetric rectangular inputs (panoramic / non-square)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "H,W,pH,pW,padding", [
        (64, 128, 8, 16, "valid"),
        (128, 64, 16, 8, "valid"),
        (37, 71, 8, 16, "same"),   # rectangular + non-divisible
        (100, 50, 7, 5, "same"),
    ],
)
def test_rectangular_2d(H, W, pH, pW, padding):
    x = gradient_image(H, W, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(pH, pW), padding=padding,
    )
    recon = reconstruct_patches(
        patches, size=(pH, pW), output_size=(H, W), padding=padding,
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)
