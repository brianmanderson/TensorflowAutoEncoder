"""Tests for `reduction='sum'` vs `reduction='mean'` in the overlap path.

Default `mean` averages contributions and recovers the original; `sum` matches
PyTorch's `torch.nn.Fold` semantics (raw sum without averaging).
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import (
    ReconstructPatches2D,
    ReconstructPatches3D,
    reconstruct_patches,
    reconstruct_patches_3d,
)


def test_2d_reduction_mean_recovers_original():
    """Mean (default) recovers x exactly from extract(x)."""
    x = np.random.RandomState(0).rand(2, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="mean",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


def test_2d_reduction_sum_does_not_recover_original():
    """Sum should produce something larger than x at overlap regions."""
    x = np.ones((1, 16, 16, 3), dtype="float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    recon_sum = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    )
    recon_sum_np = ops.convert_to_numpy(recon_sum)
    # Center pixels are covered by multiple patches with stride < size.
    # For stride=2, size=4: each interior pixel is in 4 patches.
    # So center value should be ~4 (sum of 4 contributions of 1.0).
    assert recon_sum_np.max() > 1.5, (
        f"Sum reduction should overshoot 1.0 at overlap regions; "
        f"got max={recon_sum_np.max()}"
    )


def test_2d_reduction_sum_eq_mean_times_count():
    """sum / count should equal mean (definition check)."""
    x = np.random.RandomState(1).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")

    recon_mean = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="mean",
    ))
    recon_sum = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    ))
    # Build a count tensor by summing ones-patches
    ones_patches = ops.ones_like(patches)
    counts = ops.convert_to_numpy(reconstruct_patches(
        ones_patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    ))
    # recon_sum / counts ≈ recon_mean
    np.testing.assert_allclose(recon_sum / np.maximum(counts, 1), recon_mean, atol=1e-5)


def test_3d_reduction_sum_does_not_recover_original():
    x = np.ones((1, 8, 8, 8, 2), dtype="float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), strides=2, padding="valid")
    recon_sum = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 8, 8),
        strides=(2, 2, 2), padding="valid", reduction="sum",
    )
    recon_sum_np = ops.convert_to_numpy(recon_sum)
    assert recon_sum_np.max() > 1.5


def test_nonoverlap_reduction_doesnt_matter():
    """For non-overlapping patches, mean and sum are identical (count=1)."""
    x = np.random.RandomState(2).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon_mean = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        padding="valid", reduction="mean",
    ))
    recon_sum = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        padding="valid", reduction="sum",
    ))
    np.testing.assert_allclose(recon_mean, recon_sum, atol=1e-6)
    np.testing.assert_allclose(recon_mean, x, atol=1e-6)


def test_layer_reduction_via_init():
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    )
    assert layer.reduction == "sum"
    config = layer.get_config()
    assert config["reduction"] == "sum"
    restored = ReconstructPatches2D.from_config(config)
    assert restored.reduction == "sum"


def test_invalid_reduction_rejected():
    with pytest.raises(ValueError, match="reduction.*'mean' or 'sum'"):
        ReconstructPatches2D(size=(4, 4), output_size=(16, 16), reduction="median")
    with pytest.raises(ValueError, match="reduction.*'mean' or 'sum'"):
        ReconstructPatches3D(size=(4, 4, 4), output_size=(16, 16, 16), reduction="max")


def test_op_invalid_reduction_rejected():
    x = np.zeros((1, 4, 4, 48), dtype="float32")
    with pytest.raises(ValueError, match="reduction.*'mean' or 'sum'"):
        reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4), output_size=(16, 16), padding="valid",
            reduction="bogus",
        )
