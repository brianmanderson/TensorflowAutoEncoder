"""Direct tests for PR 2: overlap support + reduction='mean'/'sum'."""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import (
    gradient_image,
    gradient_volume,
    needs_overlap,
    needs_reduction,
)


@needs_overlap
@pytest.mark.parametrize(
    "H,W,C,size,stride,padding", [
        (16, 16, 3, (4, 4), (2, 2), "valid"),
        (32, 32, 3, (8, 8), (4, 4), "valid"),
        (16, 16, 3, (4, 4), (1, 1), "valid"),  # max overlap
        (17, 19, 3, (4, 4), (2, 2), "same"),
    ],
)
def test_2d_overlap_roundtrip(H, W, C, size, stride, padding):
    """reduction='mean' (default) recovers original from overlapping patches."""
    x = gradient_image(H, W, C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, strides=stride, padding=padding,
    )
    recon = keras.layers.ReconstructPatches2D(
        size=size, output_size=(H, W),
        strides=stride, padding=padding,
    )(patches)
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, rtol=1e-5, atol=1e-5,
        err_msg=f"2D overlap roundtrip failed for {(H, W, C, size, stride, padding)}",
    )


@needs_overlap
def test_3d_overlap_roundtrip():
    D, H, W, C = 8, 16, 16, 2
    x = gradient_volume(D, H, W, C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), strides=2, padding="valid",
    )
    recon = keras.layers.ReconstructPatches3D(
        size=(4, 4, 4), output_size=(D, H, W),
        strides=(2, 2, 2), padding="valid",
    )(patches)
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, rtol=1e-5, atol=1e-5,
    )


@needs_reduction
def test_reduction_sum_overshoots_at_overlap():
    """With all-ones input, sum reduction exceeds 1 where patches overlap."""
    x = np.ones((1, 16, 16, 3), dtype="float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), strides=2, padding="valid",
    )
    recon = keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    )(patches)
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.max() > 1.5, (
        f"Sum reduction should overshoot 1.0 at overlap regions; "
        f"got max={recon_np.max()}"
    )


@needs_reduction
def test_reduction_mean_equals_sum_div_count():
    """sum / per-pixel-count == mean (definitional)."""
    x = gradient_image(16, 16, C=3, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), strides=2, padding="valid",
    )
    recon_mean = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="mean",
    )(patches))
    recon_sum = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    )(patches))
    counts = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid", reduction="sum",
    )(ops.ones_like(patches)))
    np.testing.assert_allclose(
        recon_sum / np.maximum(counts, 1), recon_mean, atol=1e-5,
    )


@needs_overlap
def test_nonoverlap_unaffected_by_overlap_path():
    """For strides == size, the result is the same as without strides kwarg."""
    x = gradient_image(16, 16, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    no_strides = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )(patches))
    explicit_strides = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(4, 4), padding="valid",
    )(patches))
    np.testing.assert_allclose(no_strides, explicit_strides, atol=1e-6)


@needs_overlap
def test_gapped_strides_rejected_2d():
    with pytest.raises(NotImplementedError, match="gapped"):
        keras.layers.ReconstructPatches2D(
            size=(4, 4), output_size=(32, 32),
            strides=(8, 8), padding="valid",
        )


@needs_reduction
def test_invalid_reduction_rejected():
    with pytest.raises(ValueError, match="reduction.*'mean' or 'sum'"):
        keras.layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16), reduction="median",
        )
