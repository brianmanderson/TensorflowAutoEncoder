"""NaN / Inf propagation tests.

If the input to `reconstruct_patches` contains NaN or Inf, the layer should
not crash. It should propagate the special values cleanly so downstream
ops can detect them. Important when working with masked regions in medical
imaging or sentinel values.

Both code paths exercised: non-overlap (reshape/transpose) and overlap
(conv_transpose + division).
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import reconstruct_patches, reconstruct_patches_3d


def test_2d_nonoverlap_nan_propagates():
    """A single NaN patch should produce NaN somewhere in the output."""
    x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
    x[0, 5, 5, 0] = np.nan
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert np.isnan(recon_np).any(), "NaN did not propagate through the layer"
    # And the non-NaN regions should still match the original (the layer
    # didn't propagate NaN everywhere via some bug).
    mask = ~np.isnan(x) & ~np.isnan(recon_np)
    np.testing.assert_allclose(recon_np[mask], x[mask], atol=1e-6)


def test_2d_nonoverlap_inf_propagates():
    x = np.random.RandomState(1).rand(1, 16, 16, 3).astype("float32")
    x[0, 10, 10, 1] = np.inf
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert np.isinf(recon_np).any(), "Inf did not propagate"
    # Inf should be at the same location.
    assert np.isinf(recon_np[0, 10, 10, 1])


def test_3d_nonoverlap_nan_propagates():
    x = np.random.RandomState(2).rand(1, 8, 8, 8, 2).astype("float32")
    x[0, 3, 3, 3, 0] = np.nan
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(2, 2, 2), padding="valid")
    recon = reconstruct_patches_3d(
        patches, size=(2, 2, 2), output_size=(8, 8, 8), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert np.isnan(recon_np).any(), "NaN did not propagate"
    # NaN should be at the same position
    assert np.isnan(recon_np[0, 3, 3, 3, 0])


def test_2d_overlap_nan_propagates():
    """Overlap path divides by count — NaN should still propagate."""
    x = np.random.RandomState(3).rand(1, 16, 16, 3).astype("float32")
    x[0, 8, 8, 0] = np.nan
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert np.isnan(recon_np).any()


def test_2d_layer_doesnt_crash_on_all_nan():
    """All-NaN input shouldn't crash — output should be all NaN."""
    x = np.full((1, 16, 16, 3), np.nan, dtype="float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    assert np.isnan(recon_np).all(), "All-NaN input should produce all-NaN output"
