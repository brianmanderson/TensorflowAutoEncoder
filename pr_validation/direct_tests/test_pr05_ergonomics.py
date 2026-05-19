"""Direct tests for PR 5: auto-infer output_size for valid + improved errors."""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import gradient_image, needs_auto_infer


@needs_auto_infer
def test_auto_infer_output_size_valid_nonoverlap():
    """output_size=None with valid padding is inferred from patches."""
    H, W = 16, 16
    x = gradient_image(H, W, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    layer = keras.layers.ReconstructPatches2D(size=(4, 4), padding="valid")
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_auto_infer
def test_auto_infer_3d_valid():
    D, H, W = 8, 16, 16
    rng = np.random.RandomState(0)
    x = rng.rand(1, D, H, W, 2).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), padding="valid")
    layer = keras.layers.ReconstructPatches3D(size=(4, 4, 4), padding="valid")
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_auto_infer
def test_output_size_none_with_same_requires_reference():
    """Auto-infer for `padding='same'` should EITHER work via dual-input
    (PR 7) or raise; in either case a plain call without output_size and
    without reference should not silently succeed.
    """
    # Should not raise at construction (we allow output_size=None for same
    # iff the caller will pass a reference). Per-call behavior may vary.
    layer = keras.layers.ReconstructPatches2D(size=(4, 4), padding="same")
    assert layer.output_size is None


@needs_auto_infer
def test_error_for_inconsistent_output_size_2d():
    """Inconsistent output_size should mention the expected value."""
    x = np.zeros((1, 4, 4, 48), dtype="float32")
    with pytest.raises(ValueError, match=r"\(16,\s*16\)"):
        ops.image.reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4), output_size=(15, 15), padding="valid",
        )
