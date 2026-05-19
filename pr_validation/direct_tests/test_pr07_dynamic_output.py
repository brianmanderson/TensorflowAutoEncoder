"""Direct tests for PR 7: dual-input dynamic output_size mode."""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import needs_dual_input


@needs_dual_input
def test_dual_input_2d_roundtrip():
    H, W, C = 16, 16, 3
    x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    layer = keras.layers.ReconstructPatches2D(size=(4, 4), padding="valid")
    recon = layer([patches, x_t])
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_dual_input
def test_dual_input_3d_same_padding():
    """The motivating case: padding='same' with non-divisible input."""
    D, H, W, C = 9, 17, 19, 2
    x = np.random.RandomState(1).rand(1, D, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), padding="same")
    layer = keras.layers.ReconstructPatches3D(size=(4, 4, 4), padding="same")
    recon = layer([patches, x_t])
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_dual_input
def test_variable_input_model_handles_multiple_sizes():
    """One compiled model accepts different concrete spatial shapes."""
    inputs = keras.Input(shape=(None, None, None, 1))
    x = keras.layers.Conv3D(2, 3, padding="same")(inputs)
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(t, size=(4, 4, 4), padding="same"),
        output_shape=(None, None, None, 4 * 4 * 4 * 2),
    )(x)
    recon = keras.layers.ReconstructPatches3D(
        size=(4, 4, 4), padding="same",
    )([patches, x])
    model = keras.Model(inputs, recon)
    rng = np.random.RandomState(2)
    for shape in [(1, 8, 16, 16, 1), (1, 12, 20, 20, 1)]:
        x_test = rng.rand(*shape).astype("float32")
        out = ops.convert_to_numpy(model(x_test))
        expected = (shape[0], shape[1], shape[2], shape[3], 2)
        assert tuple(out.shape) == expected, (
            f"For input {shape}, expected {expected}, got {out.shape}"
        )
