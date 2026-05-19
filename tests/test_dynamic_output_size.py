"""Dual-input mode: dynamic output_size via reference tensor.

The user's AttemptingHeadNetwork.py declares `Input(shape=[None,None,None,1])`
and the original ReconstructVolumePatchesLayer took `[patches, original]` as
dual input, deriving output spatial dims from `original` at call time. Our
layer supports the same pattern when called with a list:

    ReconstructPatches3D(size=...)([patches, reference])

This file proves the mode works for:
- 2D and 3D variants
- Multiple different concrete input sizes through the same compiled model
  (proving the output_size is genuinely dynamic, not cached at trace time)
- Inside a Functional API model with Input(shape=[None,...])
- Roundtrip identity: reconstruct(extract(x), ref=x) == x
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


# ---------------------------------------------------------------------------
# Basic dual-input roundtrip
# ---------------------------------------------------------------------------


def test_2d_dual_input_roundtrip():
    H, W, C = 16, 16, 3
    x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    layer = ReconstructPatches2D(size=(4, 4), padding="valid")  # no output_size
    recon = layer([patches, x_t])
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_dual_input_roundtrip():
    D, H, W, C = 8, 16, 16, 2
    x = np.random.RandomState(1).rand(1, D, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), padding="valid")
    layer = ReconstructPatches3D(size=(4, 4, 4), padding="valid")
    recon = layer([patches, x_t])
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_3d_dual_input_same_padding():
    """The motivating case: padding='same' with non-divisible input."""
    D, H, W, C = 25, 59, 55, 2
    x = np.random.RandomState(2).rand(1, D, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(8, 16, 16), padding="same")
    layer = ReconstructPatches3D(size=(8, 16, 16), padding="same")
    recon = layer([patches, x_t])
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Variable-input model (the user's autoencoder pattern)
# ---------------------------------------------------------------------------


def test_3d_variable_input_model_handles_multiple_sizes():
    """One model, multiple input sizes, same dual-input layer."""
    inputs = keras.Input(shape=(None, None, None, 1))
    x = keras.layers.Conv3D(2, 3, padding="same")(inputs)
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(t, size=(4, 4, 4), padding="same"),
        output_shape=(None, None, None, 4 * 4 * 4 * 2),
    )(x)
    recon = ReconstructPatches3D(size=(4, 4, 4), padding="same")([patches, x])
    model = keras.Model(inputs, recon)

    # Symbolic output shape should be fully dynamic
    assert recon.shape == (None, None, None, None, 2)

    # Same compiled model handles different concrete sizes.
    rng = np.random.RandomState(3)
    for shape in [(1, 16, 32, 32, 1), (1, 8, 24, 24, 1), (2, 32, 64, 64, 1)]:
        x_test = rng.rand(*shape).astype("float32")
        out = ops.convert_to_numpy(model(x_test))
        # Conv3D output has 2 channels; the model preserves spatial dims.
        expected_shape = (shape[0], shape[1], shape[2], shape[3], 2)
        assert tuple(out.shape) == expected_shape, (
            f"For input {shape}, expected output shape {expected_shape}, "
            f"got {out.shape}"
        )


def test_2d_variable_input_model():
    inputs = keras.Input(shape=(None, None, 3))
    x = keras.layers.Conv2D(4, 3, padding="same")(inputs)
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(t, size=(4, 4), padding="same"),
        output_shape=(None, None, 4 * 4 * 4),
    )(x)
    recon = ReconstructPatches2D(size=(4, 4), padding="same")([patches, x])
    model = keras.Model(inputs, recon)
    assert recon.shape == (None, None, None, 4)
    rng = np.random.RandomState(4)
    for shape in [(1, 16, 16, 3), (2, 24, 28, 3), (1, 33, 41, 3)]:
        x_test = rng.rand(*shape).astype("float32")
        out = ops.convert_to_numpy(model(x_test))
        assert tuple(out.shape) == (shape[0], shape[1], shape[2], 4)


# ---------------------------------------------------------------------------
# Error paths for dual-input
# ---------------------------------------------------------------------------


def test_2d_dual_input_wrong_list_length():
    x = np.random.RandomState(5).rand(1, 4, 4, 48).astype("float32")
    ref = np.random.RandomState(6).rand(1, 16, 16, 3).astype("float32")
    layer = ReconstructPatches2D(size=(4, 4), padding="valid")
    with pytest.raises(ValueError, match="exactly.*patches, reference"):
        layer([ops.convert_to_tensor(x)])


def test_3d_dual_input_wrong_list_length():
    x = np.random.RandomState(7).rand(1, 4, 4, 4, 128).astype("float32")
    ref = np.random.RandomState(8).rand(1, 16, 16, 16, 2).astype("float32")
    layer = ReconstructPatches3D(size=(4, 4, 4), padding="valid")
    with pytest.raises(ValueError, match="exactly.*patches, reference"):
        layer([
            ops.convert_to_tensor(x),
            ops.convert_to_tensor(ref),
            ops.convert_to_tensor(ref),
        ])


# ---------------------------------------------------------------------------
# compute_output_shape with dual input (list of shapes)
# ---------------------------------------------------------------------------


def test_2d_compute_output_shape_dual_input():
    layer = ReconstructPatches2D(size=(4, 4), padding="same")
    out = layer.compute_output_shape([(None, 4, 4, 48), (None, 16, 16, 3)])
    assert out == (None, 16, 16, 3)


def test_3d_compute_output_shape_dual_input():
    layer = ReconstructPatches3D(size=(4, 4, 4), padding="same")
    out = layer.compute_output_shape(
        [(None, 2, 4, 4, 128), (None, 8, 16, 16, 2)],
    )
    assert out == (None, 8, 16, 16, 2)
