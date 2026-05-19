"""Tests for ExtractPatches2D and ExtractPatches3D Layer wrappers.

These are thin Layer wrappers around keras.ops.image.extract_patches. They
let users build Functional/Sequential models with `extract -> reconstruct`
as symmetric Layer pairs (no Lambda needed). Saves boilerplate and avoids
the Lambda-deserialization gotcha.
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import (
    ExtractPatches2D,
    ExtractPatches3D,
    ReconstructPatches2D,
    ReconstructPatches3D,
)


# ---------------------------------------------------------------------------
# Output matches the underlying op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "H,W,C,size,padding,strides", [
        (16, 16, 3, (4, 4), "valid", None),
        (16, 16, 3, (4, 4), "valid", (2, 2)),
        (17, 19, 3, (4, 4), "same", None),
        (32, 32, 1, 8, "valid", None),  # int size
    ],
)
def test_extract2d_matches_op(H, W, C, size, padding, strides):
    x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    op_out = ops.convert_to_numpy(ops.image.extract_patches(
        x_t, size=size, strides=strides, padding=padding,
    ))
    layer_out = ops.convert_to_numpy(ExtractPatches2D(
        size=size, strides=strides, padding=padding,
    )(x_t))
    np.testing.assert_array_equal(op_out, layer_out)


@pytest.mark.parametrize(
    "D,H,W,C,size,padding", [
        (8, 16, 16, 2, (4, 4, 4), "valid"),
        (9, 17, 19, 2, (4, 4, 4), "same"),
        (8, 8, 8, 3, 4, "valid"),  # int size -> the layer normalizes to (4,4,4)
    ],
)
def test_extract3d_matches_op(D, H, W, C, size, padding):
    x = np.random.RandomState(1).rand(1, D, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    # Compare against extract_patches_3d (the explicit 3D entry point).
    # `extract_patches` dispatches on size length: int -> 2D, len-3 -> 3D.
    op_out = ops.convert_to_numpy(ops.image.extract_patches_3d(
        x_t, size=size, padding=padding,
    ))
    layer_out = ops.convert_to_numpy(ExtractPatches3D(
        size=size, padding=padding,
    )(x_t))
    np.testing.assert_array_equal(op_out, layer_out)


# ---------------------------------------------------------------------------
# compute_output_shape
# ---------------------------------------------------------------------------


def test_extract2d_compute_output_shape_valid():
    layer = ExtractPatches2D(size=(4, 4), padding="valid")
    assert layer.compute_output_shape((None, 16, 16, 3)) == (None, 4, 4, 48)


def test_extract2d_compute_output_shape_same():
    layer = ExtractPatches2D(size=(4, 4), padding="same")
    assert layer.compute_output_shape((None, 17, 19, 3)) == (None, 5, 5, 48)


def test_extract3d_compute_output_shape_valid():
    layer = ExtractPatches3D(size=(4, 4, 4), padding="valid")
    assert layer.compute_output_shape((None, 8, 16, 16, 2)) == (None, 2, 4, 4, 128)


def test_extract2d_compute_output_shape_unknown_spatial():
    layer = ExtractPatches2D(size=(4, 4), padding="valid")
    assert layer.compute_output_shape((None, None, None, 3)) == (None, None, None, 48)


# ---------------------------------------------------------------------------
# Symmetric extract -> reconstruct via Layer pair, no Lambda
# ---------------------------------------------------------------------------


def test_2d_symmetric_layer_pair_via_functional():
    H, W, C = 16, 16, 3
    inputs = keras.Input(shape=(H, W, C))
    patches = ExtractPatches2D(size=(4, 4), padding="valid")(inputs)
    recon = ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
    )(patches)
    model = keras.Model(inputs, recon)
    x = np.random.RandomState(2).rand(2, H, W, C).astype("float32")
    out = ops.convert_to_numpy(model(x))
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_3d_symmetric_layer_pair_via_functional():
    D, H, W, C = 8, 16, 16, 2
    inputs = keras.Input(shape=(D, H, W, C))
    patches = ExtractPatches3D(size=(4, 4, 4), padding="valid")(inputs)
    recon = ReconstructPatches3D(
        size=(4, 4, 4), output_size=(D, H, W), padding="valid",
    )(patches)
    model = keras.Model(inputs, recon)
    x = np.random.RandomState(3).rand(1, D, H, W, C).astype("float32")
    out = ops.convert_to_numpy(model(x))
    np.testing.assert_allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# get_config / serialization
# ---------------------------------------------------------------------------


def test_extract2d_get_config_roundtrip():
    layer = ExtractPatches2D(
        size=(3, 5), strides=(2, 3), padding="same", name="my_extract",
    )
    config = layer.get_config()
    restored = ExtractPatches2D.from_config(config)
    assert restored.size == (3, 5)
    assert restored.strides == (2, 3)
    assert restored.padding == "same"
    assert restored.name == "my_extract"


def test_extract3d_get_config_roundtrip():
    layer = ExtractPatches3D(
        size=(2, 4, 4), strides=(2, 2, 2), padding="valid",
    )
    config = layer.get_config()
    restored = ExtractPatches3D.from_config(config)
    assert restored.size == (2, 4, 4)
    assert restored.padding == "valid"


def test_extract_save_load_in_model():
    """A model with ExtractPatches Layer should save/load without custom_objects."""
    import os
    import tempfile

    H, W, C = 16, 16, 3
    inputs = keras.Input(shape=(H, W, C))
    patches = ExtractPatches2D(size=(4, 4), padding="valid")(inputs)
    recon = ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
    )(patches)
    model = keras.Model(inputs, recon)
    x = np.random.RandomState(4).rand(1, H, W, C).astype("float32")
    out_before = ops.convert_to_numpy(model(x))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
    out_after = ops.convert_to_numpy(loaded(x))
    np.testing.assert_array_equal(out_before, out_after)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_extract2d_invalid_size():
    with pytest.raises(ValueError, match="length 2"):
        ExtractPatches2D(size=(2, 3, 4))


def test_extract3d_invalid_size():
    with pytest.raises(ValueError, match="length 3"):
        ExtractPatches3D(size=(2, 3))


def test_extract2d_invalid_padding():
    with pytest.raises(ValueError, match="'same' or 'valid'"):
        ExtractPatches2D(size=(4, 4), padding="reflect")
