"""Direct tests for PR 6: ExtractPatches2D / ExtractPatches3D Layer wrappers."""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import needs_extract_layer, needs_reconstruct


@needs_extract_layer
@pytest.mark.parametrize(
    "H,W,C,size,padding,strides", [
        (16, 16, 3, (4, 4), "valid", None),
        (17, 19, 3, (4, 4), "same", None),
        (16, 16, 3, (4, 4), "valid", (2, 2)),
    ],
)
def test_extract2d_matches_op(H, W, C, size, padding, strides):
    rng = np.random.RandomState(0)
    x = rng.rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    op_out = ops.convert_to_numpy(ops.image.extract_patches(
        x_t, size=size, strides=strides, padding=padding,
    ))
    layer_out = ops.convert_to_numpy(keras.layers.ExtractPatches2D(
        size=size, strides=strides, padding=padding,
    )(x_t))
    np.testing.assert_array_equal(op_out, layer_out)


@needs_extract_layer
def test_extract3d_matches_op():
    rng = np.random.RandomState(1)
    x = rng.rand(1, 8, 16, 16, 2).astype("float32")
    x_t = ops.convert_to_tensor(x)
    op_out = ops.convert_to_numpy(ops.image.extract_patches_3d(
        x_t, size=(4, 4, 4), padding="valid",
    ))
    layer_out = ops.convert_to_numpy(keras.layers.ExtractPatches3D(
        size=(4, 4, 4), padding="valid",
    )(x_t))
    np.testing.assert_array_equal(op_out, layer_out)


@needs_extract_layer
@needs_reconstruct
def test_symmetric_layer_pair_2d():
    H, W, C = 16, 16, 3
    x = np.random.RandomState(0).rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = keras.layers.ExtractPatches2D(size=(4, 4), padding="valid")(x_t)
    recon = keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
    )(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_extract_layer
def test_extract2d_get_config_roundtrip():
    layer = keras.layers.ExtractPatches2D(
        size=(3, 5), strides=(2, 3), padding="same",
    )
    config = layer.get_config()
    restored = keras.layers.ExtractPatches2D.from_config(config)
    assert restored.size == (3, 5)
    assert restored.padding == "same"


@needs_extract_layer
def test_extract2d_compute_output_shape():
    layer = keras.layers.ExtractPatches2D(size=(4, 4), padding="valid")
    assert layer.compute_output_shape((None, 16, 16, 3)) == (None, 4, 4, 48)
