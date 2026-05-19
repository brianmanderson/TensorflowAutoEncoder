"""Direct tests for PR 1: basic non-overlap reconstruct.

Imports `keras.layers.ReconstructPatches{2,3}D` and
`keras.ops.image.reconstruct_patches{,_3d}` directly — exercises whatever
keras version is installed (or first on PYTHONPATH).

These tests should PASS on any keras that has the basic-inverse PR landed.
They will SKIP entirely on keras versions that don't have the layers.
"""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import (
    gradient_image,
    gradient_volume,
    needs_reconstruct,
)


@needs_reconstruct
@pytest.mark.parametrize(
    "H,W,C,size,padding", [
        (16, 16, 3, (4, 4), "valid"),
        (24, 32, 1, (4, 8), "valid"),
        (15, 17, 3, (5, 5), "same"),
        (33, 41, 2, (5, 7), "same"),
        (28, 28, 1, (4, 4), "valid"),   # MNIST-ish
    ],
)
def test_2d_roundtrip(H, W, C, size, padding):
    x = gradient_image(H, W, C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    layer = keras.layers.ReconstructPatches2D(
        size=size, output_size=(H, W), padding=padding,
    )
    recon = layer(patches)
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-6,
        err_msg=f"2D roundtrip failed for {(H, W, C, size, padding)}",
    )


@needs_reconstruct
@pytest.mark.parametrize(
    "D,H,W,C,size,padding", [
        (8, 16, 16, 2, (4, 4, 4), "valid"),
        (16, 32, 32, 1, (4, 8, 8), "valid"),
        (9, 17, 19, 2, (4, 4, 4), "same"),
        (8, 8, 8, 3, (2, 2, 2), "valid"),
    ],
)
def test_3d_roundtrip(D, H, W, C, size, padding):
    x = gradient_volume(D, H, W, C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=size, padding=padding)
    layer = keras.layers.ReconstructPatches3D(
        size=size, output_size=(D, H, W), padding=padding,
    )
    recon = layer(patches)
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-6,
        err_msg=f"3D roundtrip failed for {(D, H, W, C, size, padding)}",
    )


@needs_reconstruct
def test_2d_unbatched_op():
    """The op supports unbatched (rank-3) input. The Layer enforces
    InputSpec(ndim=4), so unbatched is op-only."""
    H, W, C = 16, 16, 3
    x = gradient_image(H, W, C, batch=1)[0]
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = ops.image.reconstruct_patches(
        patches, size=(4, 4), output_size=(H, W), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_reconstruct
def test_get_config_roundtrip_2d():
    layer = keras.layers.ReconstructPatches2D(
        size=(3, 5), output_size=(12, 25), padding="same", name="my_recon_2d",
    )
    config = layer.get_config()
    restored = keras.layers.ReconstructPatches2D.from_config(config)
    assert restored.size == (3, 5)
    assert restored.output_size == (12, 25)
    assert restored.padding == "same"
    assert restored.name == "my_recon_2d"


@needs_reconstruct
def test_get_config_roundtrip_3d():
    layer = keras.layers.ReconstructPatches3D(
        size=(2, 3, 4), output_size=(10, 15, 20), padding="valid",
    )
    config = layer.get_config()
    restored = keras.layers.ReconstructPatches3D.from_config(config)
    assert restored.size == (2, 3, 4)
    assert restored.output_size == (10, 15, 20)
    assert restored.padding == "valid"


@needs_reconstruct
def test_invalid_size_2d():
    with pytest.raises(ValueError, match="length 2"):
        keras.layers.ReconstructPatches2D(size=(2, 3, 4), output_size=(10, 15))


@needs_reconstruct
def test_invalid_padding():
    with pytest.raises(ValueError, match="'same' or 'valid'"):
        keras.layers.ReconstructPatches2D(
            size=(2, 2), output_size=(8, 8), padding="reflect",
        )


@needs_reconstruct
def test_op_2d_matches_layer():
    """The op and Layer should give bit-identical output."""
    x = gradient_image(16, 16, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    op_out = ops.convert_to_numpy(ops.image.reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    ))
    layer_out = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )(patches))
    np.testing.assert_array_equal(op_out, layer_out)
