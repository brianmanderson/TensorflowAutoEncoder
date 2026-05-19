"""Direct tests for PR 4: dilation_rate parameter.

Skipped on TF-CPU (dilated conv_transpose unavailable) automatically via
runtime errors caught by the feature probe.
"""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import needs_dilation


@needs_dilation
@pytest.mark.parametrize(
    "H,W,C,size,dilation,padding", [
        (16, 16, 3, (3, 3), 2, "valid"),
        (16, 16, 3, (3, 3), 2, "same"),
        (20, 20, 2, (5, 5), 2, "valid"),
    ],
)
def test_2d_dilation_roundtrip(H, W, C, size, dilation, padding):
    """With dilation_rate > 1, forward op requires strides==1; that's overlap."""
    rng = np.random.RandomState(hash((H, W, size, dilation)) & 0xFFFF)
    x = rng.rand(2, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    try:
        patches = ops.image.extract_patches(
            x_t, size=size, strides=1, dilation_rate=dilation, padding=padding,
        )
    except Exception as e:
        # Forward op may not support this combo on this backend; skip.
        pytest.skip(f"forward extract_patches doesn't support dilation here: {e}")
    recon = keras.layers.ReconstructPatches2D(
        size=size, output_size=(H, W),
        strides=(1, 1), padding=padding, dilation_rate=dilation,
    )(patches)
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-5,
        err_msg=f"2D dilation roundtrip failed for size={size}, dilation={dilation}, {padding}",
    )


@needs_dilation
def test_default_dilation_unchanged():
    """dilation_rate=1 (default) must produce same output as omitting the kwarg."""
    x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    a = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )(patches))
    b = ops.convert_to_numpy(keras.layers.ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid", dilation_rate=1,
    )(patches))
    np.testing.assert_allclose(a, b, atol=1e-6)
