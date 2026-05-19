"""Direct tests for PR 3: channels_first data format support.

Skipped on TF-CPU (NCHW conv unavailable) via the channels_first
feature-detection probe; tests will skip silently when the installed
keras hasn't landed PR 3 yet.
"""

import keras
import numpy as np
import pytest
from keras import ops

from .conftest import needs_channels_first


@needs_channels_first
@pytest.mark.parametrize(
    "H,W,C,size,padding", [
        (16, 16, 3, (4, 4), "valid"),
        (15, 17, 3, (5, 5), "same"),
        (32, 32, 1, (8, 8), "valid"),
    ],
)
def test_2d_channels_first_roundtrip(H, W, C, size, padding):
    # Build channels_first input: (B, C, H, W)
    rng = np.random.RandomState(hash((H, W, C)) & 0xFFFF)
    x = rng.rand(2, C, H, W).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, padding=padding, data_format="channels_first",
    )
    recon = keras.layers.ReconstructPatches2D(
        size=size, output_size=(H, W), padding=padding,
        data_format="channels_first",
    )(patches)
    assert tuple(recon.shape) == x.shape
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@needs_channels_first
def test_3d_channels_first_roundtrip():
    D, H, W, C = 8, 16, 16, 2
    rng = np.random.RandomState(0)
    x = rng.rand(1, C, D, H, W).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid", data_format="channels_first",
    )
    recon = keras.layers.ReconstructPatches3D(
        size=(4, 4, 4), output_size=(D, H, W), padding="valid",
        data_format="channels_first",
    )(patches)
    assert tuple(recon.shape) == x.shape
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)
