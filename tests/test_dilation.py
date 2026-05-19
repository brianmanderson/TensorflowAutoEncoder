"""Tests for `dilation_rate` support (atrous patches).

The forward `keras.ops.image.extract_patches` accepts `dilation_rate`; our
reconstruct path now handles it too. `extract_patches` constrains
`dilation_rate > 1` to require `strides == 1` (overlap-only), so the
reconstruct path uses the conv-transpose code (dilation has no meaning
in the non-overlap reshape path).
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
# 2D dilation roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "H,W,C,size,dilation,padding", [
        (16, 16, 3, (3, 3), 2, "valid"),
        (16, 16, 3, (3, 3), 3, "valid"),
        (20, 20, 2, (5, 5), 2, "valid"),
        (16, 16, 3, (3, 3), 2, "same"),
        (17, 19, 3, (3, 3), 2, "same"),
        (16, 16, 1, (3, 3), (1, 2), "valid"),  # asymmetric dilation
    ],
)
def test_2d_dilation_roundtrip(H, W, C, size, dilation, padding):
    x = np.random.RandomState(hash((H, W, C, dilation)) & 0xFFFF).rand(
        2, H, W, C,
    ).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, strides=1, dilation_rate=dilation, padding=padding,
    )
    recon = reconstruct_patches(
        patches, size=size, output_size=(H, W),
        strides=(1, 1), padding=padding, dilation_rate=dilation,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-5,
        err_msg=f"2D dilation roundtrip failed for size={size}, dilation={dilation}, {padding}",
    )


# ---------------------------------------------------------------------------
# 3D dilation roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,H,W,C,size,dilation,padding", [
        (12, 12, 12, 2, (3, 3, 3), 2, "valid"),
        (12, 12, 12, 1, (3, 3, 3), 2, "same"),
        (8, 8, 8, 2, (2, 2, 2), 2, "valid"),
    ],
)
def test_3d_dilation_roundtrip(D, H, W, C, size, dilation, padding):
    x = np.random.RandomState(hash((D, H, W, dilation)) & 0xFFFF).rand(
        1, D, H, W, C,
    ).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=size, strides=1, dilation_rate=dilation, padding=padding,
    )
    recon = reconstruct_patches_3d(
        patches, size=size, output_size=(D, H, W),
        strides=(1, 1, 1), padding=padding, dilation_rate=dilation,
    )
    np.testing.assert_allclose(
        ops.convert_to_numpy(recon), x, atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Layer parity
# ---------------------------------------------------------------------------


def test_2d_dilation_via_layer():
    H, W, C = 16, 16, 3
    x = np.random.RandomState(0).rand(1, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(3, 3), strides=1, dilation_rate=2, padding="valid",
    )
    layer = ReconstructPatches2D(
        size=(3, 3), output_size=(H, W),
        strides=(1, 1), padding="valid", dilation_rate=2,
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


def test_3d_dilation_via_layer():
    D, H, W, C = 12, 12, 12, 2
    x = np.random.RandomState(1).rand(1, D, H, W, C).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(3, 3, 3), strides=1, dilation_rate=2, padding="valid",
    )
    layer = ReconstructPatches3D(
        size=(3, 3, 3), output_size=(D, H, W),
        strides=(1, 1, 1), padding="valid", dilation_rate=2,
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Layer config preserves dilation_rate
# ---------------------------------------------------------------------------


def test_layer_get_config_includes_dilation():
    layer = ReconstructPatches2D(
        size=(3, 3), output_size=(16, 16),
        strides=(1, 1), padding="valid", dilation_rate=2,
    )
    config = layer.get_config()
    assert config["dilation_rate"] == 2
    restored = ReconstructPatches2D.from_config(config)
    assert restored.dilation_rate == 2


def test_3d_layer_get_config_includes_dilation():
    layer = ReconstructPatches3D(
        size=(3, 3, 3), output_size=(12, 12, 12),
        strides=(1, 1, 1), padding="valid", dilation_rate=2,
    )
    config = layer.get_config()
    assert config["dilation_rate"] == 2
    restored = ReconstructPatches3D.from_config(config)
    assert restored.dilation_rate == 2


# ---------------------------------------------------------------------------
# default dilation_rate=1 doesn't break anything
# ---------------------------------------------------------------------------


def test_default_dilation_unchanged_2d():
    """dilation_rate=1 (default) must produce the same output as before."""
    x = np.random.RandomState(2).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        padding="valid",  # dilation_rate defaults to 1
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)
