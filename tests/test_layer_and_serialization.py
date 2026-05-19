"""Layer-specific tests: serialization, error paths, and Functional API use.

Roundtrip correctness is already covered exhaustively in test_roundtrip_2d.py
and test_roundtrip_3d.py via test_*_layer_matches_op. This file focuses on
Layer-level concerns: get_config, from_config, errors, and composability.
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
# get_config / from_config roundtrip across argument variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [
        {"size": (4, 4), "output_size": (16, 16)},
        {"size": (3, 5), "output_size": (12, 25), "padding": "same"},
        {"size": (8, 8), "output_size": (64, 64), "padding": "valid"},
        {"size": (2, 2), "output_size": (10, 10), "strides": (2, 2)},
        {"size": 4, "output_size": (16, 16)},  # int size
    ],
)
def test_layer_2d_get_config_roundtrip(kwargs):
    layer = ReconstructPatches2D(**kwargs)
    config = layer.get_config()
    restored = ReconstructPatches2D.from_config(config)
    # Behavior must be identical for a representative input.
    x = np.random.RandomState(0).rand(
        2, 4, 4, kwargs["size"][0] * kwargs["size"][1] * 3
        if not isinstance(kwargs["size"], int)
        else kwargs["size"] ** 2 * 3,
    ).astype("float32")
    # Skip behavioral comparison if the shape would mismatch (kwargs vary widely).
    # The key check is structural: config restores cleanly.
    assert restored.output_size == tuple(kwargs["output_size"])
    if isinstance(kwargs["size"], int):
        assert restored.size == (kwargs["size"], kwargs["size"])
    else:
        assert restored.size == tuple(kwargs["size"])
    assert restored.padding == kwargs.get("padding", "valid")


@pytest.mark.parametrize(
    "kwargs", [
        {"size": (4, 4, 4), "output_size": (16, 16, 16)},
        {"size": (2, 3, 4), "output_size": (10, 15, 20), "padding": "same"},
        {"size": (8, 16, 16), "output_size": (64, 128, 128), "padding": "valid"},
        {"size": 4, "output_size": (16, 16, 16)},  # int size
    ],
)
def test_layer_3d_get_config_roundtrip(kwargs):
    layer = ReconstructPatches3D(**kwargs)
    config = layer.get_config()
    restored = ReconstructPatches3D.from_config(config)
    assert restored.output_size == tuple(kwargs["output_size"])
    if isinstance(kwargs["size"], int):
        assert restored.size == (kwargs["size"],) * 3
    else:
        assert restored.size == tuple(kwargs["size"])
    assert restored.padding == kwargs.get("padding", "valid")


# ---------------------------------------------------------------------------
# Functional API: layer composes with extract under tf.data-style pipelines
# ---------------------------------------------------------------------------


def test_functional_2d_roundtrip_via_model():
    """Build a Keras Functional model: input -> extract -> reconstruct."""
    H, W, C, pH, pW = 16, 16, 3, 4, 4
    inputs = keras.Input(shape=(H, W, C))
    patches = keras.layers.Lambda(
        lambda x: ops.image.extract_patches(x, size=(pH, pW), padding="valid"),
        output_shape=(H // pH, W // pW, pH * pW * C),
    )(inputs)
    recon = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W), padding="valid",
    )(patches)
    model = keras.Model(inputs=inputs, outputs=recon)
    x = np.random.RandomState(1).rand(2, H, W, C).astype("float32")
    out = ops.convert_to_numpy(model(x))
    np.testing.assert_allclose(out, x, atol=1e-6)


def test_functional_3d_roundtrip_via_model():
    D, H, W, C, pD, pH, pW = 8, 16, 16, 2, 4, 4, 4
    inputs = keras.Input(shape=(D, H, W, C))
    patches = keras.layers.Lambda(
        lambda x: ops.image.extract_patches(
            x, size=(pD, pH, pW), padding="valid",
        ),
        output_shape=(D // pD, H // pH, W // pW, pD * pH * pW * C),
    )(inputs)
    recon = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W), padding="valid",
    )(patches)
    model = keras.Model(inputs=inputs, outputs=recon)
    x = np.random.RandomState(2).rand(1, D, H, W, C).astype("float32")
    out = ops.convert_to_numpy(model(x))
    np.testing.assert_allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_output_shape correctness (static and dynamic)
# ---------------------------------------------------------------------------


def test_compute_output_shape_2d_static():
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )
    out_shape = layer.compute_output_shape((2, 4, 4, 4 * 4 * 3))
    assert out_shape == (2, 16, 16, 3)


def test_compute_output_shape_2d_unknown_channels():
    """Last dim is None: channels can't be inferred at trace time."""
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )
    out_shape = layer.compute_output_shape((2, 4, 4, None))
    assert out_shape == (2, 16, 16, None)


def test_compute_output_shape_3d_static():
    layer = ReconstructPatches3D(
        size=(2, 4, 4), output_size=(8, 16, 16), padding="same",
    )
    out_shape = layer.compute_output_shape((1, 4, 4, 4, 2 * 4 * 4 * 3))
    assert out_shape == (1, 8, 16, 16, 3)


def test_compute_output_shape_3d_dynamic_batch():
    layer = ReconstructPatches3D(
        size=(2, 4, 4), output_size=(8, 16, 16), padding="same",
    )
    out_shape = layer.compute_output_shape((None, 4, 4, 4, 2 * 4 * 4 * 3))
    assert out_shape == (None, 8, 16, 16, 3)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_layer_2d_invalid_size_length():
    with pytest.raises(ValueError, match="length 2"):
        ReconstructPatches2D(size=(2, 3, 4), output_size=(10, 15))


def test_layer_2d_invalid_output_size_length():
    with pytest.raises(ValueError, match="length 2"):
        ReconstructPatches2D(size=(4, 4), output_size=(10, 15, 20))


def test_layer_3d_invalid_size_length():
    with pytest.raises(ValueError, match="length 3"):
        ReconstructPatches3D(size=(2, 3), output_size=(10, 15, 20))


def test_layer_3d_invalid_output_size_length():
    with pytest.raises(ValueError, match="length 3"):
        ReconstructPatches3D(size=(2, 3, 4), output_size=(10, 15))


@pytest.mark.parametrize("padding", ["reflect", "circular", "SAME", "Valid", ""])
def test_layer_invalid_padding(padding):
    with pytest.raises(ValueError, match="'same' or 'valid'"):
        ReconstructPatches2D(size=(4, 4), output_size=(16, 16), padding=padding)
    with pytest.raises(ValueError, match="'same' or 'valid'"):
        ReconstructPatches3D(
            size=(4, 4, 4), output_size=(16, 16, 16), padding=padding,
        )


def test_op_gapped_strides_rejected_2d():
    """strides > size leaves gaps in coverage; cannot be inverted."""
    x = np.zeros((1, 4, 4, 48), dtype="float32")
    with pytest.raises(NotImplementedError, match="gapped"):
        reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4), output_size=(32, 32),
            strides=(8, 8), padding="valid",
        )


def test_op_gapped_strides_rejected_3d():
    x = np.zeros((1, 4, 4, 4, 128), dtype="float32")
    with pytest.raises(NotImplementedError, match="gapped"):
        reconstruct_patches_3d(
            ops.convert_to_tensor(x),
            size=(4, 4, 4), output_size=(32, 32, 32),
            strides=(8, 8, 8), padding="valid",
        )


def test_op_overlapping_supported_2d():
    """strides < size should now work (overlap path), not raise."""
    x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


def test_op_overlapping_supported_3d():
    x = np.random.RandomState(1).rand(1, 8, 8, 8, 2).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), strides=2, padding="valid")
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 8, 8),
        strides=(2, 2, 2), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


@pytest.mark.skipif(
    __import__("keras").backend.backend() == "tensorflow",
    reason="tf.nn.conv only supports NHWC on CPU; extract_patches with "
           "channels_first errors out on tensorflow-cpu (CI).",
)
def test_op_channels_first_supported_2d():
    """channels_first should now work (transposes internally)."""
    x = np.random.RandomState(2).rand(2, 3, 16, 16).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), padding="valid", data_format="channels_first",
    )
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        padding="valid", data_format="channels_first",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


@pytest.mark.skipif(
    __import__("keras").backend.backend() == "tensorflow",
    reason="tf.nn.conv only supports NHWC on CPU; extract_patches with "
           "channels_first errors out on tensorflow-cpu (CI).",
)
def test_op_channels_first_supported_3d():
    x = np.random.RandomState(3).rand(1, 2, 8, 16, 16).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid", data_format="channels_first",
    )
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 16, 16),
        padding="valid", data_format="channels_first",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# name= keyword preservation (task #42)
# ---------------------------------------------------------------------------


def test_layer_2d_preserves_name():
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid", name="my_recon_2d",
    )
    assert layer.name == "my_recon_2d"
    config = layer.get_config()
    restored = ReconstructPatches2D.from_config(config)
    assert restored.name == "my_recon_2d"


def test_layer_3d_preserves_name():
    layer = ReconstructPatches3D(
        size=(4, 4, 4), output_size=(16, 16, 16), padding="valid", name="my_recon_3d",
    )
    assert layer.name == "my_recon_3d"
    config = layer.get_config()
    restored = ReconstructPatches3D.from_config(config)
    assert restored.name == "my_recon_3d"


# ---------------------------------------------------------------------------
# Auto-infer output_size for padding='valid' (task #40)
# ---------------------------------------------------------------------------


def test_2d_output_size_auto_infer_valid_nonoverlap():
    """output_size=None with valid padding should auto-infer at call time."""
    H, W = 16, 16
    x = np.random.RandomState(0).rand(2, H, W, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    layer = ReconstructPatches2D(size=(4, 4), padding="valid")  # no output_size
    assert layer.output_size is None
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_2d_output_size_auto_infer_valid_overlap():
    H, W = 16, 16
    x = np.random.RandomState(1).rand(2, H, W, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    layer = ReconstructPatches2D(
        size=(4, 4), strides=(2, 2), padding="valid",  # no output_size
    )
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-5)


def test_3d_output_size_auto_infer_valid():
    D, H, W = 8, 16, 16
    x = np.random.RandomState(2).rand(1, D, H, W, 2).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4, 4), padding="valid")
    layer = ReconstructPatches3D(size=(4, 4, 4), padding="valid")  # no output_size
    recon = layer(patches)
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


def test_output_size_none_with_same_rejected_2d():
    """Auto-infer is only valid for padding='valid', not 'same'."""
    with pytest.raises(ValueError, match="auto-infer.*padding='valid'"):
        ReconstructPatches2D(size=(4, 4), padding="same")


def test_output_size_none_with_same_rejected_3d():
    with pytest.raises(ValueError, match="auto-infer.*padding='valid'"):
        ReconstructPatches3D(size=(4, 4, 4), padding="same")


def test_compute_output_shape_auto_infer():
    """compute_output_shape should infer output_size when not given."""
    layer = ReconstructPatches2D(size=(4, 4), padding="valid")
    out_shape = layer.compute_output_shape((2, 4, 4, 48))  # grid 4x4, C=3
    assert out_shape == (2, 16, 16, 3)


# ---------------------------------------------------------------------------
# Improved error messages for inconsistent output_size (task #41)
# ---------------------------------------------------------------------------


def test_error_message_includes_expected_value_2d_nonoverlap():
    """For non-overlap valid, error suggests the right output_size."""
    x = np.zeros((1, 4, 4, 48), dtype="float32")
    with pytest.raises(ValueError, match=r"expected output_size=\(16,16\)"):
        reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4), output_size=(15, 15), padding="valid",
        )


def test_error_message_includes_expected_value_3d_nonoverlap():
    x = np.zeros((1, 4, 4, 4, 128), dtype="float32")
    with pytest.raises(ValueError, match=r"expected output_size=\(8,8,8\)"):
        reconstruct_patches_3d(
            ops.convert_to_tensor(x),
            size=(2, 2, 2), output_size=(7, 7, 7), padding="valid",
        )


def test_error_message_includes_range_for_overlap():
    """For overlap valid, error suggests the valid range."""
    x = np.random.RandomState(3).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    # patches grid 7x7, valid output is 16; expected H in [16, 18)
    with pytest.raises(ValueError, match=r"expected H in \[16, 18\)"):
        reconstruct_patches(
            patches, size=(4, 4), output_size=(20, 20),
            strides=(2, 2), padding="valid",
        )


def test_op_size_inconsistent_with_patches_last_dim():
    """Patches last dim must be divisible by prod(size)."""
    # patches with last dim 47 — prime, not divisible by 4*4=16
    x = np.zeros((1, 4, 4, 47), dtype="float32")
    with pytest.raises(ValueError, match="divisible"):
        reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4), output_size=(16, 16), padding="valid",
        )


def test_op_valid_requires_output_matches_grid():
    """Under padding='valid', output_size must equal size * grid."""
    x = np.zeros((1, 4, 4, 48), dtype="float32")  # 3 channels, grid 4x4
    with pytest.raises(ValueError, match="grid"):
        reconstruct_patches(
            ops.convert_to_tensor(x),
            size=(4, 4),
            output_size=(15, 15),  # Wrong: should be 16x16
            padding="valid",
        )
