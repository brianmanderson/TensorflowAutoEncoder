"""Model save/load roundtrip tests.

`test_layer_and_serialization.py` covers get_config / from_config in isolation.
This file goes further: builds a real `keras.Model`, saves to a `.keras` file,
loads back via `keras.models.load_model`, runs inference, and asserts outputs
match. Exercises the full Keras serialization path (registry, custom_objects,
weight restoration), which `get_config` alone doesn't cover.

Note: we avoid `keras.layers.Lambda` with a Python lambda inside test models
because Keras's safe-load policy blocks deserializing arbitrary Python code.
Instead we feed pre-computed patches directly to ReconstructPatches as a
model input — this isolates the question to "does *our layer* round-trip?",
which is what we actually need to verify.
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import ReconstructPatches2D, ReconstructPatches3D


def _save_load_and_compare(model, x):
    """Save `model` to a temp .keras, load back, assert outputs match."""
    out_before = ops.convert_to_numpy(model(x))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
    out_after = ops.convert_to_numpy(loaded(x))
    np.testing.assert_array_equal(
        out_before, out_after,
        err_msg="Saved-loaded model output differs from original",
    )
    return loaded


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------


def test_2d_save_load_valid():
    H, W, C, pH, pW = 16, 16, 3, 4, 4
    patches_input = keras.Input(shape=(H // pH, W // pW, pH * pW * C))
    recon = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W), padding="valid",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(0).rand(2, H // pH, W // pW, pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


def test_2d_save_load_same():
    H, W, C, pH, pW = 17, 19, 3, 4, 4
    grid_h = (H + pH - 1) // pH
    grid_w = (W + pW - 1) // pW
    patches_input = keras.Input(shape=(grid_h, grid_w, pH * pW * C))
    recon = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W), padding="same",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(1).rand(2, grid_h, grid_w, pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


def test_2d_save_load_overlap():
    H, W, C, pH, pW = 16, 16, 3, 4, 4
    sH, sW = 2, 2
    grid_h = (H - pH) // sH + 1
    grid_w = (W - pW) // sW + 1
    patches_input = keras.Input(shape=(grid_h, grid_w, pH * pW * C))
    recon = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W),
        strides=(sH, sW), padding="valid",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(2).rand(2, grid_h, grid_w, pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_3d_save_load_valid():
    D, H, W, C = 8, 16, 16, 2
    pD, pH, pW = 4, 4, 4
    patches_input = keras.Input(shape=(D // pD, H // pH, W // pW, pD * pH * pW * C))
    recon = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W), padding="valid",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(3).rand(2, D // pD, H // pH, W // pW, pD * pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


def test_3d_save_load_same():
    D, H, W, C = 9, 17, 19, 2
    pD, pH, pW = 4, 4, 4
    grid_d = (D + pD - 1) // pD
    grid_h = (H + pH - 1) // pH
    grid_w = (W + pW - 1) // pW
    patches_input = keras.Input(shape=(grid_d, grid_h, grid_w, pD * pH * pW * C))
    recon = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W), padding="same",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(4).rand(2, grid_d, grid_h, grid_w, pD * pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


def test_3d_save_load_overlap():
    D, H, W, C = 8, 8, 8, 2
    pD, pH, pW = 4, 4, 4
    sD, sH, sW = 2, 2, 2
    grid_d = (D - pD) // sD + 1
    grid_h = (H - pH) // sH + 1
    grid_w = (W - pW) // sW + 1
    patches_input = keras.Input(shape=(grid_d, grid_h, grid_w, pD * pH * pW * C))
    recon = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W),
        strides=(sD, sH, sW), padding="valid",
    )(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(5).rand(2, grid_d, grid_h, grid_w, pD * pH * pW * C).astype("float32")
    _save_load_and_compare(model, x)


# ---------------------------------------------------------------------------
# Verify restored layer's config matches
# ---------------------------------------------------------------------------


def test_2d_loaded_layer_config_matches():
    H, W, pH, pW = 32, 32, 8, 8
    patches_input = keras.Input(shape=(H // pH, W // pW, pH * pW * 3))
    layer = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W), padding="valid",
        strides=(pH, pW), name="my_recon_2d",
    )
    recon = layer(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(6).rand(1, H // pH, W // pW, pH * pW * 3).astype("float32")
    loaded = _save_load_and_compare(model, x)
    restored_layer = loaded.get_layer("my_recon_2d")
    assert restored_layer.size == (pH, pW)
    assert restored_layer.output_size == (H, W)
    assert restored_layer.padding == "valid"


def test_3d_loaded_layer_config_matches():
    D, H, W, pD, pH, pW = 8, 16, 16, 4, 4, 4
    patches_input = keras.Input(shape=(D // pD, H // pH, W // pW, pD * pH * pW * 2))
    layer = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W), padding="valid",
        name="my_recon_3d",
    )
    recon = layer(patches_input)
    model = keras.Model(patches_input, recon)
    x = np.random.RandomState(7).rand(1, D // pD, H // pH, W // pW, pD * pH * pW * 2).astype("float32")
    loaded = _save_load_and_compare(model, x)
    restored_layer = loaded.get_layer("my_recon_3d")
    assert restored_layer.size == (pD, pH, pW)
    assert restored_layer.output_size == (D, H, W)
    assert restored_layer.padding == "valid"
