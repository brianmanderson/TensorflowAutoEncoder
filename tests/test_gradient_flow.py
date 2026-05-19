"""Gradient-flow tests for ReconstructPatches{2,3}D.

The unit roundtrip tests prove correctness in inference. These tests prove
gradients reach upstream trainable weights when the layer is part of a
training pipeline — catching sign-flips, dropped gradients, or NaN-in-grad
bugs that don't show up at inference time.

Strategy: build a tiny model `Dense → extract_patches → ReconstructPatches
→ Dense` and verify that (1) `fit()` for two epochs produces different
losses, and (2) trainable weights actually change after one step. Strong
evidence gradients flow through both the layer and through `extract_patches`.
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import ReconstructPatches2D, ReconstructPatches3D


def _snapshot(weights):
    """Backend-agnostic copy of trainable weights as a list of numpy arrays."""
    return [np.array(ops.convert_to_numpy(w)) for w in weights]


def _weights_changed(before, after, atol=0.0):
    """Returns True if at least one weight tensor differs."""
    for b, a in zip(before, after):
        if not np.allclose(b, a, atol=atol):
            return True
    return False


# ---------------------------------------------------------------------------
# 2D — non-overlap
# ---------------------------------------------------------------------------


def _build_2d_model(H, W, C, pH, pW, padding, strides=None):
    inputs = keras.Input(shape=(H, W, C))
    x = keras.layers.Dense(C, name="pre")(inputs)
    grid_h = H // pH if padding == "valid" else (H + pH - 1) // pH
    grid_w = W // pW if padding == "valid" else (W + pW - 1) // pW
    if strides is not None:
        grid_h = (H - pH) // strides[0] + 1
        grid_w = (W - pW) // strides[1] + 1
    flat = pH * pW * C
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(
            t, size=(pH, pW), strides=strides, padding=padding,
        ),
        output_shape=(grid_h, grid_w, flat),
    )(x)
    recon = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W),
        strides=strides, padding=padding,
    )(patches)
    output = keras.layers.Dense(C, name="post")(recon)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model


def test_2d_loss_changes_over_epochs():
    """Loss must change between epochs if gradients flow."""
    H, W, C = 16, 16, 3
    model = _build_2d_model(H, W, C, 4, 4, "valid")
    rng = np.random.RandomState(0)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    assert losses[0] != losses[1], (
        f"Loss did not change between epochs (got {losses}); "
        f"gradients may not be flowing through the layer."
    )


def test_2d_weights_update_after_one_step():
    """Pre-layer and post-layer Dense weights must change after one step."""
    H, W, C = 16, 16, 3
    model = _build_2d_model(H, W, C, 4, 4, "valid")
    rng = np.random.RandomState(1)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    before = _snapshot(model.trainable_weights)
    model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)
    after = _snapshot(model.trainable_weights)
    assert _weights_changed(before, after), (
        "No trainable weight changed after one fit step; "
        "gradients are not reaching upstream weights."
    )


def test_2d_same_padding_loss_changes():
    """Same as test_2d_loss_changes_over_epochs but with same padding."""
    H, W, C = 17, 19, 3   # non-divisible to force same-path crop
    model = _build_2d_model(H, W, C, 4, 4, "same")
    rng = np.random.RandomState(2)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    assert losses[0] != losses[1]


# ---------------------------------------------------------------------------
# 2D — overlap (conv_transpose path)
# ---------------------------------------------------------------------------


def test_2d_overlap_loss_changes():
    """Gradients must flow through the conv-transpose overlap path."""
    H, W, C = 16, 16, 3
    model = _build_2d_model(H, W, C, 4, 4, "valid", strides=(2, 2))
    rng = np.random.RandomState(3)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[1]


def test_2d_overlap_weights_update():
    H, W, C = 16, 16, 3
    model = _build_2d_model(H, W, C, 4, 4, "valid", strides=(2, 2))
    rng = np.random.RandomState(4)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    before = _snapshot(model.trainable_weights)
    model.fit(x_train, y_train, epochs=1, batch_size=2, verbose=0)
    after = _snapshot(model.trainable_weights)
    assert _weights_changed(before, after)


# ---------------------------------------------------------------------------
# 3D — non-overlap
# ---------------------------------------------------------------------------


def _build_3d_model(D, H, W, C, pD, pH, pW, padding, strides=None):
    inputs = keras.Input(shape=(D, H, W, C))
    x = keras.layers.Dense(C, name="pre")(inputs)
    if strides is None:
        grid_d = D // pD if padding == "valid" else (D + pD - 1) // pD
        grid_h = H // pH if padding == "valid" else (H + pH - 1) // pH
        grid_w = W // pW if padding == "valid" else (W + pW - 1) // pW
    else:
        grid_d = (D - pD) // strides[0] + 1
        grid_h = (H - pH) // strides[1] + 1
        grid_w = (W - pW) // strides[2] + 1
    flat = pD * pH * pW * C
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(
            t, size=(pD, pH, pW), strides=strides, padding=padding,
        ),
        output_shape=(grid_d, grid_h, grid_w, flat),
    )(x)
    recon = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W),
        strides=strides, padding=padding,
    )(patches)
    output = keras.layers.Dense(C, name="post")(recon)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model


def test_3d_loss_changes_over_epochs():
    D, H, W, C = 8, 16, 16, 2
    model = _build_3d_model(D, H, W, C, 4, 4, 4, "valid")
    rng = np.random.RandomState(5)
    x_train = rng.rand(2, D, H, W, C).astype("float32")
    y_train = rng.rand(2, D, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[1]


def test_3d_weights_update_after_one_step():
    D, H, W, C = 8, 16, 16, 2
    model = _build_3d_model(D, H, W, C, 4, 4, 4, "valid")
    rng = np.random.RandomState(6)
    x_train = rng.rand(2, D, H, W, C).astype("float32")
    y_train = rng.rand(2, D, H, W, C).astype("float32")
    before = _snapshot(model.trainable_weights)
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
    after = _snapshot(model.trainable_weights)
    assert _weights_changed(before, after)


def test_3d_same_padding_loss_changes():
    D, H, W, C = 9, 17, 19, 2  # non-divisible for same path
    model = _build_3d_model(D, H, W, C, 4, 4, 4, "same")
    rng = np.random.RandomState(7)
    x_train = rng.rand(2, D, H, W, C).astype("float32")
    y_train = rng.rand(2, D, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[1]


def test_3d_overlap_loss_changes():
    D, H, W, C = 8, 8, 8, 2
    model = _build_3d_model(D, H, W, C, 4, 4, 4, "valid", strides=(2, 2, 2))
    rng = np.random.RandomState(8)
    x_train = rng.rand(2, D, H, W, C).astype("float32")
    y_train = rng.rand(2, D, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[1]


# ---------------------------------------------------------------------------
# Channels_first (skipped on tensorflow-cpu — see test_channels_first.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    keras.backend.backend() == "tensorflow",
    reason="extract_patches with channels_first requires NCHW conv which "
           "is unavailable on tensorflow-cpu.",
)
def test_2d_channels_first_loss_changes():
    H, W, C = 16, 16, 3
    inputs = keras.Input(shape=(C, H, W))
    x = keras.layers.Permute((2, 3, 1))(inputs)  # to channels_last for Dense
    x = keras.layers.Dense(C, name="pre")(x)
    x = keras.layers.Permute((3, 1, 2))(x)        # back to channels_first
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(
            t, size=(4, 4), padding="valid", data_format="channels_first",
        ),
        output_shape=(4 * 4 * C, 4, 4),
    )(x)
    recon = ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
        data_format="channels_first",
    )(patches)
    recon = keras.layers.Permute((2, 3, 1))(recon)
    output = keras.layers.Dense(C, name="post")(recon)
    output = keras.layers.Permute((3, 1, 2))(output)
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")

    rng = np.random.RandomState(9)
    x_train = rng.rand(4, C, H, W).astype("float32")
    y_train = rng.rand(4, C, H, W).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
    assert losses[0] != losses[1]
