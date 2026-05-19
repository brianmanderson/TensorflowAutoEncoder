"""Mixed-precision policy tests.

Verifies the layer behaves correctly under `keras.mixed_precision` global
policies. Our layer has no trainable weights, so most of the policy logic
doesn't apply, but the identity-kernel allocation in the overlap path and
the dtype handling in casts/divisions need to interact cleanly with
autocasting.
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import (
    ReconstructPatches2D,
    ReconstructPatches3D,
    reconstruct_patches,
)


@pytest.fixture(autouse=True)
def reset_policy():
    """Reset global precision policy after each test."""
    original = keras.mixed_precision.global_policy()
    yield
    keras.mixed_precision.set_global_policy(original)


def test_2d_under_mixed_float16_nonoverlap():
    keras.mixed_precision.set_global_policy("mixed_float16")
    x = np.random.RandomState(0).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16), padding="valid",
    )
    recon = layer(patches)
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape
    assert np.isfinite(recon_np).all(), "Mixed-precision output has non-finite values"
    # Roundtrip should still be reasonable (lower precision than float32).
    np.testing.assert_allclose(recon_np, x, rtol=1e-2, atol=1e-2)


def test_2d_under_mixed_float16_overlap():
    keras.mixed_precision.set_global_policy("mixed_float16")
    x = np.random.RandomState(1).rand(1, 16, 16, 3).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), strides=2, padding="valid")
    layer = ReconstructPatches2D(
        size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    recon = layer(patches)
    recon_np = ops.convert_to_numpy(recon)
    assert np.isfinite(recon_np).all()


def test_3d_under_mixed_float16():
    keras.mixed_precision.set_global_policy("mixed_float16")
    x = np.random.RandomState(2).rand(1, 8, 8, 8, 2).astype("float32")
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(2, 2, 2), padding="valid")
    layer = ReconstructPatches3D(
        size=(2, 2, 2), output_size=(8, 8, 8), padding="valid",
    )
    recon = layer(patches)
    recon_np = ops.convert_to_numpy(recon)
    assert np.isfinite(recon_np).all()


def test_layer_in_mixed_precision_model_trains():
    """A model containing our layer should train under mixed_float16."""
    keras.mixed_precision.set_global_policy("mixed_float16")
    H, W, C = 16, 16, 3
    inputs = keras.Input(shape=(H, W, C))
    x = keras.layers.Dense(C)(inputs)
    grid_h = H // 4
    grid_w = W // 4
    flat = 4 * 4 * C
    patches = keras.layers.Lambda(
        lambda t: ops.image.extract_patches(t, size=(4, 4), padding="valid"),
        output_shape=(grid_h, grid_w, flat),
    )(x)
    recon = ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
    )(patches)
    output = keras.layers.Dense(C, dtype="float32")(recon)  # final layer fp32
    model = keras.Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    rng = np.random.RandomState(3)
    x_train = rng.rand(4, H, W, C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    assert losses[0] != losses[1]
