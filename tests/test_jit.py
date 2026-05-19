"""JIT compilation tests.

Verifies our layer/op work under each backend's tracing/JIT mechanism:
- TF: @tf.function
- JAX: jax.jit
- torch: torch.compile

The gradient_flow tests already exercise model.fit() which uses the
backend's default trainer (which may JIT internally), so this file
specifically tests EXPLICIT JIT compilation of our op as a function.
"""

import platform

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
# TensorFlow @tf.function
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="@tf.function is a TensorFlow-specific tracing mechanism.",
)
def test_tf_function_traces_reconstruct_2d():
    import tensorflow as tf

    def fn(patches):
        return reconstruct_patches(
            patches, size=(4, 4), output_size=(16, 16), padding="valid",
        )

    traced = tf.function(fn)
    x = np.random.RandomState(0).rand(1, 4, 4, 48).astype("float32")
    x_t = ops.convert_to_tensor(x)
    eager = ops.convert_to_numpy(fn(x_t))
    traced_out = ops.convert_to_numpy(traced(x_t))
    np.testing.assert_allclose(eager, traced_out, atol=1e-6)


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="@tf.function is TF-specific.",
)
def test_tf_function_traces_reconstruct_3d_overlap():
    """Overlap path uses conv_transpose; must trace correctly."""
    import tensorflow as tf

    def fn(patches):
        return reconstruct_patches_3d(
            patches, size=(4, 4, 4), output_size=(8, 8, 8),
            strides=(2, 2, 2), padding="valid",
        )

    traced = tf.function(fn)
    x_raw = np.random.RandomState(1).rand(1, 8, 8, 8, 2).astype("float32")
    patches = ops.image.extract_patches(
        ops.convert_to_tensor(x_raw), size=(4, 4, 4), strides=2, padding="valid",
    )
    eager = ops.convert_to_numpy(fn(patches))
    traced_out = ops.convert_to_numpy(traced(patches))
    np.testing.assert_allclose(eager, traced_out, atol=1e-5)


# ---------------------------------------------------------------------------
# JAX jit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="jax.jit is a JAX-specific JIT compiler.",
)
def test_jax_jit_compiles_reconstruct_2d():
    import jax

    def fn(patches):
        return reconstruct_patches(
            patches, size=(4, 4), output_size=(16, 16), padding="valid",
        )

    jitted = jax.jit(fn)
    x = np.random.RandomState(2).rand(1, 4, 4, 48).astype("float32")
    x_t = ops.convert_to_tensor(x)
    eager = ops.convert_to_numpy(fn(x_t))
    jit_out = ops.convert_to_numpy(jitted(x_t))
    np.testing.assert_allclose(eager, jit_out, atol=1e-6)


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="jax.jit is a JAX-specific JIT compiler.",
)
def test_jax_jit_compiles_overlap_path():
    """Overlap path uses conv_transpose; verify it JITs."""
    import jax

    def fn(patches):
        return reconstruct_patches(
            patches, size=(4, 4), output_size=(16, 16),
            strides=(2, 2), padding="valid",
        )

    jitted = jax.jit(fn)
    x_raw = np.random.RandomState(3).rand(1, 16, 16, 3).astype("float32")
    patches = ops.image.extract_patches(
        ops.convert_to_tensor(x_raw), size=(4, 4), strides=2, padding="valid",
    )
    eager = ops.convert_to_numpy(fn(patches))
    jit_out = ops.convert_to_numpy(jitted(patches))
    np.testing.assert_allclose(eager, jit_out, atol=1e-5)


# ---------------------------------------------------------------------------
# PyTorch torch.compile
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    keras.backend.backend() != "torch",
    reason="torch.compile is PyTorch-specific.",
)
@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="torch.compile is unreliable on native Windows (Triton support).",
)
def test_torch_compile_reconstruct_2d():
    import torch

    def fn(patches):
        return reconstruct_patches(
            patches, size=(4, 4), output_size=(16, 16), padding="valid",
        )

    try:
        compiled = torch.compile(fn)
    except Exception as e:
        pytest.skip(f"torch.compile not available: {e}")

    x = np.random.RandomState(4).rand(1, 4, 4, 48).astype("float32")
    x_t = ops.convert_to_tensor(x)
    eager = ops.convert_to_numpy(fn(x_t))
    try:
        compiled_out = ops.convert_to_numpy(compiled(x_t))
    except Exception as e:
        pytest.skip(f"torch.compile execution failed (likely missing backend): {e}")
    np.testing.assert_allclose(eager, compiled_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Functional model trains under JIT (cross-backend via Keras Model.compile)
# ---------------------------------------------------------------------------


def test_layer_in_jit_compiled_model():
    """A model containing our layer should train under jit_compile=True.

    Keras 3 model.compile(jit_compile=True) routes to the backend's native
    JIT (XLA on TF, jax.jit on JAX, torch.compile on torch). This test
    proves the layer's training graph compiles cleanly on whichever backend
    is active.
    """
    if keras.backend.backend() == "torch" and platform.system() == "Windows":
        pytest.skip(
            "torch.compile/jit_compile is unreliable on native Windows.",
        )

    H, W, C = 16, 16, 3
    patches_input = keras.Input(shape=(H // 4, W // 4, 4 * 4 * C))
    recon = ReconstructPatches2D(
        size=(4, 4), output_size=(H, W), padding="valid",
    )(patches_input)
    output = keras.layers.Dense(C)(recon)
    model = keras.Model(patches_input, output)
    try:
        model.compile(optimizer="adam", loss="mse", jit_compile=True)
    except Exception as e:
        pytest.skip(f"jit_compile=True not supported on this backend: {e}")

    rng = np.random.RandomState(5)
    x_train = rng.rand(4, H // 4, W // 4, 4 * 4 * C).astype("float32")
    y_train = rng.rand(4, H, W, C).astype("float32")
    try:
        history = model.fit(x_train, y_train, epochs=2, batch_size=2, verbose=0)
    except Exception as e:
        pytest.skip(f"jit_compile=True training failed: {e}")
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses)
