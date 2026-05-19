"""End-to-end integration test: realistic patch-bottleneck autoencoder.

Mirrors the architectural pattern in AttemptingHeadNetwork.py (encoder →
extract_patches → dense bottleneck → reconstruct → decoder) but with
static input shapes, since our current layer requires static output_size.

(For drop-in replacement of the user's literal `return_model()` function
with its `Input(shape=[None,None,None,1])` declaration, the layer needs
a dynamic-output_size mode — tracked as a separate task.)

What this test proves:
1. The layer composes correctly with extract_patches and other layers
   in a realistic model graph.
2. The model trains end-to-end: loss is finite and decreases over
   multiple epochs (so gradients flow through the full encoder, the
   bottleneck, the reconstruction, and the decoder).
3. The trained model reproduces its input reasonably well (autoencoder
   reconstruction quality, not just numeric trickery).
"""

import keras
import numpy as np
import pytest
from keras import ops

from reconstruct_patches import ReconstructPatches3D


# Top-level function (not a closure) so Keras can deserialize it cleanly.
@keras.saving.register_keras_serializable(package="test_integration")
def _extract_4x4x4_valid(t):
    return ops.image.extract_patches(t, size=(4, 4, 4), padding="valid")


def _build_autoencoder(D, H, W, C, conv_filters, patch_size, bottleneck_factor):
    """Mirrors AttemptingHeadNetwork's auto-encode + transformer + dense path.

    Architecture:
      Input
        -> Conv3D (encoder)
        -> extract_patches
        -> Reshape to (B, num_patches, flat_per_patch)
        -> Dense compress
        -> Dense expand
        -> Reshape back to grid
        -> ReconstructPatches3D
        -> Conv3D (decoder)
        -> Output
    """
    inputs = keras.Input(shape=(D, H, W, C))

    # Encoder
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="enc_conv1")(inputs)
    x = keras.layers.BatchNormalization(name="enc_bn1")(x)
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="enc_conv2")(x)

    # Patch extraction (use Lambda since ExtractPatches Layer wrapper is a
    # future task). The function passed is a top-level registered
    # @keras_serializable so save/load round-trips cleanly.
    pD, pH, pW = patch_size
    assert patch_size == (4, 4, 4), (
        "test fixture _extract_4x4x4_valid hard-codes patch_size; "
        "update both if you change."
    )
    grid_d = D // pD
    grid_h = H // pH
    grid_w = W // pW
    flat = pD * pH * pW * conv_filters
    patches = keras.layers.Lambda(
        _extract_4x4x4_valid,
        output_shape=(grid_d, grid_h, grid_w, flat),
        name="extract",
    )(x)

    # Dense bottleneck on flattened patches
    num_patches = grid_d * grid_h * grid_w
    patches_flat = keras.layers.Reshape((num_patches, flat),
                                        name="patches_to_seq")(patches)
    bottleneck_dim = max(1, flat // bottleneck_factor)
    bn = keras.layers.Dense(bottleneck_dim, activation="elu",
                            name="bottleneck")(patches_flat)
    expanded = keras.layers.Dense(flat, activation="elu",
                                  name="expand")(bn)
    expanded_grid = keras.layers.Reshape((grid_d, grid_h, grid_w, flat),
                                         name="seq_to_grid")(expanded)

    # Reconstruction
    recon = ReconstructPatches3D(
        size=patch_size, output_size=(D, H, W), padding="valid",
        name="reconstruct",
    )(expanded_grid)

    # Decoder
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="dec_conv1")(recon)
    x = keras.layers.BatchNormalization(name="dec_bn1")(x)
    output = keras.layers.Conv3D(C, 3, padding="same", activation="sigmoid",
                                 name="dec_out")(x)

    return keras.Model(inputs, output, name="patch_autoencoder")


def test_autoencoder_trains_and_loss_decreases():
    """Train a small autoencoder for several epochs; loss must decrease."""
    D, H, W, C = 8, 16, 16, 1
    model = _build_autoencoder(
        D=D, H=H, W=W, C=C,
        conv_filters=4,
        patch_size=(4, 4, 4),
        bottleneck_factor=2,
    )
    model.compile(optimizer="adam", loss="mse")

    # Random "training" data — the autoencoder learns to compress and rebuild it.
    rng = np.random.RandomState(0)
    x_train = rng.rand(8, D, H, W, C).astype("float32")
    history = model.fit(x_train, x_train, epochs=5, batch_size=2, verbose=0)
    losses = history.history["loss"]
    assert all(np.isfinite(l) for l in losses), f"Non-finite loss: {losses}"
    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 5 epochs: {losses}. "
        f"Gradients may not be flowing correctly through the full "
        f"encode → patch → reconstruct → decode pipeline."
    )


def test_autoencoder_reconstructs_input():
    """After training, the autoencoder should reproduce its input above chance."""
    D, H, W, C = 8, 16, 16, 1
    model = _build_autoencoder(
        D=D, H=H, W=W, C=C,
        conv_filters=4,
        patch_size=(4, 4, 4),
        bottleneck_factor=2,
    )
    model.compile(optimizer="adam", loss="mse")

    rng = np.random.RandomState(1)
    x_train = rng.rand(8, D, H, W, C).astype("float32")

    # Untrained baseline
    pre_loss = float(model.evaluate(x_train, x_train, verbose=0))
    model.fit(x_train, x_train, epochs=10, batch_size=2, verbose=0)
    post_loss = float(model.evaluate(x_train, x_train, verbose=0))

    assert np.isfinite(pre_loss) and np.isfinite(post_loss)
    # With purely random uniform [0,1] data the autoencoder can't learn much
    # structure — variance of random data is ~0.083, so the floor is high.
    # We assert a 5% improvement over untrained baseline, which is enough to
    # show the model is actually learning (gradients reach trainable weights)
    # but loose enough to not be flaky.
    assert post_loss < pre_loss * 0.95, (
        f"Trained loss ({post_loss:.4f}) did not improve over untrained "
        f"baseline ({pre_loss:.4f}). The autoencoder may not be learning "
        f"through our layer."
    )


def test_autoencoder_save_load_after_training():
    """A trained autoencoder containing our layer must round-trip through save/load."""
    import os
    import tempfile

    D, H, W, C = 8, 16, 16, 1
    model = _build_autoencoder(
        D=D, H=H, W=W, C=C,
        conv_filters=4,
        patch_size=(4, 4, 4),
        bottleneck_factor=2,
    )
    model.compile(optimizer="adam", loss="mse")

    rng = np.random.RandomState(2)
    x_train = rng.rand(8, D, H, W, C).astype("float32")
    model.fit(x_train, x_train, epochs=2, batch_size=4, verbose=0)

    out_before = ops.convert_to_numpy(model(x_train[:1]))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trained_autoencoder.keras")
        model.save(path)
        # _extract_4x4x4_valid is @register_keras_serializable, so no
        # custom_objects= or safe_mode= needed.
        loaded = keras.models.load_model(path)

    out_after = ops.convert_to_numpy(loaded(x_train[:1]))
    np.testing.assert_allclose(out_before, out_after, atol=1e-6)
