"""Tutorial: building a patch-bottleneck autoencoder.

Walkthrough of the layer in a realistic small autoencoder. Mirrors the
architectural pattern in AttemptingHeadNetwork.py:

    Input -> encoder convs -> ExtractPatches3D -> Reshape -> Dense
        bottleneck -> Dense expand -> Reshape -> ReconstructPatches3D
        -> decoder convs -> Output

Run as a script:

    KERAS_BACKEND=tensorflow python examples/autoencoder_tutorial.py

Designed to be readable end-to-end rather than minimal: comments explain
each architectural choice and how the patch-bottleneck pattern works.
A notebook (.ipynb) version could be derived from this if useful.
"""

import os
import sys

# Make reconstruct_patches importable when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import numpy as np
from keras import ops

from reconstruct_patches import ExtractPatches3D, ReconstructPatches3D


# ---------------------------------------------------------------------------
# 1. Build the autoencoder
# ---------------------------------------------------------------------------

def build_autoencoder(D, H, W, C, conv_filters, patch_size, bottleneck_factor):
    """Patch-bottleneck 3D autoencoder.

    The key idea: instead of bottlenecking via a strided conv that loses
    spatial structure, we extract patches, compress each patch independently
    via a Dense layer (the bottleneck), expand back, and reconstruct. The
    layer that reconstructs is the inverse of extract_patches — proven to
    recover the original input exactly when no information is lost in the
    bottleneck.
    """
    inputs = keras.Input(shape=(D, H, W, C))

    # Encoder: two conv blocks to learn local features.
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="enc_conv1")(inputs)
    x = keras.layers.BatchNormalization(name="enc_bn1")(x)
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="enc_conv2")(x)

    # Patch extraction. With strides==size (default), patches are non-overlapping
    # and tile the spatial input exactly (assuming the input divides evenly).
    pD, pH, pW = patch_size
    patches = ExtractPatches3D(size=patch_size, padding="valid",
                               name="extract")(x)
    # patches shape: (B, D/pD, H/pH, W/pW, pD*pH*pW*conv_filters)

    # Sequence-ify the patches: (B, num_patches, flat) so we can apply a
    # Dense layer that compresses each patch independently.
    grid_d, grid_h, grid_w = D // pD, H // pH, W // pW
    num_patches = grid_d * grid_h * grid_w
    flat = pD * pH * pW * conv_filters
    patches_seq = keras.layers.Reshape((num_patches, flat),
                                       name="patches_to_seq")(patches)

    # Bottleneck: compress flat features by `bottleneck_factor`.
    bottleneck_dim = max(1, flat // bottleneck_factor)
    bn = keras.layers.Dense(bottleneck_dim, activation="elu",
                            name="bottleneck")(patches_seq)
    expanded = keras.layers.Dense(flat, activation="elu",
                                  name="expand")(bn)
    expanded_grid = keras.layers.Reshape((grid_d, grid_h, grid_w, flat),
                                         name="seq_to_grid")(expanded)

    # Reconstruct: place each compressed-then-expanded patch back into the
    # spatial canvas. Inverse of ExtractPatches3D.
    recon = ReconstructPatches3D(
        size=patch_size, output_size=(D, H, W), padding="valid",
        name="reconstruct",
    )(expanded_grid)

    # Decoder: refine the reconstructed feature map back to the input shape.
    x = keras.layers.Conv3D(conv_filters, 3, padding="same", activation="elu",
                            name="dec_conv1")(recon)
    x = keras.layers.BatchNormalization(name="dec_bn1")(x)
    output = keras.layers.Conv3D(C, 3, padding="same", activation="sigmoid",
                                 name="dec_out")(x)

    return keras.Model(inputs, output, name="patch_autoencoder")


# ---------------------------------------------------------------------------
# 2. Train it on synthetic volumetric data
# ---------------------------------------------------------------------------

def main():
    D, H, W, C = 8, 16, 16, 1
    model = build_autoencoder(
        D=D, H=H, W=W, C=C,
        conv_filters=8,
        patch_size=(4, 4, 4),
        bottleneck_factor=2,
    )
    model.compile(optimizer="adam", loss="mse")
    model.summary(line_length=110)

    # Synthetic training data: random uniform volumes. With purely random data
    # the autoencoder can't learn much structure (no spatial regularities to
    # compress), but loss will still decrease meaningfully if the layers and
    # gradients are wired correctly. For real use, replace with your dataset.
    rng = np.random.RandomState(0)
    x_train = rng.rand(16, D, H, W, C).astype("float32")
    x_val = rng.rand(4, D, H, W, C).astype("float32")

    print("\nTraining for 10 epochs...")
    history = model.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=10, batch_size=4, verbose=2,
    )

    # Inspect the bottleneck representation for a single example.
    print("\nInspecting bottleneck activations for the first validation sample:")
    bottleneck_model = keras.Model(model.input, model.get_layer("bottleneck").output)
    bn_activations = ops.convert_to_numpy(bottleneck_model(x_val[:1]))
    print(f"  shape: {bn_activations.shape}")
    print(f"  mean: {bn_activations.mean():.3f}, std: {bn_activations.std():.3f}")

    # Verify save/load round-trips a trained model containing our layers.
    print("\nVerifying model saves and loads correctly...")
    model.save("trained_autoencoder.keras")
    loaded = keras.models.load_model("trained_autoencoder.keras")
    before = ops.convert_to_numpy(model(x_val[:1]))
    after = ops.convert_to_numpy(loaded(x_val[:1]))
    np.testing.assert_allclose(before, after, atol=1e-6)
    print("  save/load round-trip: OK")
    os.remove("trained_autoencoder.keras")

    print("\nFinal validation loss:", history.history["val_loss"][-1])
    print("Done.")


if __name__ == "__main__":
    main()
