"""Shared fixtures and feature-detection helpers for direct_tests.

These tests import `keras.layers.X` and `keras.ops.image.X` *directly* —
no dependency on tensorflowwork's local `reconstruct_patches.py`. That
way they validate whatever keras version is installed (or on PYTHONPATH
via the keras checkout).

For each feature added across PRs 1-7, a `has_*()` helper inspects the
installed keras to decide if the feature is present. Tests for features
that aren't present are skipped via `pytest.mark.skipif`. This lets the
same `direct_tests/` directory run against ANY keras version — from
upstream master (basic features only) to our full local mirror.
"""

from __future__ import annotations

import inspect

import keras
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------


def _layer_init_params():
    """Return the set of __init__ parameter names on ReconstructPatches2D,
    or empty set if the layer isn't installed."""
    if not hasattr(keras.layers, "ReconstructPatches2D"):
        return set()
    return set(
        inspect.signature(keras.layers.ReconstructPatches2D.__init__).parameters
    )


def has_reconstruct_layer():
    return hasattr(keras.layers, "ReconstructPatches2D") and hasattr(
        keras.layers, "ReconstructPatches3D"
    )


def has_reconstruct_op():
    return hasattr(keras.ops.image, "reconstruct_patches") and hasattr(
        keras.ops.image, "reconstruct_patches_3d"
    )


def has_reduction_param():
    """`reduction='mean'/'sum'` Layer kwarg — added in PR 2."""
    return "reduction" in _layer_init_params()


def has_overlap():
    """Strides < size accepted at construction (PR 2)."""
    if not has_reconstruct_layer():
        return False
    try:
        keras.layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16),
            strides=(2, 2), padding="valid",
        )
        return True
    except (NotImplementedError, ValueError):
        return False


def has_channels_first():
    """channels_first runs without raising NotImplementedError (PR 3).

    Detection is via call-time probe — the layer constructor accepts
    `data_format` from PR 1 already, so we can't gate on signature alone.
    """
    if not has_reconstruct_layer():
        return False
    try:
        layer = keras.layers.ReconstructPatches2D(
            size=(4, 4), output_size=(16, 16),
            padding="valid", data_format="channels_first",
        )
        patches = np.zeros((1, 48, 4, 4), dtype="float32")
        layer(keras.ops.convert_to_tensor(patches))
        return True
    except NotImplementedError:
        return False
    except Exception:
        # Any other error means the layer accepted channels_first; we just
        # mis-constructed inputs. That's a "has" signal.
        return True


def has_dilation_param():
    """`dilation_rate` Layer kwarg — added in PR 4."""
    return "dilation_rate" in _layer_init_params()


def has_auto_infer_output_size():
    """`output_size=None` accepted at construction (PR 5)."""
    if not has_reconstruct_layer():
        return False
    try:
        keras.layers.ReconstructPatches2D(size=(4, 4), padding="valid")
        return True
    except (TypeError, ValueError):
        return False


def has_extract_patches_layer():
    """`ExtractPatches{2,3}D` Layer classes — added in PR 6."""
    return hasattr(keras.layers, "ExtractPatches2D") and hasattr(
        keras.layers, "ExtractPatches3D"
    )


def has_dual_input():
    """Layer accepts `[patches, reference]` list input (PR 7).

    Detected by trying it and seeing if the layer routes to a list-handling
    code path (rather than treating the list as a single tensor).
    """
    if not has_reconstruct_layer():
        return False
    try:
        layer = keras.layers.ReconstructPatches2D(
            size=(4, 4), padding="same",
        )
    except (TypeError, ValueError):
        # Can't construct without output_size — auto-infer not supported,
        # so dual-input mode (which also defers output_size) also isn't.
        return False
    try:
        patches = keras.ops.convert_to_tensor(
            np.zeros((1, 4, 4, 48), dtype="float32")
        )
        ref = keras.ops.convert_to_tensor(
            np.zeros((1, 16, 16, 3), dtype="float32")
        )
        layer([patches, ref])
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Skip markers — module-level constants for reuse across test files
# ---------------------------------------------------------------------------


needs_reconstruct = pytest.mark.skipif(
    not has_reconstruct_layer(),
    reason="ReconstructPatches{2,3}D not present in installed keras",
)
needs_overlap = pytest.mark.skipif(
    not has_overlap(),
    reason="overlap (strides < size) not supported in installed keras",
)
needs_reduction = pytest.mark.skipif(
    not has_reduction_param(),
    reason="reduction= kwarg not in installed keras layer",
)
needs_channels_first = pytest.mark.skipif(
    not has_channels_first(),
    reason="channels_first not supported in installed keras (or TF-CPU)",
)
needs_dilation = pytest.mark.skipif(
    not has_dilation_param(),
    reason="dilation_rate= kwarg not in installed keras layer",
)
needs_auto_infer = pytest.mark.skipif(
    not has_auto_infer_output_size(),
    reason="output_size auto-infer not supported in installed keras",
)
needs_extract_layer = pytest.mark.skipif(
    not has_extract_patches_layer(),
    reason="ExtractPatches{2,3}D not present in installed keras",
)
needs_dual_input = pytest.mark.skipif(
    not has_dual_input(),
    reason="dual-input [patches, reference] mode not supported in installed keras",
)


# ---------------------------------------------------------------------------
# Helpers — deterministic test data
# ---------------------------------------------------------------------------


def gradient_image(H, W, C=1, batch=1, dtype="float32"):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype(dtype)
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


def gradient_volume(D, H, W, C=1, batch=1, dtype="float32"):
    d = np.linspace(0, 1, D)
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Dg, Hg, Wg, Cg = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((Dg + 2 * Hg + 3 * Wg + 4 * Cg) / 10.0).astype(dtype)
    return np.broadcast_to(vol[None, ...], (batch, D, H, W, C)).copy()
