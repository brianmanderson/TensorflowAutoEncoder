"""Comprehensive 2D extract -> reconstruct roundtrip evaluation.

Strategy: for every patch size in a curated list, exercise both the `"valid"`
path (input is an exact multiple of patch) and the `"same"` path (input is
non-divisible, forcing asymmetric crop on reconstruction) across multiple
residue classes. Then sweep channels and batch sizes independently.
"""

import numpy as np
import pytest
from keras import ops

from reconstruct_patches import ReconstructPatches2D, reconstruct_patches


def gradient_image(H, W, C=1, batch=1, dtype="float32"):
    """Smoothly varying 4D tensor — any reshape/crop bug becomes visible."""
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype(dtype)
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


# ---------------------------------------------------------------------------
# Test case grid
# ---------------------------------------------------------------------------

PATCH_SIZES_2D = [
    # symmetric
    (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (8, 8), (16, 16), (32, 32),
    # asymmetric
    (3, 5), (4, 7), (5, 3), (2, 8),
    # degenerate (one axis = 1)
    (1, 4), (4, 1),
]


def _gen_roundtrip_cases():
    """Yield (H, W, pH, pW, padding) covering valid and same paths.

    For valid: exact multiples (2x, 3x, 5x patch).
    For same: a few distinct residue classes per axis (1, ~half, patch-1).
    """
    cases = []
    for pH, pW in PATCH_SIZES_2D:
        # Valid: a few multiples
        for mH, mW in [(2, 2), (3, 4), (5, 3)]:
            cases.append((pH * mH, pW * mW, pH, pW, "valid"))
        # Same: residues
        offsets_h = sorted(set(o for o in (1, pH // 2, pH - 1) if o > 0))
        offsets_w = sorted(set(o for o in (1, pW // 2, pW - 1) if o > 0))
        for oh in offsets_h:
            for ow in offsets_w:
                cases.append((pH * 2 + oh, pW * 2 + ow, pH, pW, "same"))
    return cases


ROUNDTRIP_CASES_2D = _gen_roundtrip_cases()


def _id_2d(case):
    H, W, pH, pW, padding = case
    return f"H{H}_W{W}_p{pH}x{pW}_{padding}"


# ---------------------------------------------------------------------------
# Main roundtrip sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ROUNDTRIP_CASES_2D, ids=_id_2d)
def test_2d_roundtrip(case):
    H, W, pH, pW, padding = case
    x = gradient_image(H, W, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(pH, pW), padding=padding)
    recon = reconstruct_patches(
        patches, size=(pH, pW), output_size=(H, W), padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape, (
        f"Shape mismatch: in {x.shape}, out {recon_np.shape} "
        f"(H={H}, W={W}, patch=({pH},{pW}), {padding})"
    )
    np.testing.assert_allclose(
        recon_np, x, rtol=1e-6, atol=1e-6,
        err_msg=(
            f"Roundtrip failed for H={H}, W={W}, "
            f"patch=({pH},{pW}), {padding}"
        ),
    )


# ---------------------------------------------------------------------------
# Channels sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [1, 2, 3, 4, 8, 16, 32, 64])
def test_2d_channels(C):
    x = gradient_image(16, 16, C=C, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Batch sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_2d_batch(batch):
    x = gradient_image(12, 12, C=3, batch=batch)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(3, 3), padding="valid")
    recon = reconstruct_patches(
        patches, size=(3, 3), output_size=(12, 12), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Unbatched (rank-3 input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "H,W,padding", [
        (15, 15, "valid"),   # divisible by 5
        (15, 17, "same"),    # non-divisible: triggers pad-and-crop path
    ],
)
def test_2d_unbatched(H, W, padding):
    x = gradient_image(H, W, C=3, batch=1)[0]  # rank-3 input
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(5, 5), padding=padding)
    recon = reconstruct_patches(
        patches, size=(5, 5), output_size=(H, W), padding=padding,
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism: same input -> same output across calls
# ---------------------------------------------------------------------------


def test_2d_determinism():
    x = gradient_image(32, 32, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="same")
    r1 = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(32, 32), padding="same",
    ))
    r2 = ops.convert_to_numpy(reconstruct_patches(
        patches, size=(4, 4), output_size=(32, 32), padding="same",
    ))
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Layer == op equivalence (parametrized over a subset of cases)
# ---------------------------------------------------------------------------


LAYER_PARITY_CASES_2D = ROUNDTRIP_CASES_2D[::5]  # every 5th case


@pytest.mark.parametrize("case", LAYER_PARITY_CASES_2D, ids=_id_2d)
def test_2d_layer_matches_op(case):
    H, W, pH, pW, padding = case
    x = gradient_image(H, W, C=3, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(pH, pW), padding=padding)
    recon_op = reconstruct_patches(
        patches, size=(pH, pW), output_size=(H, W), padding=padding,
    )
    layer = ReconstructPatches2D(
        size=(pH, pW), output_size=(H, W), padding=padding,
    )
    recon_layer = layer(patches)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(recon_op),
        ops.convert_to_numpy(recon_layer),
    )
