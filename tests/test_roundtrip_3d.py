"""Comprehensive 3D extract -> reconstruct roundtrip evaluation.

Same strategy as the 2D suite but tuned for 3D where each test allocates a
larger tensor. The case grid is smaller per-patch-size to keep wall-clock
under control; the channel and batch sweeps fill in the remaining coverage.
"""

import numpy as np
import pytest
from keras import ops

from reconstruct_patches import ReconstructPatches3D, reconstruct_patches_3d


def gradient_volume(D, H, W, C=1, batch=1, dtype="float32"):
    d = np.linspace(0, 1, D)
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Dg, Hg, Wg, Cg = np.meshgrid(d, h, w, c, indexing="ij")
    vol = ((Dg + 2 * Hg + 3 * Wg + 4 * Cg) / 10.0).astype(dtype)
    return np.broadcast_to(vol[None, ...], (batch, D, H, W, C)).copy()


PATCH_SIZES_3D = [
    # symmetric
    (2, 2, 2), (3, 3, 3), (4, 4, 4), (8, 8, 8), (16, 16, 16),
    # asymmetric — common medical-imaging shapes
    (4, 8, 8), (8, 16, 16),
    # Note: (16, 32, 32) deliberately omitted. The forward `extract_patches`
    # builds an identity kernel of shape (16, 32, 32, C, 16*32*32*C); with
    # C=2 that's a 32768-channel kernel taking ~4 GB float32, which OOMs
    # GitHub Actions runners (7 GB). The (8, 16, 16) entry above already
    # exercises the asymmetric-medical-imaging code path with a 64 MB kernel.
    # general asymmetric / prime
    (3, 5, 7), (2, 4, 8),
    # degenerate (depth = 1, single-slice case)
    (1, 4, 4), (1, 8, 8),
]


def _gen_roundtrip_cases_3d():
    cases = []
    for pD, pH, pW in PATCH_SIZES_3D:
        # Valid: a couple of multiples
        cases.append((pD * 2, pH * 2, pW * 2, pD, pH, pW, "valid"))
        cases.append((pD * 3, pH * 2, pW * 2, pD, pH, pW, "valid"))
        # Same: residues. Limit to a small spread to avoid combinatorial blowup.
        offsets_d = sorted(set(o for o in (1, pD - 1) if o > 0))
        offsets_h = sorted(set(o for o in (1, pH // 2, pH - 1) if o > 0))
        offsets_w = sorted(set(o for o in (1, pW // 2, pW - 1) if o > 0))
        # Diagonal sweep + a couple of off-diagonal mixes — full triple-product
        # would be ~30 per patch and too slow under 3D + 3 backends.
        for od, oh, ow in zip(offsets_d, offsets_h, offsets_w):
            cases.append(
                (pD * 2 + od, pH * 2 + oh, pW * 2 + ow, pD, pH, pW, "same")
            )
        # Add a "stress" residue: pad close to a full patch on every axis
        if pD > 1 and pH > 1 and pW > 1:
            cases.append(
                (
                    pD * 2 + (pD - 1),
                    pH * 2 + (pH - 1),
                    pW * 2 + (pW - 1),
                    pD, pH, pW, "same",
                )
            )
    return cases


ROUNDTRIP_CASES_3D = _gen_roundtrip_cases_3d()


def _id_3d(case):
    D, H, W, pD, pH, pW, padding = case
    return f"D{D}_H{H}_W{W}_p{pD}x{pH}x{pW}_{padding}"


# ---------------------------------------------------------------------------
# Main roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ROUNDTRIP_CASES_3D, ids=_id_3d)
def test_3d_roundtrip(case):
    D, H, W, pD, pH, pW, padding = case
    x = gradient_volume(D, H, W, C=2, batch=2)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(pD, pH, pW), padding=padding)
    recon = reconstruct_patches_3d(
        patches, size=(pD, pH, pW), output_size=(D, H, W), padding=padding,
    )
    recon_np = ops.convert_to_numpy(recon)
    assert recon_np.shape == x.shape, (
        f"Shape mismatch: in {x.shape}, out {recon_np.shape} "
        f"(D={D}, H={H}, W={W}, patch=({pD},{pH},{pW}), {padding})"
    )
    np.testing.assert_allclose(
        recon_np, x, rtol=1e-6, atol=1e-6,
        err_msg=(
            f"Roundtrip failed for D={D}, H={H}, W={W}, "
            f"patch=({pD},{pH},{pW}), {padding}"
        ),
    )


# ---------------------------------------------------------------------------
# Channels sweep (3D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [1, 2, 3, 4, 8])
def test_3d_channels(C):
    x = gradient_volume(8, 16, 16, C=C, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), padding="valid",
    )
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 16, 16), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Batch sweep (3D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_3d_batch(batch):
    x = gradient_volume(8, 8, 8, C=2, batch=batch)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(2, 2, 2), padding="valid",
    )
    recon = reconstruct_patches_3d(
        patches, size=(2, 2, 2), output_size=(8, 8, 8), padding="valid",
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Unbatched (rank-4 input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,H,W,padding", [
        (6, 10, 14, "valid"),  # divisible by (3, 5, 7)
        (7, 11, 13, "same"),   # non-divisible: triggers pad-and-crop path
    ],
)
def test_3d_unbatched(D, H, W, padding):
    x = gradient_volume(D, H, W, C=2, batch=1)[0]  # rank-4 input
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(3, 5, 7), padding=padding,
    )
    recon = reconstruct_patches_3d(
        patches, size=(3, 5, 7), output_size=(D, H, W), padding=padding,
    )
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_3d_determinism():
    x = gradient_volume(8, 16, 16, C=2, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(2, 4, 4), padding="same",
    )
    r1 = ops.convert_to_numpy(reconstruct_patches_3d(
        patches, size=(2, 4, 4), output_size=(8, 16, 16), padding="same",
    ))
    r2 = ops.convert_to_numpy(reconstruct_patches_3d(
        patches, size=(2, 4, 4), output_size=(8, 16, 16), padding="same",
    ))
    np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Layer == op equivalence
# ---------------------------------------------------------------------------


LAYER_PARITY_CASES_3D = ROUNDTRIP_CASES_3D[::3]  # every 3rd


@pytest.mark.parametrize("case", LAYER_PARITY_CASES_3D, ids=_id_3d)
def test_3d_layer_matches_op(case):
    D, H, W, pD, pH, pW, padding = case
    x = gradient_volume(D, H, W, C=2, batch=1)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(pD, pH, pW), padding=padding,
    )
    recon_op = reconstruct_patches_3d(
        patches, size=(pD, pH, pW), output_size=(D, H, W), padding=padding,
    )
    layer = ReconstructPatches3D(
        size=(pD, pH, pW), output_size=(D, H, W), padding=padding,
    )
    recon_layer = layer(patches)
    np.testing.assert_array_equal(
        ops.convert_to_numpy(recon_op),
        ops.convert_to_numpy(recon_layer),
    )
