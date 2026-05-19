"""Dtype preservation tests.

The non-overlap path is pure reshape/transpose/slice — should preserve any
dtype the backend supports. The overlap path goes through `conv_transpose`
+ division, which restricts dtype to backend conv-supported types
(typically float32, float64, sometimes float16/bfloat16; not int).
"""

import numpy as np
import pytest
from keras import ops

from reconstruct_patches import reconstruct_patches, reconstruct_patches_3d


def _gradient_image(H, W, C=3, batch=2, dtype="float32"):
    h = np.linspace(0, 1, H)
    w = np.linspace(0, 1, W)
    c = np.linspace(0, 1, C) if C > 1 else np.array([0.5])
    Hg, Wg, Cg = np.meshgrid(h, w, c, indexing="ij")
    img = ((2 * Hg + 3 * Wg + 4 * Cg) / 9.0).astype(dtype)
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


def _integer_image(H, W, C=3, batch=2, dtype="int32"):
    """Integer-valued image — each pixel has a unique int."""
    img = np.arange(H * W * C, dtype=dtype).reshape(H, W, C)
    return np.broadcast_to(img[None, ...], (batch, H, W, C)).copy()


# ---------------------------------------------------------------------------
# Float dtypes — non-overlap path (reshape/transpose/slice)
# ---------------------------------------------------------------------------

FLOAT_DTYPES_NONOVERLAP = ["float16", "float32", "float64"]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES_NONOVERLAP)
def test_2d_nonoverlap_dtype_preserved(dtype):
    x = _gradient_image(16, 16, C=3, batch=2, dtype=dtype)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    assert dtype in str(recon.dtype), (
        f"Expected dtype to contain {dtype}, got {recon.dtype}"
    )
    # Looser tolerance for float16
    atol = {"float16": 1e-3, "float32": 1e-6, "float64": 1e-12}[dtype]
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=atol)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES_NONOVERLAP)
def test_3d_nonoverlap_dtype_preserved(dtype):
    x = np.zeros((1, 8, 8, 8, 2), dtype=dtype)
    # Gradient by index
    for d in range(8):
        for h in range(8):
            for w in range(8):
                for c in range(2):
                    x[0, d, h, w, c] = (d + 2 * h + 3 * w + 4 * c) / 100.0
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(x_t, size=(2, 2, 2), padding="valid")
    recon = reconstruct_patches_3d(
        patches, size=(2, 2, 2), output_size=(8, 8, 8), padding="valid",
    )
    assert dtype in str(recon.dtype)
    atol = {"float16": 1e-3, "float32": 1e-6, "float64": 1e-12}[dtype]
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=atol)


# ---------------------------------------------------------------------------
# bfloat16 — optional; skip if backend doesn't support
# ---------------------------------------------------------------------------

def test_2d_nonoverlap_bfloat16():
    try:
        x_np = _gradient_image(16, 16, C=3, batch=2, dtype="float32")
        x_t = ops.cast(ops.convert_to_tensor(x_np), "bfloat16")
    except Exception:
        pytest.skip("bfloat16 not supported on this backend")
    patches = ops.image.extract_patches(x_t, size=(4, 4), padding="valid")
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16), padding="valid",
    )
    assert "bfloat16" in str(recon.dtype)
    # bfloat16 has ~2-3 decimal digits of precision
    np.testing.assert_allclose(
        ops.convert_to_numpy(ops.cast(recon, "float32")), x_np, atol=1e-2,
    )


# ---------------------------------------------------------------------------
# Integer dtypes — non-overlap path only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_2d_nonoverlap_integer_dtype(dtype):
    x = _integer_image(8, 8, C=2, batch=1, dtype=dtype)
    x_t = ops.convert_to_tensor(x)
    # extract_patches may upcast int → float internally because it uses
    # `conv` with a float kernel. Cast back at the end for the comparison.
    try:
        patches = ops.image.extract_patches(x_t, size=(2, 2), padding="valid")
    except Exception:
        pytest.skip(
            "extract_patches doesn't support this integer dtype on this backend"
        )
    # If the forward upcast, the inverse output will be float; cast for compare.
    recon = reconstruct_patches(
        patches, size=(2, 2), output_size=(8, 8), padding="valid",
    )
    recon_np = ops.convert_to_numpy(recon)
    # Cast both to int for value comparison (extract_patches preserves values
    # whether it upcasts or not).
    np.testing.assert_array_equal(recon_np.astype(dtype), x)


# ---------------------------------------------------------------------------
# Float dtypes — overlap path (conv_transpose + division)
# ---------------------------------------------------------------------------

FLOAT_DTYPES_OVERLAP = ["float32", "float64"]


@pytest.mark.parametrize("dtype", FLOAT_DTYPES_OVERLAP)
def test_2d_overlap_dtype_preserved(dtype):
    x = _gradient_image(16, 16, C=3, batch=2, dtype=dtype)
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4), strides=2, padding="valid",
    )
    recon = reconstruct_patches(
        patches, size=(4, 4), output_size=(16, 16),
        strides=(2, 2), padding="valid",
    )
    assert dtype in str(recon.dtype)
    atol = {"float32": 1e-5, "float64": 1e-10}[dtype]
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=atol)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES_OVERLAP)
def test_3d_overlap_dtype_preserved(dtype):
    x = np.zeros((1, 8, 8, 8, 2), dtype=dtype)
    for d in range(8):
        for h in range(8):
            for w in range(8):
                for c in range(2):
                    x[0, d, h, w, c] = (d + 2 * h + 3 * w + 4 * c) / 100.0
    x_t = ops.convert_to_tensor(x)
    patches = ops.image.extract_patches(
        x_t, size=(4, 4, 4), strides=2, padding="valid",
    )
    recon = reconstruct_patches_3d(
        patches, size=(4, 4, 4), output_size=(8, 8, 8),
        strides=(2, 2, 2), padding="valid",
    )
    assert dtype in str(recon.dtype)
    atol = {"float32": 1e-5, "float64": 1e-10}[dtype]
    np.testing.assert_allclose(ops.convert_to_numpy(recon), x, atol=atol)
