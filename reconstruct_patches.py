"""Prototype: inverse of `keras.ops.image.extract_patches`.

Designed to live at `keras/src/ops/image.py` alongside `extract_patches` once
upstreamed. Multi-backend (TF/JAX/PyTorch) via `keras.ops` and `keras` `backend.*`
only — no `tf.*` calls.

Forward op output format (mirror exactly):
    3D channels_last: (B, gD, gH, gW, pD*pH*pW*C)
    2D channels_last: (B, gH, gW, pH*pW*C)
    channels_first variants put the flattened-patch dim first:
        3D: (B, pD*pH*pW*C, gD, gH, gW)
        2D: (B, pH*pW*C, gH, gW)
    Unbatched drops the leading B.

Inverse output:
    3D channels_last: (B, D, H, W, C)
    2D channels_last: (B, H, W, C)
    channels_first variants put C first.

Stride support:
    strides == size (default): fast reshape/transpose/slice path.
    strides < size on any axis (overlapping): conv-transpose path with
        averaging to recover the original from overlap sums.
    strides > size on any axis (gapped): rejected — information is lost.

Dilation support: dilation_rate != 1 is not yet supported (would be a
follow-up to handle dilated kernels in both paths).
"""

import keras
from keras import ops
from keras.layers import Layer

try:
    from keras.src import backend
    from keras.src.api_export import keras_export
except ImportError:
    from keras import backend
    def keras_export(*args, **kwargs):
        def _decorator(obj):
            return obj
        return _decorator


# ---------------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------------

def _normalize_strides(strides, size, fn_name):
    """Default strides to size; reject gapped (stride > size on any axis)."""
    if strides is None:
        return tuple(size)
    if isinstance(strides, int):
        strides = (strides,) * len(size)
    strides = tuple(strides)
    if len(strides) != len(size):
        raise ValueError(
            f"`strides` must have the same length as `size`. "
            f"Got strides={strides}, size={size}"
        )
    for s, k in zip(strides, size):
        if s > k:
            raise NotImplementedError(
                f"`{fn_name}` does not support gapped patches "
                f"(stride > size). Got strides={strides}, size={size}. "
                f"With stride > size, information between patches is lost "
                f"and cannot be recovered."
            )
        if s < 1:
            raise ValueError(
                f"`strides` entries must be >= 1. Got strides={strides}"
            )
    return strides


def _is_nonoverlapping(strides, size):
    return tuple(strides) == tuple(size)


def _infer_output_size_valid(patches, size, strides, data_format):
    """Compute output_size from patches shape for padding='valid'.

    For valid padding, the inverse is deterministic:
        output_size[i] = (grid[i] - 1) * stride[i] + size[i]

    `patches.shape` must have all grid dimensions statically known.
    """
    data_format = backend.standardize_data_format(data_format)
    is_3d = (len(size) == 3)
    rank = len(patches.shape)
    if data_format == "channels_last":
        # batched: (B, grid..., flat); unbatched: (grid..., flat)
        first_grid_axis = 0 if rank == len(size) + 1 else 1
        grid = patches.shape[first_grid_axis:first_grid_axis + len(size)]
    else:
        # batched: (B, flat, grid...); unbatched: (flat, grid...)
        first_grid_axis = 1 if rank == len(size) + 1 else 2
        grid = patches.shape[first_grid_axis:first_grid_axis + len(size)]
    if any(g is None for g in grid):
        raise ValueError(
            f"Cannot auto-infer output_size for valid padding: at least one "
            f"grid dimension is unknown. patches.shape={patches.shape}. "
            f"Pass output_size explicitly."
        )
    return tuple(
        (g - 1) * s + k for g, s, k in zip(grid, strides, size)
    )


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------

@keras_export("keras.ops.image.reconstruct_patches")
def reconstruct_patches(
    patches,
    size,
    output_size=None,
    strides=None,
    padding="valid",
    data_format=None,
    reduction="mean",
    dilation_rate=1,
):
    """Reconstructs image(s) or volume(s) from patches.

    Inverse of `keras.ops.image.extract_patches`. Supports both non-overlapping
    (`strides == size`) and overlapping (`strides < size`) cases. For
    overlapping patches, contributions are summed and (when
    `reduction="mean"`, the default) divided by per-pixel overlap count.
    With `reduction="mean"`, the original is exactly recovered when patches
    were extracted from a consistent input. With `reduction="sum"`, the
    output matches `torch.nn.Fold` semantics (raw sum, no averaging).

    Args:
        patches: Patches tensor as produced by `extract_patches`.
            For 2D patches: 3D `(gH, gW, pH*pW*C)` or 4D
                `(B, gH, gW, pH*pW*C)` (channels_last);
                `(B, pH*pW*C, gH, gW)` (channels_first batched).
            For 3D patches: 4D `(gD, gH, gW, pD*pH*pW*C)` or 5D
                `(B, gD, gH, gW, pD*pH*pW*C)` (channels_last);
                `(B, pD*pH*pW*C, gD, gH, gW)` (channels_first batched).
        size: Patch size, matching the `size` used for extraction.
            Length 2 tuple for 2D, length 3 tuple for 3D, or int.
        output_size: Spatial shape of the original image/volume before
            extraction. Length 2 tuple `(H, W)` for 2D, length 3 tuple
            `(D, H, W)` for 3D.
        strides: int or tuple. Must be <= `size` on every axis. Defaults
            to `size` (non-overlapping). When less than `size`, overlapping
            reconstruction is used.
        padding: `"same"` or `"valid"`, matching the extraction.
        data_format: `"channels_last"` or `"channels_first"`. Defaults to
            `keras.config.image_data_format()`.
        reduction: `"mean"` (default) or `"sum"`. How overlapping
            contributions are combined. `"mean"` recovers the original
            input; `"sum"` matches PyTorch `torch.nn.Fold` semantics.
            Ignored when patches do not overlap.

    Returns:
        Reconstructed image/volume, matching `patches`' batched-ness and
        `data_format`.

    Examples:

    >>> import numpy as np
    >>> import keras
    >>> from keras import ops
    >>> from reconstruct_patches import reconstruct_patches
    >>> image = np.random.rand(1, 16, 16, 3).astype("float32")
    >>> patches = ops.image.extract_patches(image, size=(4, 4), padding="valid")
    >>> tuple(patches.shape)
    (1, 4, 4, 48)
    >>> recon = reconstruct_patches(
    ...     patches, size=(4, 4), output_size=(16, 16), padding="valid",
    ... )
    >>> tuple(recon.shape)
    (1, 16, 16, 3)
    """
    if not isinstance(size, int):
        if not isinstance(size, (tuple, list)):
            raise TypeError(
                "Invalid `size` argument. Expected an int or a tuple. "
                f"Received: size={size} of type {type(size).__name__}"
            )
        if len(size) not in (2, 3):
            raise ValueError(
                "Invalid `size` argument. Expected a tuple of length 2 or 3. "
                f"Received: size={size} with length {len(size)}"
            )
    if reduction not in ("mean", "sum"):
        raise ValueError(
            f"`reduction` must be 'mean' or 'sum'. Received: {reduction}"
        )

    if not isinstance(size, int) and len(size) == 3:
        return _reconstruct_patches_3d(
            patches, size, output_size, strides, padding, data_format,
            reduction, dilation_rate,
        )
    return _reconstruct_patches_2d(
        patches, size, output_size, strides, padding, data_format,
        reduction, dilation_rate,
    )


@keras_export("keras.ops.image.reconstruct_patches_3d")
def reconstruct_patches_3d(
    patches,
    size,
    output_size=None,
    strides=None,
    padding="valid",
    data_format=None,
    reduction="mean",
    dilation_rate=1,
):
    """Reconstructs volume(s) from 3D patches. See `reconstruct_patches`.

    Examples:

    >>> import numpy as np
    >>> import keras
    >>> from keras import ops
    >>> from reconstruct_patches import reconstruct_patches_3d
    >>> volume = np.random.rand(1, 8, 16, 16, 2).astype("float32")
    >>> patches = ops.image.extract_patches_3d(volume, size=(4, 4, 4), padding="valid")
    >>> tuple(patches.shape)
    (1, 2, 4, 4, 128)
    >>> recon = reconstruct_patches_3d(
    ...     patches, size=(4, 4, 4), output_size=(8, 16, 16), padding="valid",
    ... )
    >>> tuple(recon.shape)
    (1, 8, 16, 16, 2)
    """
    if isinstance(size, int):
        size = (size, size, size)
    if reduction not in ("mean", "sum"):
        raise ValueError(
            f"`reduction` must be 'mean' or 'sum'. Received: {reduction}"
        )
    return _reconstruct_patches_3d(
        patches, size, output_size, strides, padding, data_format,
        reduction, dilation_rate,
    )


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def _reconstruct_patches_2d(
    patches, size, output_size, strides=None, padding="valid",
    data_format=None, reduction="mean", dilation_rate=1,
):
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError(
            "Invalid `size`. Expected length 2 for 2D reconstruction. "
            f"Got: size={size}"
        )
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    strides = _normalize_strides(strides, size, "reconstruct_patches")
    if output_size is None:
        if padding != "valid":
            raise ValueError(
                "`output_size=None` (auto-infer) is only supported for "
                "padding='valid'. For padding='same', the original size is "
                "ambiguous from patches alone — please pass output_size "
                "explicitly."
            )
        output_size = _infer_output_size_valid(patches, size, strides, data_format)
    if len(output_size) != 2:
        raise ValueError(
            "Invalid `output_size`. Expected length 2 (H, W). "
            f"Got: output_size={output_size}"
        )
    data_format = backend.standardize_data_format(data_format)

    # Channels_first -> transpose to channels_last, compute, transpose back.
    if data_format == "channels_first":
        if len(patches.shape) == 3:    # unbatched (flat, gH, gW)
            patches_cl = ops.transpose(patches, axes=(1, 2, 0))
        elif len(patches.shape) == 4:  # batched (B, flat, gH, gW)
            patches_cl = ops.transpose(patches, axes=(0, 2, 3, 1))
        else:
            raise ValueError(
                f"`patches` has unexpected rank for 2D channels_first: "
                f"got shape {patches.shape}"
            )
        result_cl = _reconstruct_patches_2d_cl(
            patches_cl, size, output_size, strides, padding, reduction, dilation_rate,
        )
        if len(patches.shape) == 3:
            return ops.transpose(result_cl, axes=(2, 0, 1))
        return ops.transpose(result_cl, axes=(0, 3, 1, 2))

    return _reconstruct_patches_2d_cl(
        patches, size, output_size, strides, padding, reduction, dilation_rate,
    )


def _reconstruct_patches_2d_cl(patches, size, output_size, strides, padding, reduction, dilation_rate=1):
    """Channels_last 2D core: dispatches between non-overlap and overlap paths."""
    _unbatched = (len(patches.shape) == 3)
    if _unbatched:
        patches = ops.expand_dims(patches, axis=0)

    if _is_nonoverlapping(strides, size) and dilation_rate == 1:
        # No overlap, no dilation: reduction doesn't matter (count is 1 everywhere)
        result = _reconstruct_2d_nonoverlap_cl(patches, size, output_size, padding)
    else:
        result = _reconstruct_2d_overlap_cl(
            patches, size, output_size, strides, padding, reduction, dilation_rate,
        )

    if _unbatched:
        result = ops.squeeze(result, axis=0)
    return result


def _reconstruct_2d_nonoverlap_cl(patches, size, output_size, padding):
    """Fast reshape/transpose/slice path for strides == size, channels_last."""
    pH, pW = size
    H, W = output_size

    shp = ops.shape(patches)
    B, gH, gW = shp[0], shp[1], shp[2]
    static_flat = patches.shape[-1]
    if static_flat is None:
        C = shp[3] // (pH * pW)
    else:
        if static_flat % (pH * pW) != 0:
            raise ValueError(
                f"`patches` last dim ({static_flat}) is not divisible by "
                f"prod(size) ({pH * pW})."
            )
        C = static_flat // (pH * pW)

    x = ops.reshape(patches, (B, gH, gW, pH, pW, C))
    x = ops.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (B, gH * pH, gW * pW, C))

    if padding == "same":
        pad_total_h = gH * pH - H
        pad_total_w = gW * pW - W
        begin = [0, pad_total_h // 2, pad_total_w // 2, 0]
        out_shape = [B, H, W, C]
        x = ops.slice(x, begin, out_shape)
    else:
        if gH * pH != H or gW * pW != W:
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"size * grid. Got output_size=({H},{W}); for grid=({gH},{gW}) "
                f"and size=({pH},{pW}) expected output_size=({gH*pH},{gW*pW})."
            )
    return x


def _reconstruct_2d_overlap_cl(patches, size, output_size, strides, padding, reduction="mean", dilation_rate=1):
    """conv-transpose path for overlapping strides, channels_last.

    Each overlapping output pixel gets the SUM of contributing patches; we
    divide by the count of contributions to recover the average. When the
    input was extracted from a consistent image, sum == count * original,
    so the average is exact.
    """
    pH, pW = size
    H, W = output_size
    sH, sW = strides

    static_flat = patches.shape[-1]
    if static_flat is None:
        raise ValueError(
            "For overlapping reconstruction, the last dim of `patches` "
            "must be statically known."
        )
    if static_flat % (pH * pW) != 0:
        raise ValueError(
            f"`patches` last dim ({static_flat}) is not divisible by "
            f"prod(size) ({pH * pW})."
        )
    C = static_flat // (pH * pW)
    out_dim = pH * pW * C

    # Identity kernel: (pH, pW, C_out=C, C_in=out_dim) for conv_transpose
    kernel = backend.numpy.eye(out_dim, dtype=patches.dtype)
    kernel = backend.numpy.reshape(kernel, (pH, pW, C, out_dim))

    grid_h = patches.shape[1]
    grid_w = patches.shape[2]
    if grid_h is None or grid_w is None:
        raise ValueError(
            "For overlapping reconstruction, the patch-grid dims of "
            "`patches` must be statically known."
        )

    # Effective kernel size accounting for dilation: (k - 1) * d + 1
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate)
    dH, dW = dilation_rate
    eff_pH = (pH - 1) * dH + 1
    eff_pW = (pW - 1) * dW + 1

    if padding == "valid":
        op_h = H - (grid_h - 1) * sH - eff_pH
        op_w = W - (grid_w - 1) * sW - eff_pW
        if not (0 <= op_h < sH):
            min_valid = (grid_h - 1) * sH + eff_pH
            raise ValueError(
                f"output_size H={H} is inconsistent. For grid_h={grid_h}, "
                f"stride={sH}, effective_patch={eff_pH} (patch={pH}, "
                f"dilation={dH}), expected H in [{min_valid}, {min_valid + sH})."
            )
        if not (0 <= op_w < sW):
            min_valid = (grid_w - 1) * sW + eff_pW
            raise ValueError(
                f"output_size W={W} is inconsistent. For grid_w={grid_w}, "
                f"stride={sW}, effective_patch={eff_pW} (patch={pW}, "
                f"dilation={dW}), expected W in [{min_valid}, {min_valid + sW})."
            )
        output_padding = (op_h, op_w)
    else:  # same
        # output_padding=None lets the backend infer (avoids a TF translation
        # bug where explicit (0,0) fails). For "same" we always crop after.
        output_padding = None

    output_sum = backend.nn.conv_transpose(
        inputs=patches,
        kernel=kernel,
        strides=(sH, sW),
        padding=padding,
        output_padding=output_padding,
        data_format="channels_last",
        dilation_rate=dilation_rate,
    )
    if reduction == "mean":
        counts = backend.nn.conv_transpose(
            inputs=ops.ones_like(patches),
            kernel=kernel,
            strides=(sH, sW),
            padding=padding,
            output_padding=output_padding,
            data_format="channels_last",
            dilation_rate=dilation_rate,
        )

    if padding == "same":
        cur_shape = ops.shape(output_sum)
        cur_h, cur_w = cur_shape[1], cur_shape[2]
        pad_total_h = cur_h - H
        pad_total_w = cur_w - W
        begin = [0, pad_total_h // 2, pad_total_w // 2, 0]
        B = cur_shape[0]
        out_shape = [B, H, W, C]
        output_sum = ops.slice(output_sum, begin, out_shape)
        if reduction == "mean":
            counts = ops.slice(counts, begin, out_shape)

    if reduction == "mean":
        one = ops.cast(1, output_sum.dtype)
        return output_sum / ops.maximum(counts, one)
    return output_sum  # reduction == "sum"


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def _reconstruct_patches_3d(
    patches, size, output_size, strides=None, padding="valid",
    data_format=None, reduction="mean", dilation_rate=1,
):
    if isinstance(size, int):
        size = (size, size, size)
    if len(size) != 3:
        raise ValueError(
            "Invalid `size`. Expected length 3 for 3D reconstruction. "
            f"Got: size={size}"
        )
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    strides = _normalize_strides(strides, size, "reconstruct_patches_3d")
    if output_size is None:
        if padding != "valid":
            raise ValueError(
                "`output_size=None` (auto-infer) is only supported for "
                "padding='valid'. For padding='same', the original size is "
                "ambiguous from patches alone — please pass output_size "
                "explicitly."
            )
        output_size = _infer_output_size_valid(patches, size, strides, data_format)
    if len(output_size) != 3:
        raise ValueError(
            "Invalid `output_size`. Expected length 3 (D, H, W). "
            f"Got: output_size={output_size}"
        )
    data_format = backend.standardize_data_format(data_format)

    if data_format == "channels_first":
        if len(patches.shape) == 4:    # unbatched (flat, gD, gH, gW)
            patches_cl = ops.transpose(patches, axes=(1, 2, 3, 0))
        elif len(patches.shape) == 5:  # batched (B, flat, gD, gH, gW)
            patches_cl = ops.transpose(patches, axes=(0, 2, 3, 4, 1))
        else:
            raise ValueError(
                f"`patches` has unexpected rank for 3D channels_first: "
                f"got shape {patches.shape}"
            )
        result_cl = _reconstruct_patches_3d_cl(
            patches_cl, size, output_size, strides, padding, reduction, dilation_rate,
        )
        if len(patches.shape) == 4:
            return ops.transpose(result_cl, axes=(3, 0, 1, 2))
        return ops.transpose(result_cl, axes=(0, 4, 1, 2, 3))

    return _reconstruct_patches_3d_cl(
        patches, size, output_size, strides, padding, reduction, dilation_rate,
    )


def _reconstruct_patches_3d_cl(patches, size, output_size, strides, padding, reduction, dilation_rate=1):
    _unbatched = (len(patches.shape) == 4)
    if _unbatched:
        patches = ops.expand_dims(patches, axis=0)

    if _is_nonoverlapping(strides, size) and dilation_rate == 1:
        result = _reconstruct_3d_nonoverlap_cl(patches, size, output_size, padding)
    else:
        result = _reconstruct_3d_overlap_cl(
            patches, size, output_size, strides, padding, reduction, dilation_rate,
        )

    if _unbatched:
        result = ops.squeeze(result, axis=0)
    return result


def _reconstruct_3d_nonoverlap_cl(patches, size, output_size, padding):
    pD, pH, pW = size
    D, H, W = output_size

    shp = ops.shape(patches)
    B, gD, gH, gW = shp[0], shp[1], shp[2], shp[3]
    static_flat = patches.shape[-1]
    if static_flat is None:
        C = shp[4] // (pD * pH * pW)
    else:
        if static_flat % (pD * pH * pW) != 0:
            raise ValueError(
                f"`patches` last dim ({static_flat}) is not divisible by "
                f"prod(size) ({pD * pH * pW})."
            )
        C = static_flat // (pD * pH * pW)

    x = ops.reshape(patches, (B, gD, gH, gW, pD, pH, pW, C))
    x = ops.transpose(x, axes=(0, 1, 4, 2, 5, 3, 6, 7))
    x = ops.reshape(x, (B, gD * pD, gH * pH, gW * pW, C))

    if padding == "same":
        pad_total_d = gD * pD - D
        pad_total_h = gH * pH - H
        pad_total_w = gW * pW - W
        begin = [0, pad_total_d // 2, pad_total_h // 2, pad_total_w // 2, 0]
        out_shape = [B, D, H, W, C]
        x = ops.slice(x, begin, out_shape)
    else:
        if gD * pD != D or gH * pH != H or gW * pW != W:
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"size * grid. Got output_size=({D},{H},{W}); for "
                f"grid=({gD},{gH},{gW}) and size=({pD},{pH},{pW}) expected "
                f"output_size=({gD*pD},{gH*pH},{gW*pW})."
            )
    return x


def _reconstruct_3d_overlap_cl(patches, size, output_size, strides, padding, reduction="mean", dilation_rate=1):
    pD, pH, pW = size
    D, H, W = output_size
    sD, sH, sW = strides

    static_flat = patches.shape[-1]
    if static_flat is None:
        raise ValueError(
            "For overlapping reconstruction, the last dim of `patches` "
            "must be statically known."
        )
    if static_flat % (pD * pH * pW) != 0:
        raise ValueError(
            f"`patches` last dim ({static_flat}) is not divisible by "
            f"prod(size) ({pD * pH * pW})."
        )
    C = static_flat // (pD * pH * pW)
    out_dim = pD * pH * pW * C

    kernel = backend.numpy.eye(out_dim, dtype=patches.dtype)
    kernel = backend.numpy.reshape(kernel, (pD, pH, pW, C, out_dim))

    grid_d = patches.shape[1]
    grid_h = patches.shape[2]
    grid_w = patches.shape[3]
    if grid_d is None or grid_h is None or grid_w is None:
        raise ValueError(
            "For overlapping reconstruction, the patch-grid dims of "
            "`patches` must be statically known."
        )

    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate, dilation_rate)
    dD, dH, dW = dilation_rate
    eff_pD = (pD - 1) * dD + 1
    eff_pH = (pH - 1) * dH + 1
    eff_pW = (pW - 1) * dW + 1

    if padding == "valid":
        op_d = D - (grid_d - 1) * sD - eff_pD
        op_h = H - (grid_h - 1) * sH - eff_pH
        op_w = W - (grid_w - 1) * sW - eff_pW
        for label, op, stride, grid, patch, dim in (
            ("D", op_d, sD, grid_d, eff_pD, D),
            ("H", op_h, sH, grid_h, eff_pH, H),
            ("W", op_w, sW, grid_w, eff_pW, W),
        ):
            if not (0 <= op < stride):
                min_valid = (grid - 1) * stride + patch
                raise ValueError(
                    f"output_size {label}={dim} is inconsistent. For "
                    f"grid_{label.lower()}={grid}, stride={stride}, "
                    f"effective_patch={patch}, expected {label} in "
                    f"[{min_valid}, {min_valid + stride})."
                )
        output_padding = (op_d, op_h, op_w)
    else:
        output_padding = None  # see comment in 2D version

    output_sum = backend.nn.conv_transpose(
        inputs=patches,
        kernel=kernel,
        strides=(sD, sH, sW),
        padding=padding,
        output_padding=output_padding,
        data_format="channels_last",
        dilation_rate=dilation_rate,
    )
    if reduction == "mean":
        counts = backend.nn.conv_transpose(
            inputs=ops.ones_like(patches),
            kernel=kernel,
            strides=(sD, sH, sW),
            padding=padding,
            output_padding=output_padding,
            data_format="channels_last",
            dilation_rate=dilation_rate,
        )

    if padding == "same":
        cur_shape = ops.shape(output_sum)
        cur_d, cur_h, cur_w = cur_shape[1], cur_shape[2], cur_shape[3]
        pad_total_d = cur_d - D
        pad_total_h = cur_h - H
        pad_total_w = cur_w - W
        begin = [
            0,
            pad_total_d // 2,
            pad_total_h // 2,
            pad_total_w // 2,
            0,
        ]
        B = cur_shape[0]
        out_shape = [B, D, H, W, C]
        output_sum = ops.slice(output_sum, begin, out_shape)
        if reduction == "mean":
            counts = ops.slice(counts, begin, out_shape)

    if reduction == "mean":
        one = ops.cast(1, output_sum.dtype)
        return output_sum / ops.maximum(counts, one)
    return output_sum  # reduction == "sum"


# ---------------------------------------------------------------------------
# Layer wrappers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Forward Layer wrappers (thin around keras.ops.image.extract_patches)
# ---------------------------------------------------------------------------


def _compute_grid(input_size, kernel, stride, padding):
    """Single-axis output dim for a conv with given args; matches extract_patches."""
    if input_size is None:
        return None
    if padding == "valid":
        return max(0, (input_size - kernel) // stride + 1)
    # same
    return (input_size + stride - 1) // stride


def _extract_compute_output_shape(input_shape, size, strides, padding, data_format):
    """Shared compute_output_shape for ExtractPatches{2,3}D layer wrappers."""
    n_spatial = len(size)
    rank = len(input_shape)
    if rank not in (n_spatial + 1, n_spatial + 2):
        raise ValueError(
            f"ExtractPatches expected rank {n_spatial + 1} or "
            f"{n_spatial + 2} (with batch); got input_shape={input_shape}"
        )
    batch = input_shape[0] if rank == n_spatial + 2 else None
    if data_format == "channels_last":
        channels = input_shape[-1]
        spatial = input_shape[-(n_spatial + 1):-1]
    else:
        if rank == n_spatial + 2:
            channels = input_shape[1]
            spatial = input_shape[2:]
        else:  # unbatched
            channels = input_shape[0]
            spatial = input_shape[1:]
    flat = None
    if channels is not None:
        flat = channels
        for s in size:
            flat *= s
    grid = tuple(_compute_grid(d, k, s, padding)
                 for d, k, s in zip(spatial, size, strides))
    if data_format == "channels_last":
        spatial_out = grid + (flat,)
    else:
        spatial_out = (flat,) + grid
    if rank == n_spatial + 2:
        return (batch,) + spatial_out
    return spatial_out


@keras_export("keras.layers.ExtractPatches2D")
@keras.saving.register_keras_serializable(package="reconstruct_patches")
class ExtractPatches2D(Layer):
    """Layer wrapper for `keras.ops.image.extract_patches` (2D).

    Identical semantics to the op; provided so users can compose extract →
    reconstruct as symmetric Layer pairs in Functional/Sequential models
    without needing a Lambda layer.

    Args:
        size: int or tuple `(pH, pW)`. Patch size.
        strides: int or tuple. Defaults to `size` (non-overlapping).
        padding: `"valid"` or `"same"`.
        data_format: `"channels_last"` or `"channels_first"`.

    Input shape:
        4D `(batch, H, W, C)` for channels_last, or
        4D `(batch, C, H, W)` for channels_first.

    Output shape:
        4D `(batch, gH, gW, pH*pW*C)` for channels_last, or
        4D `(batch, pH*pW*C, gH, gW)` for channels_first.

    Examples:

    >>> import numpy as np
    >>> from reconstruct_patches import ExtractPatches2D
    >>> image = np.random.rand(2, 16, 16, 3).astype("float32")
    >>> layer = ExtractPatches2D(size=(4, 4), padding="valid")
    >>> patches = layer(image)
    >>> tuple(patches.shape)
    (2, 4, 4, 48)
    """

    def __init__(
        self,
        size,
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError(
                f"`size` must be int or tuple of length 2; got {size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'; got {padding}"
            )
        self.size = tuple(size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, images):
        return ops.image.extract_patches(
            images,
            size=self.size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides)
        return _extract_compute_output_shape(
            input_shape, self.size, strides, self.padding, self.data_format,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        return {**base_config, **config}


@keras_export("keras.layers.ExtractPatches3D")
@keras.saving.register_keras_serializable(package="reconstruct_patches")
class ExtractPatches3D(Layer):
    """Layer wrapper for `keras.ops.image.extract_patches` (3D).

    Identical semantics to the op with a length-3 `size`. See
    `ExtractPatches2D` for description; this variant handles 5D volume
    inputs and 3D patch sizes.
    """

    def __init__(
        self,
        size,
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size, size)
        if len(size) != 3:
            raise ValueError(
                f"`size` must be int or tuple of length 3; got {size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'; got {padding}"
            )
        self.size = tuple(size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, volumes):
        # Use the explicit *_3d entry point so the dispatcher doesn't pick the
        # 2D path when called via a generic helper that re-routes by size length.
        return ops.image.extract_patches_3d(
            volumes,
            size=self.size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        return _extract_compute_output_shape(
            input_shape, self.size, strides, self.padding, self.data_format,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        return {**base_config, **config}


# ---------------------------------------------------------------------------
# Inverse Layer wrappers
# ---------------------------------------------------------------------------


@keras_export("keras.layers.ReconstructPatches3D")
@keras.saving.register_keras_serializable(package="reconstruct_patches")
class ReconstructPatches3D(Layer):
    """Layer wrapper for `keras.ops.image.reconstruct_patches_3d`."""

    def __init__(
        self,
        size,
        output_size=None,
        strides=None,
        padding="valid",
        data_format=None,
        reduction="mean",
        dilation_rate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size, size)
        if len(size) != 3:
            raise ValueError(
                f"`size` must be an int or a tuple of length 3. "
                f"Received: size={size}"
            )
        if output_size is None:
            # `output_size=None` is legal in two modes:
            # 1. padding='valid' — output_size is inferred from patches shape
            #    at call time (deterministic for valid).
            # 2. dual-input call(): user passes [patches, reference] at
            #    call time and output_size is derived from reference's shape.
            # Both cases are handled in call(); no error here.
            pass
        elif len(output_size) != 3:
            raise ValueError(
                f"`output_size` must be a tuple of length 3 (D, H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"`reduction` must be 'mean' or 'sum'. Received: {reduction}"
            )
        # Eagerly validate strides (rejects gapped strides at construct time).
        _normalize_strides(strides, size, "ReconstructPatches3D")
        self.size = tuple(size)
        self.output_size = tuple(output_size) if output_size is not None else None
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        self.reduction = reduction
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        # Dual-input mode: [patches, reference_tensor]. Uses ops.shape(reference)
        # at call time to derive output_size, enabling drop-in replacement for
        # the original ReconstructVolumePatchesLayer pattern from variable-input
        # models declared with Input(shape=[None,None,None,C]).
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(
                    "ReconstructPatches3D called with a list expects exactly "
                    f"[patches, reference], got list of length {len(inputs)}."
                )
            patches, reference = inputs
            ref_shape = ops.shape(reference)
            if self.data_format == "channels_last":
                # batched (B, D, H, W, C): spatial at axes 1, 2, 3
                output_size = (ref_shape[1], ref_shape[2], ref_shape[3])
            else:
                # batched (B, C, D, H, W): spatial at axes 2, 3, 4
                output_size = (ref_shape[2], ref_shape[3], ref_shape[4])
        else:
            patches = inputs
            output_size = self.output_size
        return reconstruct_patches_3d(
            patches,
            size=self.size,
            output_size=output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            reduction=self.reduction,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_shape(self, input_shape):
        patch_volume = self.size[0] * self.size[1] * self.size[2]
        # Dual-input: input_shape is [patches_shape, reference_shape].
        if isinstance(input_shape, list) and len(input_shape) == 2:
            patches_shape, ref_shape = input_shape
            # Use reference's spatial dims as output_size (may have Nones).
            if self.data_format == "channels_last":
                output_size = tuple(ref_shape[1:4])
            else:
                output_size = tuple(ref_shape[2:5])
            input_shape = patches_shape  # for the rest of the computation
        elif self.output_size is not None:
            output_size = self.output_size
        else:
            output_size = self._infer_output_size_from_shape(input_shape)
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 5:
                return (input_shape[0],) + tuple(output_size) + (channels,)
            elif len(input_shape) == 4:
                return tuple(output_size) + (channels,)
        else:
            if len(input_shape) == 5:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + tuple(output_size)
            elif len(input_shape) == 4:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + tuple(output_size)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches3D: "
            f"{len(input_shape)}"
        )

    def _infer_output_size_from_shape(self, input_shape):
        """Compute output_size from input_shape for padding='valid'."""
        if self.data_format == "channels_last":
            grid = input_shape[-4:-1] if len(input_shape) == 5 else input_shape[:-1]
        else:
            grid = input_shape[-3:] if len(input_shape) >= 4 else input_shape[1:]
        if any(g is None for g in grid):
            return (None, None, None)
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides, strides)
        return tuple(
            (g - 1) * s + k for g, s, k in zip(grid, strides, self.size)
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "reduction": self.reduction,
            "dilation_rate": self.dilation_rate,
        }
        return {**base_config, **config}


@keras_export("keras.layers.ReconstructPatches2D")
@keras.saving.register_keras_serializable(package="reconstruct_patches")
class ReconstructPatches2D(Layer):
    """Layer wrapper for `keras.ops.image.reconstruct_patches` (2D)."""

    def __init__(
        self,
        size,
        output_size=None,
        strides=None,
        padding="valid",
        data_format=None,
        reduction="mean",
        dilation_rate=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2:
            raise ValueError(
                f"`size` must be an int or a tuple of length 2. "
                f"Received: size={size}"
            )
        if output_size is None:
            # See ReconstructPatches3D for the two legal output_size=None modes.
            pass
        elif len(output_size) != 2:
            raise ValueError(
                f"`output_size` must be a tuple of length 2 (H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"`reduction` must be 'mean' or 'sum'. Received: {reduction}"
            )
        # Eagerly validate strides (rejects gapped strides at construct time).
        _normalize_strides(strides, size, "ReconstructPatches2D")
        self.size = tuple(size)
        self.output_size = tuple(output_size) if output_size is not None else None
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)
        self.reduction = reduction
        self.dilation_rate = dilation_rate

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError(
                    "ReconstructPatches2D called with a list expects exactly "
                    f"[patches, reference], got list of length {len(inputs)}."
                )
            patches, reference = inputs
            ref_shape = ops.shape(reference)
            if self.data_format == "channels_last":
                output_size = (ref_shape[1], ref_shape[2])
            else:
                output_size = (ref_shape[2], ref_shape[3])
        else:
            patches = inputs
            output_size = self.output_size
        return reconstruct_patches(
            patches,
            size=self.size,
            output_size=output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            reduction=self.reduction,
            dilation_rate=self.dilation_rate,
        )

    def compute_output_shape(self, input_shape):
        patch_volume = self.size[0] * self.size[1]
        if isinstance(input_shape, list) and len(input_shape) == 2:
            patches_shape, ref_shape = input_shape
            if self.data_format == "channels_last":
                output_size = tuple(ref_shape[1:3])
            else:
                output_size = tuple(ref_shape[2:4])
            input_shape = patches_shape
        elif self.output_size is not None:
            output_size = self.output_size
        else:
            output_size = self._infer_output_size_from_shape(input_shape)
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 4:
                return (input_shape[0],) + tuple(output_size) + (channels,)
            elif len(input_shape) == 3:
                return tuple(output_size) + (channels,)
        else:
            if len(input_shape) == 4:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + tuple(output_size)
            elif len(input_shape) == 3:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + tuple(output_size)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches2D: "
            f"{len(input_shape)}"
        )

    def _infer_output_size_from_shape(self, input_shape):
        if self.data_format == "channels_last":
            grid = input_shape[-3:-1] if len(input_shape) == 4 else input_shape[:-1]
        else:
            grid = input_shape[-2:] if len(input_shape) >= 3 else input_shape[1:]
        if any(g is None for g in grid):
            return (None, None)
        strides = self.strides if self.strides is not None else self.size
        if isinstance(strides, int):
            strides = (strides, strides)
        return tuple(
            (g - 1) * s + k for g, s, k in zip(grid, strides, self.size)
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "reduction": self.reduction,
            "dilation_rate": self.dilation_rate,
        }
        return {**base_config, **config}
