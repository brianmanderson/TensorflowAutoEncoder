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


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------

@keras_export("keras.ops.image.reconstruct_patches")
def reconstruct_patches(
    patches,
    size,
    output_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Reconstructs image(s) or volume(s) from patches.

    Inverse of `keras.ops.image.extract_patches`. Supports both non-overlapping
    (`strides == size`) and overlapping (`strides < size`) cases. For
    overlapping patches, the result is the per-pixel mean of overlapping
    contributions, which exactly recovers the original when patches were
    extracted from a consistent input.

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

    Returns:
        Reconstructed image/volume, matching `patches`' batched-ness and
        `data_format`.
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

    if not isinstance(size, int) and len(size) == 3:
        return _reconstruct_patches_3d(
            patches, size, output_size, strides, padding, data_format,
        )
    return _reconstruct_patches_2d(
        patches, size, output_size, strides, padding, data_format,
    )


@keras_export("keras.ops.image.reconstruct_patches_3d")
def reconstruct_patches_3d(
    patches,
    size,
    output_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    """Reconstructs volume(s) from 3D patches. See `reconstruct_patches`."""
    if isinstance(size, int):
        size = (size, size, size)
    return _reconstruct_patches_3d(
        patches, size, output_size, strides, padding, data_format,
    )


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def _reconstruct_patches_2d(
    patches, size, output_size, strides=None, padding="valid", data_format=None,
):
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError(
            "Invalid `size`. Expected length 2 for 2D reconstruction. "
            f"Got: size={size}"
        )
    if len(output_size) != 2:
        raise ValueError(
            "Invalid `output_size`. Expected length 2 (H, W). "
            f"Got: output_size={output_size}"
        )
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    strides = _normalize_strides(strides, size, "reconstruct_patches")
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
            patches_cl, size, output_size, strides, padding,
        )
        if len(patches.shape) == 3:
            return ops.transpose(result_cl, axes=(2, 0, 1))
        return ops.transpose(result_cl, axes=(0, 3, 1, 2))

    return _reconstruct_patches_2d_cl(
        patches, size, output_size, strides, padding,
    )


def _reconstruct_patches_2d_cl(patches, size, output_size, strides, padding):
    """Channels_last 2D core: dispatches between non-overlap and overlap paths."""
    _unbatched = (len(patches.shape) == 3)
    if _unbatched:
        patches = ops.expand_dims(patches, axis=0)

    if _is_nonoverlapping(strides, size):
        result = _reconstruct_2d_nonoverlap_cl(patches, size, output_size, padding)
    else:
        result = _reconstruct_2d_overlap_cl(
            patches, size, output_size, strides, padding,
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
                f"size * grid. Got output_size=({H},{W}), "
                f"grid=({gH},{gW}), size=({pH},{pW})."
            )
    return x


def _reconstruct_2d_overlap_cl(patches, size, output_size, strides, padding):
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

    if padding == "valid":
        op_h = H - (grid_h - 1) * sH - pH
        op_w = W - (grid_w - 1) * sW - pW
        if not (0 <= op_h < sH):
            raise ValueError(
                f"output_size H={H} is inconsistent with grid_h={grid_h}, "
                f"stride={sH}, patch={pH}. Expected H in "
                f"[(grid-1)*stride + patch, (grid-1)*stride + patch + stride)."
            )
        if not (0 <= op_w < sW):
            raise ValueError(
                f"output_size W={W} is inconsistent with grid_w={grid_w}, "
                f"stride={sW}, patch={pW}."
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
    )
    counts = backend.nn.conv_transpose(
        inputs=ops.ones_like(patches),
        kernel=kernel,
        strides=(sH, sW),
        padding=padding,
        output_padding=output_padding,
        data_format="channels_last",
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
        counts = ops.slice(counts, begin, out_shape)

    one = ops.cast(1, output_sum.dtype)
    return output_sum / ops.maximum(counts, one)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def _reconstruct_patches_3d(
    patches, size, output_size, strides=None, padding="valid", data_format=None,
):
    if isinstance(size, int):
        size = (size, size, size)
    if len(size) != 3:
        raise ValueError(
            "Invalid `size`. Expected length 3 for 3D reconstruction. "
            f"Got: size={size}"
        )
    if len(output_size) != 3:
        raise ValueError(
            "Invalid `output_size`. Expected length 3 (D, H, W). "
            f"Got: output_size={output_size}"
        )
    if padding not in ("same", "valid"):
        raise ValueError(
            f"Invalid `padding`. Expected 'same' or 'valid'. Got: {padding}"
        )
    strides = _normalize_strides(strides, size, "reconstruct_patches_3d")
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
            patches_cl, size, output_size, strides, padding,
        )
        if len(patches.shape) == 4:
            return ops.transpose(result_cl, axes=(3, 0, 1, 2))
        return ops.transpose(result_cl, axes=(0, 4, 1, 2, 3))

    return _reconstruct_patches_3d_cl(
        patches, size, output_size, strides, padding,
    )


def _reconstruct_patches_3d_cl(patches, size, output_size, strides, padding):
    _unbatched = (len(patches.shape) == 4)
    if _unbatched:
        patches = ops.expand_dims(patches, axis=0)

    if _is_nonoverlapping(strides, size):
        result = _reconstruct_3d_nonoverlap_cl(patches, size, output_size, padding)
    else:
        result = _reconstruct_3d_overlap_cl(
            patches, size, output_size, strides, padding,
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
                f"size * grid. Got output_size=({D},{H},{W}), "
                f"grid=({gD},{gH},{gW}), size=({pD},{pH},{pW})."
            )
    return x


def _reconstruct_3d_overlap_cl(patches, size, output_size, strides, padding):
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

    if padding == "valid":
        op_d = D - (grid_d - 1) * sD - pD
        op_h = H - (grid_h - 1) * sH - pH
        op_w = W - (grid_w - 1) * sW - pW
        for label, op, stride in (
            ("D", op_d, sD), ("H", op_h, sH), ("W", op_w, sW),
        ):
            if not (0 <= op < stride):
                raise ValueError(
                    f"output_size {label}={output_size} is inconsistent "
                    f"with grid, stride, patch on the {label} axis."
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
    )
    counts = backend.nn.conv_transpose(
        inputs=ops.ones_like(patches),
        kernel=kernel,
        strides=(sD, sH, sW),
        padding=padding,
        output_padding=output_padding,
        data_format="channels_last",
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
        counts = ops.slice(counts, begin, out_shape)

    one = ops.cast(1, output_sum.dtype)
    return output_sum / ops.maximum(counts, one)


# ---------------------------------------------------------------------------
# Layer wrappers
# ---------------------------------------------------------------------------

@keras_export("keras.layers.ReconstructPatches3D")
class ReconstructPatches3D(Layer):
    """Layer wrapper for `keras.ops.image.reconstruct_patches_3d`."""

    def __init__(
        self,
        size,
        output_size,
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
                f"`size` must be an int or a tuple of length 3. "
                f"Received: size={size}"
            )
        if len(output_size) != 3:
            raise ValueError(
                f"`output_size` must be a tuple of length 3 (D, H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        # Eagerly validate strides (rejects gapped strides at construct time).
        _normalize_strides(strides, size, "ReconstructPatches3D")
        self.size = tuple(size)
        self.output_size = tuple(output_size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, patches):
        return reconstruct_patches_3d(
            patches,
            size=self.size,
            output_size=self.output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            patch_volume = self.size[0] * self.size[1] * self.size[2]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 5:
                return (input_shape[0],) + self.output_size + (channels,)
            elif len(input_shape) == 4:
                return self.output_size + (channels,)
        else:
            patch_volume = self.size[0] * self.size[1] * self.size[2]
            if len(input_shape) == 5:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + self.output_size
            elif len(input_shape) == 4:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + self.output_size
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches3D: "
            f"{len(input_shape)}"
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        return {**base_config, **config}


@keras_export("keras.layers.ReconstructPatches2D")
class ReconstructPatches2D(Layer):
    """Layer wrapper for `keras.ops.image.reconstruct_patches` (2D)."""

    def __init__(
        self,
        size,
        output_size,
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
                f"`size` must be an int or a tuple of length 2. "
                f"Received: size={size}"
            )
        if len(output_size) != 2:
            raise ValueError(
                f"`output_size` must be a tuple of length 2 (H, W). "
                f"Received: output_size={output_size}"
            )
        if padding not in ("same", "valid"):
            raise ValueError(
                f"`padding` must be 'same' or 'valid'. "
                f"Received: padding={padding}"
            )
        # Eagerly validate strides (rejects gapped strides at construct time).
        _normalize_strides(strides, size, "ReconstructPatches2D")
        self.size = tuple(size)
        self.output_size = tuple(output_size)
        self.strides = strides
        self.padding = padding
        self.data_format = backend.standardize_data_format(data_format)

    def call(self, patches):
        return reconstruct_patches(
            patches,
            size=self.size,
            output_size=self.output_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
        )

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            flat = input_shape[-1]
            patch_volume = self.size[0] * self.size[1]
            channels = None if flat is None else flat // patch_volume
            if len(input_shape) == 4:
                return (input_shape[0],) + self.output_size + (channels,)
            elif len(input_shape) == 3:
                return self.output_size + (channels,)
        else:
            patch_volume = self.size[0] * self.size[1]
            if len(input_shape) == 4:
                flat = input_shape[1]
                channels = None if flat is None else flat // patch_volume
                return (input_shape[0], channels) + self.output_size
            elif len(input_shape) == 3:
                flat = input_shape[0]
                channels = None if flat is None else flat // patch_volume
                return (channels,) + self.output_size
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches2D: "
            f"{len(input_shape)}"
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "size": self.size,
            "output_size": self.output_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        return {**base_config, **config}
