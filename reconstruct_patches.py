"""Prototype: inverse of `keras.ops.image.extract_patches`.

Designed to live at `keras/src/ops/image.py` alongside `extract_patches` once
upstreamed. Multi-backend (TF/JAX/PyTorch) via `keras.ops` and `keras` `backend.*`
only — no `tf.*` calls.

Forward op output format (mirror exactly):
    3D: (B, gD, gH, gW, pD*pH*pW*C)   for channels_last, batched
    2D: (B, gH, gW, pH*pW*C)          for channels_last, batched
    Unbatched drops the leading B.

Inverse output:
    3D: (B, D, H, W, C)
    2D: (B, H, W, C)

Scope of this prototype: non-overlapping reconstruction (strides == size,
dilation_rate == 1). Overlap-add reconstruction (the analogue of
`torch.nn.Fold` over arbitrary strides) is intentionally deferred — it raises
NotImplementedError, and would be a follow-up PR using a transposed-conv with
identity kernel.
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
# Ops layer
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
    """Reconstructs image(s) or volume(s) from non-overlapping patches.

    Inverse of `keras.ops.image.extract_patches` for the non-overlapping case
    (`strides == size`). For overlapping reconstruction, see
    `reconstruct_patches_overlapping` (not yet implemented).

    Args:
        patches: Patches tensor as produced by `extract_patches`.
            For 2D patches: 3D `(gH, gW, pH*pW*C)` or
            4D `(B, gH, gW, pH*pW*C)`.
            For 3D patches: 4D `(gD, gH, gW, pD*pH*pW*C)` or
            5D `(B, gD, gH, gW, pD*pH*pW*C)`.
        size: Patch size, matching the `size` used for extraction.
            Length 2 tuple for 2D, length 3 tuple for 3D, or int.
        output_size: Spatial shape of the original image/volume before
            extraction. Length 2 tuple `(H, W)` for 2D, length 3 tuple
            `(D, H, W)` for 3D. Required so that `"same"` padding can be
            unambiguously inverted.
        strides: Currently must equal `size` (non-overlapping). Defaults
            to `size`.
        padding: `"same"` or `"valid"`, matching the extraction.
        data_format: `"channels_last"` or `"channels_first"`. Defaults to
            `keras.config.image_data_format()`.

    Returns:
        Reconstructed image/volume:
            2D: 3D or 4D matching `patches` batched-ness.
            3D: 4D or 5D matching `patches` batched-ness.
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
            patches, size, output_size, strides, padding, data_format
        )
    return _reconstruct_patches_2d(
        patches, size, output_size, strides, padding, data_format
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
    """Reconstructs volume(s) from non-overlapping 3D patches.

    Inverse of `keras.ops.image.extract_patches_3d`.

    Args:
        patches: 4D `(gD, gH, gW, pD*pH*pW*C)` or
            5D `(B, gD, gH, gW, pD*pH*pW*C)`.
        size: int or tuple `(patch_depth, patch_height, patch_width)`.
        output_size: tuple `(D, H, W)` — original spatial shape.
        strides: must equal `size`. Defaults to `size`.
        padding: `"same"` or `"valid"`.
        data_format: `"channels_last"` or `"channels_first"`.

    Returns:
        Reconstructed volume, 4D or 5D matching input.
    """
    if isinstance(size, int):
        size = (size, size, size)
    return _reconstruct_patches_3d(
        patches, size, output_size, strides, padding, data_format
    )


# ---------------------------------------------------------------------------
# Backend-level implementations
# ---------------------------------------------------------------------------

def _validate_non_overlapping(size, strides, name):
    if strides is None:
        return size
    if isinstance(strides, int):
        strides = (strides,) * len(size)
    if tuple(strides) != tuple(size):
        raise NotImplementedError(
            f"`{name}` currently supports only non-overlapping "
            f"reconstruction (strides == size). Got strides={strides} "
            f"and size={size}. Overlap-add reconstruction is planned for "
            f"a follow-up."
        )
    return tuple(strides)


def _crop_to_output_same(volume, pad_total_per_axis, output_size, axes):
    """Crop a padded volume back to `output_size` along the given axes.

    Mirrors the way `backend.nn.conv(padding="same")` distributes padding:
    pad_before = pad_total // 2, pad_after = pad_total - pad_before.
    """
    begin = [0] * len(volume.shape)
    out_shape = list(ops.shape(volume))
    for axis, pad_total, out_dim in zip(axes, pad_total_per_axis, output_size):
        begin[axis] = pad_total // 2
        out_shape[axis] = out_dim
    return ops.slice(volume, begin, out_shape)


def _reconstruct_patches_3d(
    patches, size, output_size, strides=None, padding="valid", data_format=None,
):
    _validate_non_overlapping(size, strides, "reconstruct_patches_3d")
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
    data_format = backend.standardize_data_format(data_format)

    pD, pH, pW = size
    D, H, W = output_size

    _unbatched = False
    if len(patches.shape) == 4:
        _unbatched = True
        patches = ops.expand_dims(patches, axis=0)

    # patches: (B, gD, gH, gW, pD*pH*pW*C) for channels_last
    if data_format == "channels_first":
        # Forward op uses conv data_format; for channels_first we'd need to
        # transpose first. Defer for clarity — channels_last is the primary
        # path in keras.ops.image.
        raise NotImplementedError(
            "channels_first is not yet supported in this prototype."
        )

    shp = ops.shape(patches)
    B, gD, gH, gW = shp[0], shp[1], shp[2], shp[3]
    flat_last = shp[4]
    # C is unknown from `patches` alone; compute it from flat_last / (pD*pH*pW)
    # statically when possible, dynamically otherwise.
    static_flat = patches.shape[-1]
    if static_flat is None:
        # dynamic: assume flat_last is divisible by pD*pH*pW
        C = flat_last // (pD * pH * pW)
    else:
        if static_flat % (pD * pH * pW) != 0:
            raise ValueError(
                f"`patches` last dim ({static_flat}) is not divisible by "
                f"prod(size) ({pD * pH * pW}). Are `size` and the patches "
                f"tensor consistent?"
            )
        C = static_flat // (pD * pH * pW)

    # (B, gD, gH, gW, pD*pH*pW*C) -> (B, gD, gH, gW, pD, pH, pW, C)
    x = ops.reshape(patches, (B, gD, gH, gW, pD, pH, pW, C))
    # -> (B, gD, pD, gH, pH, gW, pW, C)
    x = ops.transpose(x, axes=(0, 1, 4, 2, 5, 3, 6, 7))
    # -> (B, gD*pD, gH*pH, gW*pW, C)
    x = ops.reshape(x, (B, gD * pD, gH * pH, gW * pW, C))

    if padding == "same":
        # padded_D = gD*pD, original D = output_size[0]
        # pad_total = padded - original; same convention as backend.nn.conv "same"
        pad_total_d = gD * pD - D
        pad_total_h = gH * pH - H
        pad_total_w = gW * pW - W
        x = _crop_to_output_same(
            x, (pad_total_d, pad_total_h, pad_total_w), (D, H, W),
            axes=(1, 2, 3),
        )
    else:
        # valid: gD*pD must equal output_size[0]. Slice anyway for safety.
        if gD * pD != D or gH * pH != H or gW * pW != W:
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"size * grid. Got output_size=({D},{H},{W}), "
                f"grid=({gD},{gH},{gW}), size=({pD},{pH},{pW})."
            )

    if _unbatched:
        x = ops.squeeze(x, axis=0)
    return x


def _reconstruct_patches_2d(
    patches, size, output_size, strides=None, padding="valid", data_format=None,
):
    _validate_non_overlapping(size, strides, "reconstruct_patches")
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
    data_format = backend.standardize_data_format(data_format)

    pH, pW = size
    H, W = output_size

    _unbatched = False
    if len(patches.shape) == 3:
        _unbatched = True
        patches = ops.expand_dims(patches, axis=0)

    if data_format == "channels_first":
        raise NotImplementedError(
            "channels_first is not yet supported in this prototype."
        )

    shp = ops.shape(patches)
    B, gH, gW = shp[0], shp[1], shp[2]
    flat_last = shp[3]
    static_flat = patches.shape[-1]
    if static_flat is None:
        C = flat_last // (pH * pW)
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
        x = _crop_to_output_same(
            x, (pad_total_h, pad_total_w), (H, W), axes=(1, 2),
        )
    else:
        if gH * pH != H or gW * pW != W:
            raise ValueError(
                f"`padding='valid'` requires output_size to equal "
                f"size * grid. Got output_size=({H},{W}), "
                f"grid=({gH},{gW}), size=({pH},{pW})."
            )

    if _unbatched:
        x = ops.squeeze(x, axis=0)
    return x


# ---------------------------------------------------------------------------
# Layer wrappers
# ---------------------------------------------------------------------------

@keras_export("keras.layers.ReconstructPatches3D")
class ReconstructPatches3D(Layer):
    """Layer wrapper for `keras.ops.image.reconstruct_patches_3d`.

    Inverse of `ExtractPatches3D` (or `keras.ops.image.extract_patches` with a
    length-3 `size`) for the non-overlapping case.

    Args:
        size: int or tuple `(pD, pH, pW)`.
        output_size: tuple `(D, H, W)` — original spatial shape.
        strides: must equal `size`. Defaults to `size`.
        padding: `"same"` or `"valid"`.
        data_format: `"channels_last"` or `"channels_first"`.
    """

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
        # patches: (B, gD, gH, gW, flat) or (gD, gH, gW, flat)
        if len(input_shape) == 5:
            B = input_shape[0]
            return (B,) + self.output_size + (self._channels(input_shape),)
        elif len(input_shape) == 4:
            return self.output_size + (self._channels(input_shape),)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches3D: {len(input_shape)}"
        )

    def _channels(self, input_shape):
        flat = input_shape[-1]
        if flat is None:
            return None
        return flat // (self.size[0] * self.size[1] * self.size[2])

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
    """Layer wrapper for `keras.ops.image.reconstruct_patches` (2D).

    Inverse of `ExtractPatches2D` for the non-overlapping case.

    Args:
        size: int or tuple `(pH, pW)`.
        output_size: tuple `(H, W)` — original spatial shape.
        strides: must equal `size`. Defaults to `size`.
        padding: `"same"` or `"valid"`.
        data_format: `"channels_last"` or `"channels_first"`.
    """

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
        if len(input_shape) == 4:
            B = input_shape[0]
            return (B,) + self.output_size + (self._channels(input_shape),)
        elif len(input_shape) == 3:
            return self.output_size + (self._channels(input_shape),)
        raise ValueError(
            f"Unexpected patches rank for ReconstructPatches2D: {len(input_shape)}"
        )

    def _channels(self, input_shape):
        flat = input_shape[-1]
        if flat is None:
            return None
        return flat // (self.size[0] * self.size[1])

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
