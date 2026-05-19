"""Quick performance benchmarks for ReconstructPatches{2,3}D.

Run:
    KERAS_BACKEND=tensorflow python benchmarks/benchmark_reconstruct.py

Compares wall-clock per call for the non-overlap path (reshape/transpose)
vs the overlap path (conv_transpose + averaging), across a few representative
shapes. Also measures the layer overhead vs the bare op.

Not a pytest test — purely informational. Numbers vary by machine, backend,
and hardware. Use to inform "do I need the fast path?" decisions in
production code.
"""

import argparse
import os
import statistics
import sys
import time

# Make `reconstruct_patches` importable when running as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras  # noqa: F401 (forces backend selection from env)
import numpy as np
from keras import ops

from reconstruct_patches import (
    ReconstructPatches2D,
    ReconstructPatches3D,
    reconstruct_patches,
    reconstruct_patches_3d,
)


def _time(fn, n_warmup=3, n_iters=20):
    """Run `fn` n_warmup + n_iters times, return median runtime in seconds."""
    for _ in range(n_warmup):
        fn()
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        # Force materialization for JAX (async).
        _ = ops.convert_to_numpy(result)
        timings.append(time.perf_counter() - t0)
    return statistics.median(timings)


def bench_2d():
    print("\n=== 2D ===")
    print(f"{'shape':<25} {'patch':<10} {'stride':<10} {'mode':<12} "
          f"{'op (ms)':<10} {'layer (ms)':<12} {'overhead':<10}")
    print("-" * 95)
    cases = [
        # (B, H, W, C, size, strides)
        (1, 224, 224, 3, (16, 16), (16, 16)),    # ViT-Base
        (1, 224, 224, 3, (16, 16), (8, 8)),      # 50% overlap ViT
        (1, 224, 224, 3, (16, 16), (1, 1)),      # max overlap (sliding window)
        (1, 512, 512, 3, (32, 32), (32, 32)),
        (1, 512, 512, 3, (32, 32), (16, 16)),
    ]
    for B, H, W, C, size, strides in cases:
        rng = np.random.RandomState(0)
        x = rng.rand(B, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(
            x_t, size=size, strides=strides, padding="valid",
        )
        is_overlap = strides != size
        mode = "overlap" if is_overlap else "non-overlap"
        out_size = (
            (H if not is_overlap else (patches.shape[1] - 1) * strides[0] + size[0]),
            (W if not is_overlap else (patches.shape[2] - 1) * strides[1] + size[1]),
        )

        layer = ReconstructPatches2D(
            size=size, output_size=out_size,
            strides=strides, padding="valid",
        )

        t_op = _time(lambda: reconstruct_patches(
            patches, size=size, output_size=out_size,
            strides=strides, padding="valid",
        ))
        t_layer = _time(lambda: layer(patches))

        overhead = (t_layer - t_op) / t_op * 100
        print(f"{str((B, H, W, C)):<25} {str(size):<10} {str(strides):<10} "
              f"{mode:<12} {t_op*1000:<10.2f} {t_layer*1000:<12.2f} "
              f"{overhead:+.0f}%")


def bench_3d():
    print("\n=== 3D ===")
    print(f"{'shape':<28} {'patch':<14} {'stride':<14} {'mode':<12} "
          f"{'op (ms)':<10} {'layer (ms)':<12} {'overhead':<10}")
    print("-" * 105)
    cases = [
        # (B, D, H, W, C, size, strides)
        (1, 64, 128, 128, 1, (8, 16, 16), (8, 16, 16)),    # typical CT chunk, non-overlap
        (1, 64, 128, 128, 1, (8, 16, 16), (4, 8, 8)),      # 50% overlap
        (1, 32, 64, 64, 2, (4, 8, 8), (4, 8, 8)),
        (1, 32, 64, 64, 2, (4, 8, 8), (2, 4, 4)),
    ]
    for B, D, H, W, C, size, strides in cases:
        rng = np.random.RandomState(1)
        x = rng.rand(B, D, H, W, C).astype("float32")
        x_t = ops.convert_to_tensor(x)
        patches = ops.image.extract_patches(
            x_t, size=size, strides=strides, padding="valid",
        )
        is_overlap = strides != size
        mode = "overlap" if is_overlap else "non-overlap"
        out_size = (
            (D if not is_overlap else (patches.shape[1] - 1) * strides[0] + size[0]),
            (H if not is_overlap else (patches.shape[2] - 1) * strides[1] + size[1]),
            (W if not is_overlap else (patches.shape[3] - 1) * strides[2] + size[2]),
        )

        layer = ReconstructPatches3D(
            size=size, output_size=out_size,
            strides=strides, padding="valid",
        )

        t_op = _time(lambda: reconstruct_patches_3d(
            patches, size=size, output_size=out_size,
            strides=strides, padding="valid",
        ))
        t_layer = _time(lambda: layer(patches))

        overhead = (t_layer - t_op) / t_op * 100
        print(f"{str((B, D, H, W, C)):<28} {str(size):<14} {str(strides):<14} "
              f"{mode:<12} {t_op*1000:<10.2f} {t_layer*1000:<12.2f} "
              f"{overhead:+.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", choices=("2d", "3d", "both"), default="both")
    args = parser.parse_args()
    print(f"backend: {keras.backend.backend()}")
    print(f"keras: {keras.__version__}")
    if args.dims in ("2d", "both"):
        bench_2d()
    if args.dims in ("3d", "both"):
        bench_3d()


if __name__ == "__main__":
    main()
