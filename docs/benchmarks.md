# Benchmarks

Wall-clock per call on `ReconstructPatches{2,3}D` and the bare ops.
Numbers vary by machine and backend; treat them as relative comparisons,
not absolute floors. Run `benchmarks/benchmark_reconstruct.py` on your own
hardware for accurate numbers.

## Setup

- CPU: Windows dev machine (representative consumer hardware)
- Backend: TensorFlow 2.21 (CPU)
- keras 3.14.1
- Median of 20 iterations after 3 warmup calls

## Results

### 2D

| Input shape | Patch | Stride | Mode | op (ms) | Layer (ms) | Layer overhead |
|---|---|---|---|---|---|---|
| (1, 224, 224, 3) | (16, 16) | (16, 16) | non-overlap | 0.65 | 0.82 | +27% |
| (1, 224, 224, 3) | (16, 16) | (8, 8) | overlap (2x) | 22.53 | 21.39 | -5% |
| (1, 224, 224, 3) | (16, 16) | (1, 1) | overlap (max) | 5906.74 | 5955.99 | +1% |
| (1, 512, 512, 3) | (32, 32) | (32, 32) | non-overlap | 2.44 | 2.84 | +16% |
| (1, 512, 512, 3) | (32, 32) | (16, 16) | overlap (2x) | 172.69 | 197.81 | +15% |

### 3D

| Input shape | Patch | Stride | Mode | op (ms) | Layer (ms) | Layer overhead |
|---|---|---|---|---|---|---|
| (1, 64, 128, 128, 1) | (8, 16, 16) | (8, 16, 16) | non-overlap | 2.11 | 2.62 | +24% |
| (1, 64, 128, 128, 1) | (8, 16, 16) | (4, 8, 8) | overlap (2x) | 234.18 | 241.34 | +3% |
| (1, 32, 64, 64, 2) | (4, 8, 8) | (4, 8, 8) | non-overlap | 0.92 | 1.08 | +17% |
| (1, 32, 64, 64, 2) | (4, 8, 8) | (2, 4, 4) | overlap (2x) | 51.38 | 50.83 | -1% |

## Observations

1. **Non-overlap is 30–100× faster than overlap.** The non-overlap path is
   pure `reshape → transpose → slice` (no math); the overlap path runs two
   `conv_transpose` calls (one for the sum, one for the count) plus a divide.
   If you don't *need* overlapping patches, don't use them — the bare
   reshape path is essentially free.

2. **Layer overhead is small** (5–25%) and mostly fixed cost
   (`compute_output_shape` symbolic checks, Python attribute lookups).
   In a training loop where `model.fit()` JIT-compiles the call site,
   this overhead is amortized to zero.

3. **Stride = 1 (sliding window) is expensive at large patches.** The
   `224 × 224` × `16 × 16` × stride-1 case takes ~6 seconds per call.
   For sliding-window inference workloads, plan accordingly — batch your
   inputs, use GPU, or consider a tiled approach with a small overlap rather
   than dense stride-1.

4. **3D scales as you'd expect.** A 64×128×128 CT-scan chunk reconstructs
   in ~2ms (non-overlap) or ~240ms (50% overlap), proportional to the patch
   count.

## Recommendations

- **Default to non-overlap** unless you specifically need overlap (sliding
  window, super-resolution, smooth seams in segmentation).
- For sliding-window inference, **batch as many input crops as your GPU
  memory allows** rather than running one at a time.
- The Layer-vs-op overhead is small enough that you should prefer the
  Layer in any Functional / Sequential context for the API ergonomics.
- For training, `model.compile(jit_compile=True)` removes most layer
  overhead via backend-native JIT.
