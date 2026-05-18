# Evaluation test suite

Exhaustive correctness tests for the patch extract/reconstruct pipeline.

## Layout

- `test_roundtrip_2d.py` — parametric grid of 2D extract→reconstruct cases.
  Covers symmetric/asymmetric/degenerate patch sizes × multiple residue classes
  for `padding="same"`. Sweeps channels and batch sizes independently.
- `test_roundtrip_3d.py` — same for 3D, with a smaller per-patch grid because
  5D tensors get expensive fast.
- `test_edge_cases.py` — degenerate inputs (input == patch, patch size 1,
  input smaller than patch on an axis) and real-world scenarios (ViT
  patching, medical-imaging volumes, MNIST/CIFAR-style classification).
- `test_layer_and_serialization.py` — `get_config`/`from_config` roundtrip,
  Functional API integration, `compute_output_shape` correctness, and the
  full set of `ValueError`/`NotImplementedError` paths.

## Running

From the repo root:

```bash
# One backend
KERAS_BACKEND=tensorflow pytest tests/ -v

# All three backends
for backend in tensorflow jax torch; do
    KERAS_BACKEND=$backend pytest tests/ -v
done

# Just the basic / fast tests
pytest tests/test_layer_and_serialization.py -v
```

On Windows PowerShell:

```powershell
$env:KERAS_BACKEND = "tensorflow"; pytest tests/ -v
$env:KERAS_BACKEND = "jax";        pytest tests/ -v
$env:KERAS_BACKEND = "torch";      pytest tests/ -v
```

## What "passing" proves

Together with the existing `test_reconstruct_patches.py` at the repo root,
green CI on all three backends proves:

1. `reconstruct_patches(extract_patches(x)) == x` for every patch size and
   spatial-dimension class the API supports (non-overlapping, channels_last).
2. The Layer wrappers produce identical output to the underlying ops.
3. `get_config`/`from_config` round-trips losslessly.
4. Invalid arguments raise informative errors at construction time, not at
   call time.
