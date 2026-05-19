# PR validation harness

`validate.py` is the gate for any upstream PR branch. It runs both:

1. The **focused keras-checkout tests** for the PR's branch (the small set
   of tests we wrote that live in `keras/src/layers/reshaping/...` and
   `keras/src/ops/image_test.py`).
2. The **substantial tensorflowwork tests** for the features available in
   this PR (selected via the `PR_TESTS` dict at the top of `validate.py`).

Under all three backends (TF, JAX, PyTorch) unless overridden with
`--backend`.

## Quick start

```bash
# Validate PR 1 (assumes the keras checkout is on pr/01-basic-reconstruct)
python pr_validation/validate.py 1

# Validate PR 2 on PyTorch only (faster iteration)
python pr_validation/validate.py 2 --backend torch

# Skip the substantial suite (focused tests only — fastest)
python pr_validation/validate.py 3 --skip-tensorflowwork
```

Exits non-zero if anything fails. **Don't push the branch upstream if
validation fails.**

## How it works

For each backend:

1. Sets `KERAS_BACKEND=<backend>`.
2. Runs `pytest <focused tests>` in the keras checkout. These tests live
   alongside the upstream code and exercise it directly.
3. Sets `PYTHONPATH` to include both the keras checkout (so `import keras`
   hits the checkout, not pip's site-packages) and the tensorflowwork
   directory (so `from reconstruct_patches import ...` finds the local
   prototype that the tensorflowwork tests use).
4. Runs `pytest <tensorflowwork test subset>`. Tests for features the
   current PR branch doesn't yet support are either selected out via the
   `PR_TESTS` mapping or skipped via existing `pytest.mark.skipif`
   markers (channels_first on TF, dilation on TF, etc.).

## Adding a new PR

Edit `PR_TESTS` in `validate.py`:

```python
PR_TESTS = {
    ...
    8: (
        ["keras/src/.../my_new_test.py"],
        ["tests/my_new_feature_test.py"],
    ),
}
```

Then `python pr_validation/validate.py 8` works.

## Why bother running the tensorflowwork tests too?

The keras-checkout tests for each PR are focused (10–25 cases). The
tensorflowwork tests are 465+ cases across shape grids, dtypes, edge
cases, real-world ViT/medical shapes, save/load round-trips, gradient
flow, mixed precision, JIT compilation, etc. Running the relevant
subset against each PR branch ensures we don't accidentally break the
substantial behaviors we validated locally when slicing them into
upstream-sized chunks.

If a tensorflowwork test fails on a PR branch but passes on the local
prototype, that's the bug.
