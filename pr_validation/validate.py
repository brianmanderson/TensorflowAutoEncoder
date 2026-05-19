"""Validation harness for upstream PR branches.

Run with the keras checkout already on the right `pr/NN-name` branch:

    python pr_validation/validate.py 1                 # all backends
    python pr_validation/validate.py 2 --backend torch # single backend
    python pr_validation/validate.py 3 --skip-tensorflowwork

Exits non-zero if any test fails. See pr_validation/README.md for details.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Project roots — adjust if you've moved either repo.
TFW_ROOT = Path(__file__).resolve().parent.parent
KERAS_ROOT = TFW_ROOT.parent / "keras"
PYTHON = sys.executable


# For each upstream PR, list:
# - keras_tests: focused test files in the keras checkout that should pass
# - tfw_tests: substantial-suite files in tensorflowwork that should pass
#              (subset selected by which features the PR implements)
PR_TESTS: dict[int, tuple[list[str], list[str]]] = {
    1: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches_3d",
        ],
        # Tests covering the basic non-overlap path only.
        [
            "tests/test_roundtrip_2d.py::test_2d_roundtrip",
            "tests/test_roundtrip_3d.py::test_3d_roundtrip",
            "tests/test_edge_cases.py::test_2d_input_equals_patch",
            "tests/test_edge_cases.py::test_3d_input_equals_patch",
            "tests/test_edge_cases.py::test_2d_patch_size_one",
            "tests/test_edge_cases.py::test_3d_patch_size_one",
        ],
    ),
    2: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches_3d",
        ],
        # PR 1 tests plus the overlap and reduction tests.
        [
            "tests/test_roundtrip_2d.py::test_2d_roundtrip",
            "tests/test_roundtrip_3d.py::test_3d_roundtrip",
            "tests/test_overlapping_strides.py",
            "tests/test_reduction.py",
        ],
    ),
    3: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        # Add channels_first tests (auto-skipped on TF backend).
        [
            "tests/test_roundtrip_2d.py::test_2d_roundtrip",
            "tests/test_channels_first.py",
        ],
    ),
    4: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        # Adds dilation tests (auto-skipped on TF backend).
        [
            "tests/test_overlapping_strides.py",
            "tests/test_dilation.py",
        ],
    ),
    5: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        # Adds auto-infer + better-errors tests (subset of layer_and_serialization).
        [
            "tests/test_layer_and_serialization.py",
        ],
    ),
    6: (
        [
            "keras/src/layers/reshaping/extract_patches2d_test.py",
            "keras/src/layers/reshaping/extract_patches3d_test.py",
        ],
        # Pure ExtractPatches Layer tests.
        [
            "tests/test_extract_layers.py",
        ],
    ),
    7: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        # Dual-input dynamic output_size.
        [
            "tests/test_dynamic_output_size.py",
        ],
    ),
}

BACKENDS = ("tensorflow", "jax", "torch")


def fail(msg: str) -> None:
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"\n=== {msg} ===")


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return proc.returncode


def assert_branch_matches_pr(pr: int) -> None:
    """Refuse to run if the keras checkout isn't on the expected branch."""
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(KERAS_ROOT), capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(f"could not determine current branch in {KERAS_ROOT}")
    branch = result.stdout.strip()
    expected_prefix = f"pr/{pr:02d}-"
    if not branch.startswith(expected_prefix):
        fail(
            f"keras checkout is on branch {branch!r}; expected something "
            f"starting with {expected_prefix!r}. "
            f"`cd {KERAS_ROOT} && git checkout pr/{pr:02d}-...` first."
        )


def build_env(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    # Make `import keras` hit the local checkout, and `from reconstruct_patches
    # import ...` resolve to tensorflowwork's local prototype.
    env["PYTHONPATH"] = os.pathsep.join([
        str(KERAS_ROOT),
        str(TFW_ROOT),
        env.get("PYTHONPATH", ""),
    ])
    return env


def run_pytest(
    cwd: Path, env: dict[str, str], targets: list[str], label: str,
) -> bool:
    info(f"{label} in {cwd.name}")
    if not targets:
        print("(no targets — skipped)")
        return True
    rc = run([PYTHON, "-m", "pytest", *targets, "-v", "--tb=short", "-q"], cwd, env)
    if rc != 0:
        print(f"FAILED: {label}", file=sys.stderr)
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pr", type=int, choices=sorted(PR_TESTS),
                   help="Upstream PR number (1-7)")
    p.add_argument("--backend", choices=BACKENDS + ("all",), default="all",
                   help="Backend to validate on (default: all three)")
    p.add_argument("--skip-tensorflowwork", action="store_true",
                   help="Skip the substantial-suite tests; focused only.")
    p.add_argument("--skip-keras-checkout", action="store_true",
                   help="Skip the focused keras-checkout tests.")
    args = p.parse_args()

    assert_branch_matches_pr(args.pr)
    keras_tests, tfw_tests = PR_TESTS[args.pr]

    backends = BACKENDS if args.backend == "all" else (args.backend,)
    all_ok = True
    failures: list[str] = []

    for backend in backends:
        info(f"BACKEND: {backend}")
        env = build_env(backend)
        if not args.skip_keras_checkout:
            ok = run_pytest(
                KERAS_ROOT, env, keras_tests,
                f"focused keras-checkout tests ({backend})",
            )
            if not ok:
                all_ok = False
                failures.append(f"{backend}: keras-checkout tests")
        if not args.skip_tensorflowwork:
            ok = run_pytest(
                TFW_ROOT, env, tfw_tests,
                f"tensorflowwork substantial suite subset ({backend})",
            )
            if not ok:
                all_ok = False
                failures.append(f"{backend}: tensorflowwork tests")

    print()
    if all_ok:
        info(f"ALL VALIDATION PASSED for PR {args.pr}")
        print("Safe to push and open the upstream PR.")
        return 0
    print(f"VALIDATION FAILED for PR {args.pr}:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    print(
        "\nDo NOT push. See docs/UPSTREAM_PR_PROCESS.md for debugging tips.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
