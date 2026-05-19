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
# Each PR maps to (keras_checkout_tests, direct_tests) where:
#   keras_checkout_tests = focused test files inside the keras checkout
#                          that exercise the upstream PR's code directly
#   direct_tests = pr_validation/direct_tests/test_pr*.py files that
#                  import `keras.layers.X` / `keras.ops.image.X` directly
#                  (no dependency on tensorflowwork's local prototype).
#                  Cumulative — PR N runs direct tests for PRs 1..N.
PR_TESTS: dict[int, tuple[list[str], list[str]]] = {
    1: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches_3d",
        ],
        ["pr_validation/direct_tests/test_pr01_basic.py"],
    ),
    2: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches",
            "keras/src/ops/image_test.py::ImageOpsDynamicShapeTest::test_reconstruct_patches_3d",
        ],
        [
            "pr_validation/direct_tests/test_pr01_basic.py",
            "pr_validation/direct_tests/test_pr02_overlap.py",
        ],
    ),
    3: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        [
            "pr_validation/direct_tests/test_pr01_basic.py",
            "pr_validation/direct_tests/test_pr03_channels_first.py",
        ],
    ),
    4: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        [
            "pr_validation/direct_tests/test_pr01_basic.py",
            "pr_validation/direct_tests/test_pr02_overlap.py",
            "pr_validation/direct_tests/test_pr04_dilation.py",
        ],
    ),
    5: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        [
            "pr_validation/direct_tests/test_pr01_basic.py",
            "pr_validation/direct_tests/test_pr05_ergonomics.py",
        ],
    ),
    6: (
        [
            "keras/src/layers/reshaping/extract_patches2d_test.py",
            "keras/src/layers/reshaping/extract_patches3d_test.py",
        ],
        ["pr_validation/direct_tests/test_pr06_extract_layers.py"],
    ),
    7: (
        [
            "keras/src/layers/reshaping/reconstruct_patches2d_test.py",
            "keras/src/layers/reshaping/reconstruct_patches3d_test.py",
        ],
        [
            "pr_validation/direct_tests/test_pr01_basic.py",
            "pr_validation/direct_tests/test_pr07_dynamic_output.py",
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
    p.add_argument("--skip-direct-tests", action="store_true",
                   help="Skip the direct_tests suite; run focused tests only.")
    p.add_argument("--skip-tensorflowwork", action="store_true",
                   help="Alias for --skip-direct-tests (legacy flag name).")
    p.add_argument("--skip-keras-checkout", action="store_true",
                   help="Skip the focused keras-checkout tests.")
    args = p.parse_args()

    assert_branch_matches_pr(args.pr)
    keras_tests, direct_tests = PR_TESTS[args.pr]
    skip_direct = args.skip_direct_tests or args.skip_tensorflowwork

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
        if not skip_direct:
            ok = run_pytest(
                TFW_ROOT, env, direct_tests,
                f"direct_tests against installed keras ({backend})",
            )
            if not ok:
                all_ok = False
                failures.append(f"{backend}: direct_tests")

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
