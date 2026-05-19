"""Root conftest — ensures `reconstruct_patches.py` is importable from tests/.

Pytest auto-discovers this file at the rootdir and runs it before tests are
collected, so any test file in tests/ can `from reconstruct_patches import ...`
regardless of the cwd pytest is invoked from.

Also enables JAX float64 support before keras is imported. JAX defaults to
"x32 mode" which silently downcasts float64 inputs to float32 — without this
flag, the dtype-preservation tests would fail under the JAX backend.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if os.environ.get("KERAS_BACKEND") == "jax":
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        pass
