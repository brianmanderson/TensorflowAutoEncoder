"""Root conftest — ensures `reconstruct_patches.py` is importable from tests/.

Pytest auto-discovers this file at the rootdir and runs it before tests are
collected, so any test file in tests/ can `from reconstruct_patches import ...`
regardless of the cwd pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
