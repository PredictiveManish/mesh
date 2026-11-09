"""Test bootstrap.

mesh.protocol imports torch at module level for the tensor codec, but the
framing logic under test here (length validation, pack/unpack) is pure
Python. When torch is not installed, a stub keeps those tests runnable; the
tests that genuinely need torch (test_smoke.py) still require it.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # pragma: no cover - environment dependent
    import torch  # noqa: F401
except ImportError:
    sys.modules.setdefault("torch", types.ModuleType("torch"))
