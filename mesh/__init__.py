"""mesh -- coordination layer for distributed training on volunteer hardware.

Prototype. See docs/PRODUCTION-READINESS.md for what is real and what is not.
"""
from .types import JobSpec, NodeProfile, Placement, Verdict

__version__ = "0.1.0"
__all__ = ["JobSpec", "NodeProfile", "Placement", "Verdict"]
