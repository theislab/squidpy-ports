"""Reference outputs for the JAX STalign port in squidpy.

The upstream PyTorch implementation is vendored at ``vendor/STalign`` and pinned to a
single commit. This package runs it, unmodified, and writes the numbers squidpy's
``tests/experimental/tl/test_align_stalign.py`` compares against.
"""

from . import fixtures, upstream

__all__ = ["fixtures", "upstream"]
