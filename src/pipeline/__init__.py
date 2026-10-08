"""RAG pipeline package.

The main pipeline imports optional heavyweight ML dependencies.  Keep the
package itself lightweight so retry and evaluation helpers can be tested on a
machine that has not installed a CUDA runtime.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.rag_pipeline import run_rag

__all__ = ["run_rag"]


def __getattr__(name):
    if name == "run_rag":
        from src.pipeline.rag_pipeline import run_rag

        return run_rag
    raise AttributeError("module {0!r} has no attribute {1!r}".format(__name__, name))
