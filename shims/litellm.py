"""Minimal stand-in for the `litellm` package.

KernelBench's `kernelbench/utils.py` imports `from litellm import completion` at
module level. That symbol is only called from `query_server`, the LLM-kernel-
generation path; the correctness/timing harness never invokes it.

The real package is unavailable in this offline environment (PyPI is
unreachable), and modifying upstream KernelBench sources is out of scope for this
port. This shim satisfies the import with a stub that raises a clear error if any
code path actually attempts a completion request.
"""

__all__ = ["completion"]


def completion(*args, **kwargs):
    raise RuntimeError(
        "litellm.completion() is unavailable: the `litellm` package is not "
        "installed in this offline environment, and no LLM API is reachable. "
        "KernelBench's LLM generation path cannot run here; the correctness and "
        "timing harness does not require it."
    )
