"""Minimal stand-in for the `openai` package.

KernelBench's `kernelbench/utils.py` imports `from openai import OpenAI` at module
level. That import is only exercised by the LLM-kernel-generation path
(`query_server`); the correctness/timing harness never constructs a client.

The real package is unavailable in this offline environment (PyPI is
unreachable), and modifying upstream KernelBench sources is out of scope for this
port. This shim therefore satisfies the import with a stub that fails loudly if
any code path actually tries to talk to an LLM API, rather than silently
returning garbage.

It is kept deliberately dependency-free and stateless.
"""

__all__ = ["OpenAI"]


class _Unavailable:
    """Raised on any attempt to use the stubbed-out API client."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The `openai` package is not installed in this environment, and no "
            "LLM API is reachable. KernelBench's LLM generation path is "
            "unavailable; the correctness/timing harness does not require it."
        )

    def __getattr__(self, name):
        raise RuntimeError(
            f"openai.{name} is unavailable: the `openai` package is not "
            "installed in this offline environment."
        )


OpenAI = _Unavailable
