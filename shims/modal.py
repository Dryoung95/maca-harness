"""Minimal stand-in for the `modal` package, sufficient for KernelBench's
``scripts/run_and_check.py`` in ``eval_mode=local``.

The real modal (a cloud-function runner) is unavailable in this offline
environment (PyPI unreachable), and KernelBench's upstream sources must not be
modified. run_and_check.py imports modal at module level and constructs
``modal.App(...)`` / decorates a class with ``@app.cls(...)`` even when running
locally; those decorated methods are only invoked through Modal's remote
machinery, which is never reached in ``eval_mode=local``.

This stub therefore makes construction and decoration no-ops that return the
callable unchanged, and raises loudly if anything actually attempts a remote
call — rather than silently pretending to succeed.
"""

__all__ = ["App", "Image", "method"]


class Image:
    """modal.Image stub: run_and_check.py builds a container image at import
    time via a chained builder API (from_registry().apt_install()...). The
    resulting image is only used for remote evaluation, so each builder method
    returns self unchanged."""

    def __init__(self):
        pass

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def uv_sync(self, *args, **kwargs):
        return self

    def run_commands(self, *args, **kwargs):
        return self

    def env(self, *args, **kwargs):
        return self

    def add_local_dir(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class _Unavailable:
    """Rejects any attempt to actually invoke a remote Modal resource."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Modal cloud evaluation is unavailable in this offline environment. "
            "Use eval_mode=local instead."
        )

    def __getattr__(self, name):
        raise RuntimeError(
            f"modal.{name} is unavailable: the `modal` package is not installed. "
            "Use eval_mode=local instead."
        )


def _identity(*args, **kwargs):
    # app.cls(...) decorates a class; app.method() decorates a method. Both are
    # only reached via Modal's remote machinery, so returning the target
    # unchanged is sufficient for local evaluation.
    def deco(target):
        return target

    return deco


def method(*args, **kwargs):
    """modal.method() decorates remote-callable methods; a no-op locally."""
    def deco(target):
        return target
    return deco


class App:
    """modal.App stub: supports only the local-mode construction surface."""

    def __init__(self, *args, **kwargs):
        pass

    # ``with app.run():`` would open a remote session.
    def run(self):
        return _Unavailable()

    # make the stub falsy so ``with app.run():`` (if ever attempted) fails
    # loudly through _Unavailable above rather than silently continuing.
    def __bool__(self):
        return True

    def __getattr__(self, name):
        # app.cls(...) and any other builder API
        return _identity


App.with_options = staticmethod(lambda **kwargs: App())
