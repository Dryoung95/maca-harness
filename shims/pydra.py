"""Minimal stand-in for the `pydra` package, sufficient for KernelBench's
``scripts/run_and_check.py`` in ``eval_mode=local``.

The real pydra is unavailable in this offline environment (PyPI unreachable),
and KernelBench's upstream sources must not be modified. run_and_check.py uses
only a small surface of pydra:

    from pydra import REQUIRED, Config
    @pydra.main(base=ScriptConfig)
    def main(config: ScriptConfig): ...

pydra's CLI convention is ``key=value`` overrides appended to argv, where the
value is a Python literal (``level=1``, ``gpu_arch=['Ada']``, ``verbose=True``)
or a bare string (``ref_origin=kernelbench``). This shim reproduces that: it
builds a config instance from the ``base`` class, applies argv overrides, and
fails with a clear message if a REQUIRED field is left unset.

Only the pieces run_and_check.py touches are implemented; this is not a general
pydra replacement.
"""

import ast
import sys

__all__ = ["REQUIRED", "Config", "main"]


class _Required:
    """Sentinel marking a config field the caller must supply."""

    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


class Config:
    """Base class for pydra-style config objects."""

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def __repr__(self):
        return f"Config({self.to_dict()})"


def _parse_value(raw):
    """Parse a ``key=value`` string: Python literal if possible, else a bare string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def main(base=None):
    """Decorator reproducing ``@pydra.main(base=ConfigClass)`` for local evaluation."""
    if base is None:
        raise ValueError("pydra.main requires base=ConfigClass")

    def decorator(fn):
        def wrapper():
            config = base()
            overrides = {}
            for arg in sys.argv[1:]:
                if "=" not in arg:
                    raise ValueError(
                        f"pydra shim: expected key=value arguments, got {arg!r}"
                    )
                key, _, raw = arg.partition("=")
                if not hasattr(config, key):
                    raise ValueError(
                        f"pydra shim: {key!r} is not a field of {base.__name__}; "
                        f"known fields: {sorted(vars(config))}"
                    )
                overrides[key] = _parse_value(raw)

            missing = [
                k
                for k, v in vars(config).items()
                if isinstance(v, _Required) and k not in overrides
            ]
            if missing:
                raise ValueError(
                    f"pydra shim: required field(s) {missing} not set; "
                    "pass them as key=value command-line arguments"
                )
            for k, v in overrides.items():
                setattr(config, k, v)
            return fn(config)

        return wrapper

    return decorator
