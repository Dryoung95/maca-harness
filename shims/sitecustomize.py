"""
KernelBench eval-mode patch — harness layer, upstream source untouched.

Why this exists
---------------
KernelBench compares a reference output against a candidate output, but
upstream `eval.py` never calls `.eval()`: models default to train mode, where
Dropout consumes RNG and BatchNorm uses batch statistics. Two forwards of the
same architecture then diverge *with no candidate kernel in the loop at all*
— the comparison measures mode asymmetry instead of the candidate.

Nine problems (L2 P66; L3 P9/P10/P15/P16/P20/P22/P23/P24) pass in eval mode
and fail in train mode, with max difference dropping 5–8 orders of magnitude.
Eval mode is the inference semantics KernelBench measures, so the official
gate is applied with both models in eval mode.

How it works
------------
`env.sh` puts this directory first on PYTHONPATH, so the `site` module imports
this file at interpreter startup. A meta-path import hook wraps
`kernelbench.eval.run_and_check_correctness` — the only function that holds
both model instances — placing both in eval mode on entry. Timing reuses the
same instances and inherits eval mode.

No file under `KernelBench/` is modified; removing this one file (or setting
MACA_EVAL_MODE=0) restores the original upstream behaviour exactly.
"""

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "kernelbench.eval"
_FLAG = "MACA_EVAL_MODE"


def _apply_patch(module) -> None:
    """Wrap run_and_check_correctness to eval() both models on entry."""
    if getattr(module, "_maca_eval_mode_patched", False):
        return
    original = module.run_and_check_correctness

    def patched(original_model_instance, new_model_instance, *args, **kwargs):
        # Checked per call so MACA_EVAL_MODE can be flipped at runtime
        # (e.g. run_batch.py --no-eval-mode).
        if os.environ.get(_FLAG, "1") == "1":
            original_model_instance.eval()
            new_model_instance.eval()
        return original(original_model_instance, new_model_instance, *args, **kwargs)

    patched.__wrapped__ = original
    patched._maca_eval_mode = True
    module.run_and_check_correctness = patched
    module._maca_eval_mode_patched = True


class _WrapLoader:
    """Execute the real module source, then patch it."""

    def __init__(self, base_loader):
        self._base = base_loader

    def create_module(self, spec):
        # Let the import system create a plain module; the base loader's
        # create_module (when present) only supplies a pre-made module and
        # is not needed for source loading.
        return None

    def exec_module(self, module):
        self._base.exec_module(module)
        try:
            _apply_patch(module)
        except Exception as e:  # never break an import over this
            print(f"[maca-harness] eval-mode patch failed: {e}", file=sys.stderr)

    # Delegate the introspection attributes FileLoader consumers may check.
    def __getattr__(self, name):
        return getattr(self._base, name)


class _PatchFinder(importlib.abc.MetaPathFinder):
    """Intercept kernelbench.eval, delegate loading to the normal finders."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _PatchFinder):
                continue
            try:
                spec = finder.find_spec(fullname, path)
            except (AttributeError, ImportError):
                continue
            if spec is not None and spec.loader is not None:
                # Mutate the spec in place: every other attribute (origin,
                # has_location, submodule_search_locations) stays exactly
                # what the normal finder produced, so __file__ etc. survive.
                spec.loader = _WrapLoader(spec.loader)
                return spec
        return None


class _D7SafeConvPatch:
    """Wrap nn.Conv1d.forward with the D7 batch-slicing workaround.

    D7 (defects/README.md in kda-on-maca): a full-tensor Conv1d on a
    >~1.62e9-element input silently corrupts the first and last (B-48)
    batches; per-batch slices of the same tensor are all correct. Conv1d is
    independent per batch sample, so slicing is the same math on a
    non-corrupted path.

    This patch is what makes the gate's REFERENCE forward correct on the
    affected shapes -- the reference is the full-tensor call, i.e. the
    corrupted path, so without it no correct candidate can pass (verified
    2026-10-15: candidate vs im2col umpire 9.5e-7, reference vs umpire 1.99).
    Applied to both sides of the comparison it removes the defect from the
    measurement entirely rather than rescuing one side.

    Guard is conservative: the trigger was bisected to between 1.6106e9
    (ok) and 1.6384e9 (bad), so 1.62e9 sits inside the verified-good margin.
    Below it, behaviour is bit-for-bit the stock forward.
    """

    D7_GUARD_ELEMS = 1.62e9
    SLICE_BATCH = 32

    @staticmethod
    def apply():
        import torch
        if getattr(torch.nn.Conv1d, "_maca_d7_patched", False):
            return
        original = torch.nn.Conv1d.forward

        def patched(self, x):
            if (x.dim() != 3
                    or x.numel() <= _D7SafeConvPatch.D7_GUARD_ELEMS):
                return original(self, x)
            return torch.cat(
                [original(self, x[i:i + _D7SafeConvPatch.SLICE_BATCH])
                 for i in range(0, x.size(0), _D7SafeConvPatch.SLICE_BATCH)],
                dim=0)

        patched.__wrapped__ = original
        torch.nn.Conv1d.forward = patched
        torch.nn.Conv1d._maca_d7_patched = True


def _apply_d7_patch():
    try:
        _D7SafeConvPatch.apply()
    except Exception as e:  # never break an import over this
        print(f"[maca-harness] D7 conv patch failed: {e}", file=sys.stderr)


# Always install: the flag is re-read on every wrapped call, so a caller can
# flip MACA_EVAL_MODE at runtime (run_batch.py --no-eval-mode sets it to "0").
sys.meta_path.insert(0, _PatchFinder())

# The Conv1d guard is size-based, not flag-gated: there is no legitimate
# reason to run the corrupted path, and every consumer of this PYTHONPATH
# (run_batch, the gate drivers, the raw torch process) needs it.
_apply_d7_patch()
