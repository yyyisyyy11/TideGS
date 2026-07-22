"""TideGS release strategy.

The public release entry is ``train_tidegs.py``.  The release runtime
enters the TideGS runtime path through ``runtime.py`` and
``TideStorageAdapter``.
"""

__all__ = [
    "TideGaussianModel",
    "train_tide_batch",
    "validate_tide_runtime_args",
]


def __getattr__(name):
    if name == "TideGaussianModel":
        from .gaussian_model import TideGaussianModel

        return TideGaussianModel
    if name in {"train_tide_batch", "validate_tide_runtime_args"}:
        from .runtime import train_tide_batch, validate_tide_runtime_args

        return {
            "train_tide_batch": train_tide_batch,
            "validate_tide_runtime_args": validate_tide_runtime_args,
        }[name]
    raise AttributeError(name)
