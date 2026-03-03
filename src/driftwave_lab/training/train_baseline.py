"""Training loop for baseline models (thin wrapper around :func:`train_fno.train`).

The unified training function works for any model registered in the factory,
so this module simply re-exports it for discoverability.

Example
-------
>>> from driftwave_lab.training.train_baseline import train
>>> history = train(cfg)
"""

from driftwave_lab.training.train_fno import train

__all__ = ["train"]
