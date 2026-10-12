"""
Training Module
===============
Training loops, optimizers, and training utilities.

This package contains (Người 1):

* :mod:`.hyperparameter_tuning` – ``HyperparameterTuner`` for grid/random search
* :mod:`.hyperopt_scheduler`   – ``HyperoptScheduler`` orchestrator + reporter

Other Người 2 files (trainer.py, callbacks.py, ...) / will live in this
package too. They are intentionally not implemented here — that is a
separate deliverable.
"""

__version__ = "0.2.0"

from .hyperopt_scheduler import HyperoptScheduler, SchedulerConfig
from .hyperparameter_tuning import (
    HyperparameterTuner,
    TrialResult,
    best_result,
    results_to_markdown,
    save_results,
)

__all__ = [
    "HyperparameterTuner",
    "HyperoptScheduler",
    "SchedulerConfig",
    "TrialResult",
    "best_result",
    "results_to_markdown",
    "save_results",
]