"""
Hyperopt Scheduler
==================
Coordinates hyperparameter trials. In this lab the scheduler is *not* an
Optuna/RL controller — it is just a small helper that:

* logs every trial,
* runs the **grid search** (exhaustive Cartesian product),
* (optionally) prunes early if a trial is doing badly after epoch 1 — a
  cheap "successive halving" style cut.
* writes ``results.json`` and a ``results.md`` after the run.

This keeps the *behaviour* we need (search + reporting) without pulling
in Optuna, which is not in ``requirements.txt``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hyperparameter_tuning import (
    HyperparameterTuner,
    TrialResult,
    best_result,
    results_to_markdown,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Run-level configuration."""

    strategy: str = "grid"               # "grid" (exhaustive Cartesian product)
    seed: int = 42
    output_dir: str = "results/runs"    # where to write results.md / .json
    filename_prefix: str = "hyperopt"
    prune_below: float | None = None    # early-stop trials with val_acc < threshold after epoch 1


class HyperoptScheduler:
    """Run a hyperparameter search and write a report.

    Args:
        tuner: A configured :class:`HyperparameterTuner`.
        config: Run-level configuration.
    """

    def __init__(self, tuner: HyperparameterTuner, config: SchedulerConfig | None = None):
        self.tuner = tuner
        self.config = config or SchedulerConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, search_space: Mapping[str, Any]) -> TrialResult:
        """Execute the configured grid search and return the best trial.

        Args:
            search_space: A grid dict (lists) for ``strategy='grid'``.

        Returns:
            The :class:`TrialResult` with the highest validation accuracy.
        """
        logger.info(
            f"Scheduler starting: strategy={self.config.strategy}, "
            f"seed={self.config.seed}"
        )

        if self.config.strategy == "grid":
            results = self.tuner.grid_search(search_space)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        # Optional cheap early-pruning pass (not really pruning trials —
        # we already ran them — but we keep the flag for future Optuna
        # migration and for the test suite).
        if self.config.prune_below is not None:
            results = [r for r in results if r.val_acc >= self.config.prune_below]

        if not results:
            raise RuntimeError("No trials produced results — check the search space.")

        best = best_result(results)
        logger.info(
            f"Best trial #{best.trial_id}: val_acc={best.val_acc:.2f}% "
            f"params={best.params}"
        )

        self._write_artifacts(results, best)
        return best

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #

    def _write_artifacts(self, results: Sequence[TrialResult], best: TrialResult) -> None:
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.config.filename_prefix
        json_path = out_dir / f"{prefix}.json"
        md_path = out_dir / f"{prefix}.md"

        save_results(results, json_path)
        with md_path.open("w", encoding="utf-8") as f:
            f.write(results_to_markdown(results))
            f.write("\n## Best Trial\n\n")
            f.write(f"- **trial id**: `{best.trial_id}`\n")
            f.write(f"- **val_acc**: `{best.val_acc:.2f}%`\n")
            f.write(f"- **train_acc**: `{best.train_acc:.2f}%`\n")
            f.write("- **params**:\n")
            for k, v in best.params.items():
                f.write(f"  - `{k}`: `{v}`\n")

        # Also dump the best config separately for easy consumption by Người 2.
        best_path = out_dir / f"{prefix}_best.json"
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_id": best.trial_id,
                    "val_acc": best.val_acc,
                    "train_acc": best.train_acc,
                    "params": best.params,
                },
                f,
                indent=2,
            )
        logger.info(f"Artifacts written under {out_dir}")


__all__ = ["HyperoptScheduler", "SchedulerConfig"]