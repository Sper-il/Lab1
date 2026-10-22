"""
run_experiments.py
==================
End-to-end script for **Người 1**: Build Network + Hyperparameter Experiment.

What it runs (in order)
-----------------------
1. Load the FashionMNIST loaders from ``src.data.split`` (Phase 1).
2. Run :func:`src.models.model_selection.compare_architectures` with a
   fixed mini-config to pick the most promising architecture family.
3. Run :class:`src.training.HyperparameterTuner` with the search space
   from ``configs/hyperopt_config.yaml`` (exhaustive grid search).
4. Print the best (model, hyperparameters) to stdout **and** save the
   per-trial results under ``results/runs/``.

Why a separate file? ``main.py`` belongs to Người 2; this file is
deliberately the Người 1 entry point so that the two deliverables don't
collide.

Usage
-----
::

    # full search (default = 16 grid trials, 10 epochs each)
    python run_experiments.py

    # quick smoke test
    python run_experiments.py --smoke

    # fewer epochs
    python run_experiments.py --epochs 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Make `src` importable when running from the repo root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from src.models.model_factory import ModelFactory  # noqa: E402
from src.models.model_selection import (  # noqa: E402
    compare_architectures,
    results_to_markdown as arch_results_to_md,
)
from src.training import HyperoptScheduler, SchedulerConfig  # noqa: E402
from src.training.hyperparameter_tuning import (  # noqa: E402
    HyperparameterTuner,
    TrialResult,
    best_result,
    results_to_markdown,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_experiments")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_loaders(batch_size: int = 64):
    """Use the existing Phase 1 split pipeline (50k train / 10k val / 10k test)."""
    from src.data.split import DEFAULT_VAL_RATIO, split_pipeline

    train_loader, val_loader, test_loader, _ = split_pipeline(
        data_dir="./data/processed",
        batch_size=batch_size,
        val_ratio=DEFAULT_VAL_RATIO,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def _select_search_space(cfg: dict, strategy: str) -> dict:
    block = "grid_search"
    if block not in cfg:
        raise KeyError(f"configs/hyperopt_config.yaml missing '{block}' block")
    return cfg[block]


# ---------------------------------------------------------------------------
# Smoke test (used by pytest too)
# ---------------------------------------------------------------------------


def smoke_test() -> dict:
    """Quick verification that all 4 architectures work end-to-end.

    Returns:
        A dict ``{model_name: (logits_shape_ok, loss_decreases)}``.
    """
    import torch.nn.functional as F

    device = torch.device("cpu")
    # Use a small enough lr that loss is *guaranteed* to go down on a single
    # mini-batch (any sensible architecture will satisfy this).
    lr = 1e-3
    results = {}
    for name in ModelFactory.list_models():
        model = ModelFactory.create(name).to(device)
        # Use a larger batch + more steps so loss decrease is meaningful
        # (4-sample batches make BatchNorm statistics too noisy for deep_mlp).
        x1 = torch.randn(64, 1, 28, 28, device=device)
        x2 = torch.randn(64, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        logits = model(x1)
        ok_shape = tuple(logits.shape) == (64, 10)
        loss1 = F.cross_entropy(logits, y).item()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(5):
            for xb in (x1, x2):
                opt.zero_grad()
                F.cross_entropy(model(xb), y).backward()
                opt.step()
        loss2 = F.cross_entropy(model(x1), y).item()
        ok_loss = loss2 < loss1
        results[name] = (ok_shape, ok_loss, loss1, loss2)
        logger.info(
            f"[smoke] {name}: shape_ok={ok_shape}, loss {loss1:.4f} -> {loss2:.4f} "
            f"({'OK' if ok_loss else 'FAIL'})"
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Người 1 — Build Network + Hyperparameter Experiment")
    parser.add_argument("--config", default="configs/hyperopt_config.yaml",
                        help="Path to the hyperopt yaml config.")
    parser.add_argument("--strategy", choices=["grid"], default="grid",
                        help="Search strategy (only 'grid' is supported).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs_per_trial.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for the data loaders (informational).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the architecture smoke test, then exit.")
    parser.add_argument("--skip-architecture-selection", action="store_true",
                        help="Skip the compare_architectures step.")
    args = parser.parse_args()

    # 1. Smoke test ---------------------------------------------------- #
    logger.info("Running forward-pass smoke test on all architectures...")
    smoke_results = smoke_test()
    all_ok = all(ok_shape and ok_loss for ok_shape, ok_loss, *_ in smoke_results.values())
    if not all_ok:
        logger.error("Smoke test failed!")
        return 1
    if args.smoke:
        logger.info("Smoke test OK — exiting.")
        return 0

    # 2. Load config --------------------------------------------------- #
    cfg_path = PROJECT_ROOT / args.config
    cfg = _load_config(cfg_path)

    strategy = args.strategy or cfg.get("strategy", "grid")
    epochs = args.epochs or cfg.get("epochs_per_trial", 10)

    logger.info(f"Strategy={strategy}, epochs_per_trial={epochs}")

    # 3. Build loaders ------------------------------------------------- #
    logger.info("Building data loaders ...")
    train_loader, val_loader, _ = _build_loaders(batch_size=args.batch_size)

    # 4. Architecture selection (quick) ------------------------------- #
    if not args.skip_architecture_selection:
        logger.info("Step 4a — comparing architectures (fixed config, quick)...")
        arch_results = compare_architectures(
            model_names=cfg.get("models", ModelFactory.list_models()),
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=min(2, epochs),
            lr=1e-3,
        )
        arch_report = arch_results_to_md(arch_results)
        out_path = Path("results/runs/architecture_selection.md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(arch_report, encoding="utf-8")
        logger.info(f"Architecture comparison written to {out_path}")
        for r in arch_results:
            logger.info(
                f"  {r.name:14s} val_acc={r.val_acc:.2f}% params={r.num_params:,}"
            )

    # 5. Hyperparameter search ---------------------------------------- #
    search_space = _select_search_space(cfg, strategy)

    # Ensure `epochs` is in the trial params so every trial honours the CLI flag
    search_space = dict(search_space)
    search_space.setdefault("epochs", epochs)

    tuner = HyperparameterTuner(
        to_tune=cfg.get("models", ModelFactory.list_models()),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    scheduler = HyperoptScheduler(
        tuner,
        SchedulerConfig(
            strategy=strategy,
            seed=int(cfg.get("seed", 42)),
            output_dir=cfg.get("output", {}).get("dir", "results/runs"),
            filename_prefix=cfg.get("output", {}).get("filename_prefix", "hyperopt"),
            prune_below=cfg.get("prune_below"),
        ),
    )
    best = scheduler.run(search_space)

    # 6. Final summary ------------------------------------------------ #
    summary_lines = [
        "",
        "=" * 72,
        "NGÜOI 1 SUMMARY — Best (model, hyperparams)",
        "=" * 72,
        f"trial_id     : {best.trial_id}",
        f"val_acc      : {best.val_acc:.2f}%",
        f"train_acc    : {best.train_acc:.2f}%",
        "params       :",
    ]
    for k, v in best.params.items():
        summary_lines.append(f"  - {k}: {v}")
    summary_lines.append("=" * 72)
    print("\n".join(summary_lines))

    return 0


if __name__ == "__main__":
    sys.exit(main())