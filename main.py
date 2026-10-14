"""Command-line entry point for the end-to-end training pipeline."""

from __future__ import annotations

import argparse

import run_training as training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the FashionMNIST classifier")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    config = training_pipeline.load_config(args.config)
    config = training_pipeline.apply_overrides(config, args)
    return training_pipeline.run_training(config)


if __name__ == "__main__":
    main()
