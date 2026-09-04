from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from recsys.config import load_config
from recsys.data.download import download_with_kagglehub
from recsys.data.preprocess import preprocess_events
from recsys.data.sample import write_sample_events
from recsys.data.split import temporal_split
from recsys.logging import configure_logging
from recsys.pipeline.train import evaluate_artifact, train_baselines, train_ranker

LOGGER = logging.getLogger(__name__)


def cmd_download(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = download_with_kagglehub(config.dataset["kaggle_slug"], config.dataset["raw_dir"])
    LOGGER.info("Dataset events available at %s", path)


def cmd_make_sample(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    path = write_sample_events(args.output or config.dataset["events_file"])
    LOGGER.info("Sample events written to %s", path)


def cmd_preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    summary = preprocess_events(
        config.dataset["events_file"],
        config.dataset["processed_dir"],
        config.dataset["event_weights"],
        int(config.dataset["min_user_events"]),
        int(config.dataset["min_item_events"]),
    )
    LOGGER.info(json.dumps(summary, indent=2))


def cmd_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    events_path = Path(config.dataset["processed_dir"]) / "events.parquet"
    summary = temporal_split(
        events_path,
        config.dataset["processed_dir"],
        int(config.split["validation_days"]),
        int(config.split["test_days"]),
    )
    LOGGER.info(json.dumps(summary, indent=2))


def cmd_train_baselines(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    results = train_baselines(config)
    LOGGER.info(json.dumps(results, indent=2))


def cmd_train_ranker(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    results = train_ranker(config)
    LOGGER.info(json.dumps(results, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    results = evaluate_artifact(config, args.model)
    LOGGER.info(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce recsys ranking CLI")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--config", default="configs/retailrocket.yaml")
    download.set_defaults(handler=cmd_download)

    sample = subparsers.add_parser("make-sample")
    sample.add_argument("--config", default="configs/retailrocket.yaml")
    sample.add_argument("--output", default=None)
    sample.set_defaults(handler=cmd_make_sample)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--config", default="configs/retailrocket.yaml")
    preprocess.set_defaults(handler=cmd_preprocess)

    split = subparsers.add_parser("split")
    split.add_argument("--config", default="configs/retailrocket.yaml")
    split.set_defaults(handler=cmd_split)

    baselines = subparsers.add_parser("train-baselines")
    baselines.add_argument("--config", default="configs/retailrocket.yaml")
    baselines.set_defaults(handler=cmd_train_baselines)

    ranker = subparsers.add_parser("train-ranker")
    ranker.add_argument("--config", default="configs/retailrocket.yaml")
    ranker.set_defaults(handler=cmd_train_ranker)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/retailrocket.yaml")
    evaluate.add_argument("--model", required=True)
    evaluate.set_defaults(handler=cmd_evaluate)

    args = parser.parse_args()
    configure_logging(args.log_level)
    args.handler(args)


if __name__ == "__main__":
    main()
