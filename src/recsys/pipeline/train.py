from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from recsys.config import ProjectConfig
from recsys.evaluation.metrics import evaluate_recommendations
from recsys.models.common import ground_truth
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.ranker import RankingRecommender
from recsys.models.svd import SVDRecommender


def _log_mlflow_run(
    config: ProjectConfig,
    run_name: str,
    metrics: dict[str, float] | dict[str, dict[str, float]],
    params: dict[str, object],
) -> None:
    try:
        import mlflow
    except ImportError:
        return

    tracking = config.tracking
    mlflow.set_tracking_uri(str(tracking["tracking_uri"]))
    mlflow.set_experiment(str(tracking["experiment_name"]))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for key, value in metrics.items():
            if isinstance(value, dict):
                for metric_name, metric_value in value.items():
                    mlflow.log_metric(f"{key}_{metric_name}", float(metric_value))
            else:
                mlflow.log_metric(key, float(value))


def _split_paths(config: ProjectConfig) -> dict[str, Path]:
    root = Path(config.dataset["processed_dir"])
    return {
        "train": root / "train.parquet",
        "val": root / "val.parquet",
        "test": root / "test.parquet",
    }


def _recommend_for_users(model: object, users: list[int], k: int) -> dict[int, list[int]]:
    return {int(user): model.recommend(int(user), k) for user in users}


def evaluate_model(
    model: object,
    evaluation_events: pd.DataFrame,
    positive_events: list[str],
    catalog_items: list[int],
    k: int,
) -> dict[str, float]:
    truth = ground_truth(evaluation_events, positive_events)
    recommendations = _recommend_for_users(model, list(truth), k)
    return evaluate_recommendations(recommendations, truth, k, catalog_items)


def train_baselines(config: ProjectConfig) -> dict:
    paths = _split_paths(config)
    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])
    k = int(config.models["top_k"])
    positive_events = list(config.split["positive_events"])

    popularity = PopularityRecommender().fit(train)
    item_knn = ItemKNNRecommender(**config.models["item_knn"]).fit(train)
    svd = SVDRecommender(**config.models["svd"]).fit(train)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    joblib.dump(popularity, artifacts / "popularity.joblib")
    joblib.dump(item_knn, artifacts / "item_knn.joblib")
    joblib.dump(svd, artifacts / "svd.joblib")

    catalog = list(popularity.item_scores)
    results = {
        "popularity": evaluate_model(popularity, val, positive_events, catalog, k),
        "item_knn": evaluate_model(item_knn, val, positive_events, catalog, k),
        "svd": evaluate_model(svd, val, positive_events, catalog, k),
    }
    _log_mlflow_run(
        config,
        "baselines",
        results,
        {
            "top_k": k,
            "item_knn_max_neighbors": config.models["item_knn"]["max_neighbors"],
            "svd_n_components": config.models["svd"]["n_components"],
            "split": "temporal_validation",
        },
    )
    output = Path("reports/baseline_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def train_ranker(config: ProjectConfig) -> dict:
    paths = _split_paths(config)
    train = pd.read_parquet(paths["train"])
    val = pd.read_parquet(paths["val"])
    test = pd.read_parquet(paths["test"])
    k = int(config.models["top_k"])
    positive_events = list(config.split["positive_events"])

    popularity = PopularityRecommender().fit(train)
    item_knn = ItemKNNRecommender(**config.models["item_knn"]).fit(train)
    svd = SVDRecommender(**config.models["svd"]).fit(train)
    ranker = RankingRecommender(
        popularity=popularity,
        item_knn=item_knn,
        svd=svd,
        candidate_k=int(config.models["candidate_k"]),
    ).fit(
        train,
        val,
        positive_events,
        int(config.models["ranker"]["negatives_per_positive"]),
        int(config.project["seed"]),
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    joblib.dump(ranker, artifacts / "ranker.joblib")

    catalog = list(popularity.item_scores)
    results = {
        "validation": evaluate_model(ranker, val, positive_events, catalog, k),
        "test": evaluate_model(ranker, test, positive_events, catalog, k),
    }
    _log_mlflow_run(
        config,
        "two_stage_ranker",
        results,
        {
            "top_k": k,
            "candidate_k": config.models["candidate_k"],
            "negatives_per_positive": config.models["ranker"]["negatives_per_positive"],
            "ranker_model": type(ranker.model).__name__,
            "split": "temporal_validation_test",
        },
    )
    output = Path("reports/ranker_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def evaluate_artifact(config: ProjectConfig, model_path: str | Path) -> dict:
    model = joblib.load(model_path)
    test = pd.read_parquet(_split_paths(config)["test"])
    positive_events = list(config.split["positive_events"])
    k = int(config.models["top_k"])
    catalog = list(getattr(model, "popularity", model).item_scores)
    results = evaluate_model(model, test, positive_events, catalog, k)
    output = Path("reports/evaluation_metrics.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
