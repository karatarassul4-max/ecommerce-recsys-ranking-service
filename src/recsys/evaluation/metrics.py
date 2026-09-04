from __future__ import annotations

import math
from collections.abc import Iterable


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / min(len(relevant), k)


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
    catalog_items: Iterable[int],
) -> dict[str, float]:
    users = [user for user in ground_truth if ground_truth[user]]
    if not users:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "map": 0.0, "coverage": 0.0}

    recommended_items = set()
    precision = []
    recall = []
    ndcg = []
    map_scores = []
    for user in users:
        recs = recommendations.get(user, [])
        recommended_items.update(recs[:k])
        relevant = ground_truth[user]
        precision.append(precision_at_k(recs, relevant, k))
        recall.append(recall_at_k(recs, relevant, k))
        ndcg.append(ndcg_at_k(recs, relevant, k))
        map_scores.append(average_precision_at_k(recs, relevant, k))

    catalog = set(catalog_items)
    return {
        "precision": float(sum(precision) / len(precision)),
        "recall": float(sum(recall) / len(recall)),
        "ndcg": float(sum(ndcg) / len(ndcg)),
        "map": float(sum(map_scores) / len(map_scores)),
        "coverage": float(len(recommended_items) / len(catalog)) if catalog else 0.0,
    }
