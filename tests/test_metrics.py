from recsys.evaluation.metrics import evaluate_recommendations, ndcg_at_k, recall_at_k


def test_recall_at_k_counts_relevant_hits() -> None:
    assert recall_at_k([1, 2, 3], {2, 4}, 3) == 0.5


def test_ndcg_rewards_better_ranking() -> None:
    better = ndcg_at_k([1, 2, 3], {1, 3}, 3)
    worse = ndcg_at_k([2, 3, 1], {1, 3}, 3)
    assert better > worse


def test_evaluate_recommendations_reports_coverage() -> None:
    metrics = evaluate_recommendations(
        recommendations={1: [10, 11], 2: [12, 13]},
        ground_truth={1: {10}, 2: {13}},
        k=2,
        catalog_items=[10, 11, 12, 13, 14],
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["coverage"] == 0.8

