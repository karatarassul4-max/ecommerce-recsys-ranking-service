from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from recsys.models.common import build_user_history, filter_seen


@dataclass
class ItemKNNRecommender:
    max_neighbors: int = 100
    neighbors: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    user_history: dict[int, set[int]] = field(default_factory=dict)

    def fit(self, events: pd.DataFrame) -> ItemKNNRecommender:
        self.user_history = build_user_history(events)
        item_popularity = events.groupby("item_id")["user_id"].nunique().to_dict()
        co_counts: dict[tuple[int, int], float] = {}

        for items in self.user_history.values():
            unique_items = sorted(items)
            for left, right in combinations(unique_items, 2):
                co_counts[(left, right)] = co_counts.get((left, right), 0.0) + 1.0
                co_counts[(right, left)] = co_counts.get((right, left), 0.0) + 1.0

        grouped: dict[int, list[tuple[int, float]]] = {}
        for (left, right), count in co_counts.items():
            denom = (item_popularity.get(left, 1) * item_popularity.get(right, 1)) ** 0.5
            grouped.setdefault(left, []).append((right, count / denom))

        self.neighbors = {
            item: sorted(values, key=lambda pair: pair[1], reverse=True)[: self.max_neighbors]
            for item, values in grouped.items()
        }
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self.user_history.get(int(user_id), set())
        scores: dict[int, float] = {}
        for item in seen:
            for neighbor, score in self.neighbors.get(item, []):
                if neighbor not in seen:
                    scores[neighbor] = scores.get(neighbor, 0.0) + score
        ranked = [
            item
            for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        ]
        return filter_seen(ranked, seen, k)

    def score(self, user_id: int, item_id: int) -> float:
        seen = self.user_history.get(int(user_id), set())
        score = 0.0
        for item in seen:
            score += dict(self.neighbors.get(item, [])).get(int(item_id), 0.0)
        return score
