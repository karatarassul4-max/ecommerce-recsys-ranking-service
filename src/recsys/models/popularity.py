from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from recsys.models.common import build_user_history, filter_seen


@dataclass
class PopularityRecommender:
    top_items: list[int] = field(default_factory=list)
    item_scores: dict[int, float] = field(default_factory=dict)
    user_history: dict[int, set[int]] = field(default_factory=dict)

    def fit(self, events: pd.DataFrame) -> PopularityRecommender:
        scores = events.groupby("item_id")["event_weight"].sum().sort_values(ascending=False)
        self.item_scores = {int(item): float(score) for item, score in scores.items()}
        self.top_items = list(self.item_scores)
        self.user_history = build_user_history(events)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        seen = self.user_history.get(int(user_id), set())
        return filter_seen(self.top_items, seen, k)

    def score(self, item_id: int) -> float:
        return self.item_scores.get(int(item_id), 0.0)
