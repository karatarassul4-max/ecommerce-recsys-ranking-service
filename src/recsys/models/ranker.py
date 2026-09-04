from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from recsys.models.common import filter_seen
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.models.svd import SVDRecommender


@dataclass
class RankingRecommender:
    popularity: PopularityRecommender
    item_knn: ItemKNNRecommender
    svd: SVDRecommender
    candidate_k: int = 100
    model: HistGradientBoostingClassifier = field(default_factory=HistGradientBoostingClassifier)

    def _features(self, user_id: int, item_id: int) -> list[float]:
        user_history = self.popularity.user_history.get(int(user_id), set())
        return [
            self.popularity.score(item_id),
            self.item_knn.score(user_id, item_id),
            self.svd.score(user_id, item_id),
            float(len(user_history)),
            float(item_id in user_history),
        ]

    def candidate_items(self, user_id: int) -> list[int]:
        candidates = []
        candidates.extend(self.svd.recommend(user_id, self.candidate_k))
        candidates.extend(self.item_knn.recommend(user_id, self.candidate_k))
        candidates.extend(self.popularity.recommend(user_id, self.candidate_k))
        return filter_seen(candidates, set(), self.candidate_k)

    def fit(
        self,
        _train_events: pd.DataFrame,
        validation_events: pd.DataFrame,
        positive_events: list[str],
        negatives_per_positive: int,
        seed: int,
    ) -> RankingRecommender:
        rng = Random(seed)
        positives = validation_events[validation_events["event"].isin(positive_events)]
        catalog = list(self.popularity.item_scores)
        rows = []
        labels = []

        for event in positives.itertuples(index=False):
            user_id = int(event.user_id)
            positive_item = int(event.item_id)
            if user_id not in self.popularity.user_history:
                continue
            rows.append(self._features(user_id, positive_item))
            labels.append(1)
            seen = set(self.popularity.user_history.get(user_id, set())) | {positive_item}
            pool = [
                item
                for item in catalog[: max(1000, negatives_per_positive * 20)]
                if item not in seen
            ]
            for negative_item in rng.sample(pool, k=min(negatives_per_positive, len(pool))):
                rows.append(self._features(user_id, int(negative_item)))
                labels.append(0)

        if not rows or len(set(labels)) < 2:
            raise ValueError("Not enough positive and negative samples to train ranker")

        self.model.fit(np.asarray(rows, dtype=float), np.asarray(labels, dtype=int))
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        candidates = self.candidate_items(user_id)
        if not candidates:
            return self.popularity.recommend(user_id, k)
        features = np.asarray([self._features(user_id, item) for item in candidates], dtype=float)
        scores = self.model.predict_proba(features)[:, 1]
        ranked = [
            item
            for item, _ in sorted(
                zip(candidates, scores, strict=False),
                key=lambda pair: pair[1],
                reverse=True,
            )
        ]
        seen = self.popularity.user_history.get(int(user_id), set())
        return filter_seen(ranked, seen, k)
