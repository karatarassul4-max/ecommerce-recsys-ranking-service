from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from recsys.models.common import build_user_history, filter_seen


@dataclass
class SVDRecommender:
    n_components: int = 64
    user_index: dict[int, int] = field(default_factory=dict)
    item_index: dict[int, int] = field(default_factory=dict)
    index_item: dict[int, int] = field(default_factory=dict)
    item_factors: np.ndarray | None = None
    user_factors: np.ndarray | None = None
    user_history: dict[int, set[int]] = field(default_factory=dict)

    def fit(self, events: pd.DataFrame) -> SVDRecommender:
        users = sorted(int(user) for user in events["user_id"].unique())
        items = sorted(int(item) for item in events["item_id"].unique())
        self.user_index = {user: idx for idx, user in enumerate(users)}
        self.item_index = {item: idx for idx, item in enumerate(items)}
        self.index_item = {idx: item for item, idx in self.item_index.items()}
        self.user_history = build_user_history(events)

        rows = events["user_id"].map(self.user_index).to_numpy()
        cols = events["item_id"].map(self.item_index).to_numpy()
        values = events["event_weight"].to_numpy(dtype=float)
        matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))

        n_components = min(self.n_components, max(1, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = normalize(svd.fit_transform(matrix))
        self.item_factors = normalize(svd.components_.T)
        return self

    def recommend(self, user_id: int, k: int) -> list[int]:
        if self.user_factors is None or self.item_factors is None:
            return []
        idx = self.user_index.get(int(user_id))
        if idx is None:
            return []
        scores = self.item_factors @ self.user_factors[idx]
        ranked_indices = np.argsort(scores)[::-1]
        ranked_items = [self.index_item[int(index)] for index in ranked_indices]
        return filter_seen(ranked_items, self.user_history.get(int(user_id), set()), k)

    def score(self, user_id: int, item_id: int) -> float:
        if self.user_factors is None or self.item_factors is None:
            return 0.0
        user_idx = self.user_index.get(int(user_id))
        item_idx = self.item_index.get(int(item_id))
        if user_idx is None or item_idx is None:
            return 0.0
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])
