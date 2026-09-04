from __future__ import annotations

from collections import defaultdict

import pandas as pd


def build_user_history(events: pd.DataFrame) -> dict[int, set[int]]:
    history: dict[int, set[int]] = defaultdict(set)
    for row in events[["user_id", "item_id"]].itertuples(index=False):
        history[int(row.user_id)].add(int(row.item_id))
    return dict(history)


def ground_truth(events: pd.DataFrame, positive_events: list[str]) -> dict[int, set[int]]:
    positives = events[events["event"].isin(positive_events)]
    return build_user_history(positives)


def filter_seen(items: list[int], seen: set[int], limit: int) -> list[int]:
    output = []
    for item in items:
        if item in seen or item in output:
            continue
        output.append(item)
        if len(output) >= limit:
            break
    return output
