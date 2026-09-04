from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "visitorid", "event", "itemid"}


def load_events(path: str | Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(events.columns)
    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")
    return events


def preprocess_events(
    events_path: str | Path,
    output_dir: str | Path,
    event_weights: dict[str, float],
    min_user_events: int,
    min_item_events: int,
) -> dict:
    events = load_events(events_path)
    events = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    events = events[["timestamp", "event", "user_id", "item_id"]]
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", utc=True)
    events["event_weight"] = events["event"].map(event_weights).fillna(1.0).astype(float)
    events = events.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)

    user_counts = events["user_id"].value_counts()
    item_counts = events["item_id"].value_counts()
    events = events[
        events["user_id"].isin(user_counts[user_counts >= min_user_events].index)
        & events["item_id"].isin(item_counts[item_counts >= min_item_events].index)
    ].reset_index(drop=True)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output / "events.parquet", index=False)
    summary = {
        "events": int(len(events)),
        "users": int(events["user_id"].nunique()),
        "items": int(events["item_id"].nunique()),
        "event_counts": {str(k): int(v) for k, v in events["event"].value_counts().items()},
        "min_timestamp": events["timestamp"].min().isoformat(),
        "max_timestamp": events["timestamp"].max().isoformat(),
    }
    (output / "preprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
