from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def temporal_split(
    events_path: str | Path,
    output_dir: str | Path,
    validation_days: int,
    test_days: int,
) -> dict:
    events = pd.read_parquet(events_path).sort_values("timestamp").reset_index(drop=True)
    max_ts = events["timestamp"].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=validation_days)

    train = events[events["timestamp"] < val_start].copy()
    val = events[(events["timestamp"] >= val_start) & (events["timestamp"] < test_start)].copy()
    test = events[events["timestamp"] >= test_start].copy()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output / "train.parquet", index=False)
    val.to_parquet(output / "val.parquet", index=False)
    test.to_parquet(output / "test.parquet", index=False)

    summary = {
        "train_events": int(len(train)),
        "val_events": int(len(val)),
        "test_events": int(len(test)),
        "val_start": val_start.isoformat(),
        "test_start": test_start.isoformat(),
        "max_timestamp": max_ts.isoformat(),
        "leakage_check": {
            "train_max_before_val": bool(train.empty or train["timestamp"].max() < val_start),
            "val_max_before_test": bool(val.empty or val["timestamp"].max() < test_start),
        },
    }
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
