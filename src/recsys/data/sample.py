from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_sample_events(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2024-01-01T00:00:00Z")
    day_ms = 24 * 60 * 60 * 1000
    rows = [
        (0, 1, "view", 10),
        (1, 1, "addtocart", 11),
        (2, 1, "view", 12),
        (3, 2, "view", 10),
        (4, 2, "transaction", 12),
        (5, 2, "view", 13),
        (6, 3, "view", 11),
        (7, 3, "addtocart", 13),
        (8, 3, "view", 14),
        (9, 4, "view", 10),
        (10, 4, "transaction", 14),
        (11, 4, "view", 15),
        (12, 5, "view", 11),
        (13, 5, "addtocart", 15),
        (14, 5, "view", 12),
        (20, 1, "addtocart", 13),
        (21, 2, "transaction", 14),
        (22, 3, "addtocart", 10),
        (23, 4, "view", 11),
        (24, 5, "transaction", 13),
        (40, 1, "transaction", 14),
        (41, 2, "addtocart", 11),
        (42, 3, "transaction", 12),
        (43, 4, "addtocart", 13),
        (44, 5, "transaction", 10),
    ]
    rows = [
        (int(base.timestamp() * 1000) + day * day_ms, user_id, event, item_id)
        for day, user_id, event, item_id in rows
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid"])
    frame.to_csv(output, index=False)
    return output
