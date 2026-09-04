import pandas as pd

from recsys.data.split import temporal_split


def test_temporal_split_keeps_future_out_of_training(tmp_path) -> None:
    events = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-01-20",
                    "2024-01-30",
                    "2024-02-05",
                ],
                utc=True,
            ),
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 11, 10, 12, 13],
            "event": ["view", "addtocart", "view", "transaction", "addtocart"],
            "event_weight": [1.0, 3.0, 1.0, 5.0, 3.0],
        }
    )
    source = tmp_path / "events.parquet"
    output = tmp_path / "processed"
    events.to_parquet(source, index=False)

    summary = temporal_split(source, output, validation_days=10, test_days=10)

    assert summary["leakage_check"]["train_max_before_val"] is True
    assert summary["leakage_check"]["val_max_before_test"] is True
    assert len(pd.read_parquet(output / "train.parquet")) == 2
    assert len(pd.read_parquet(output / "val.parquet")) == 1
    assert len(pd.read_parquet(output / "test.parquet")) == 2

