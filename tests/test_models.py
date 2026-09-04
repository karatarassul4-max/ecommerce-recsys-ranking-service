import pandas as pd

from recsys.models.popularity import PopularityRecommender


def test_popularity_recommender_excludes_seen_items() -> None:
    events = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 11, 10, 12, 13],
            "event_weight": [1.0, 3.0, 1.0, 5.0, 2.0],
        }
    )
    model = PopularityRecommender().fit(events)

    recommendations = model.recommend(user_id=1, k=3)

    assert 10 not in recommendations
    assert 11 not in recommendations
    assert recommendations[0] == 12

