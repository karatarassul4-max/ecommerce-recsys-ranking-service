# Senior ML Review Notes

## Strengths

- Uses a real implicit-feedback e-commerce dataset instead of a toy recommendation notebook.
- Avoids random split leakage by using a temporal split.
- Implements simple baselines before adding a two-stage ranker.
- Separates data, modeling, evaluation, and serving code into production-style modules.
- Keeps CI fast and deterministic.

## Full-Run Findings

- The full RetailRocket run produced 1,726,175 preprocessed events, 404,977 users, and 138,387 items.
- The temporal split used 1,428,607 train events, 149,921 validation events, and 147,647 test events.
- Popularity remains the strongest first-run quality baseline on the test split.
- The ranker improves catalog coverage over popularity, but does not beat popularity on NDCG/MAP yet.

## Current Risks

- Ranking features are intentionally lightweight; stronger item/user features could improve quality.
- Cold-start behavior is basic and should be documented in any interview discussion.
- The item-kNN implementation is educational and transparent, but may need optimization for very large catalogs.

## Recommended Next Iterations

- Add experiment results from the full dataset run.
- Add inference latency benchmark report.
- Add drift simulation using item popularity and event-rate distribution changes.
- Add a small Streamlit demo if a visual recruiter-facing demo becomes useful.
