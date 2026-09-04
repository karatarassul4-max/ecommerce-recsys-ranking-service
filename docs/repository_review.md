# Senior ML Review Notes

## Strengths

- Uses a real implicit-feedback e-commerce dataset instead of a toy recommendation notebook.
- Avoids random split leakage by using a temporal split.
- Implements simple baselines before adding a two-stage ranker.
- Separates data, modeling, evaluation, and serving code into production-style modules.
- Keeps CI fast and deterministic.

## Current Risks

- Full RetailRocket metrics are not yet committed; they should be added only after a real run.
- Ranking features are intentionally lightweight; stronger item/user features could improve quality.
- Cold-start behavior is basic and should be documented in any interview discussion.
- The item-kNN implementation is educational and transparent, but may need optimization for very large catalogs.

## Recommended Next Iterations

- Add experiment results from the full dataset run.
- Add inference latency benchmark report.
- Add drift simulation using item popularity and event-rate distribution changes.
- Add a small Streamlit demo if a visual recruiter-facing demo becomes useful.

