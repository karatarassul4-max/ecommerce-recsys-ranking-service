# E-commerce Recommender Ranking Service

Production-style recommendation system for implicit-feedback e-commerce events.

The project uses the public RetailRocket e-commerce dataset and follows a realistic offline ML lifecycle:
raw interactions -> preprocessing -> temporal split -> baselines -> two-stage recommendation -> offline evaluation -> model artifact -> FastAPI serving -> tests/CI.

## Dataset

**RetailRocket E-commerce Dataset**

- Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
- Main file: `events.csv`
- Events: `view`, `addtocart`, `transaction`
- User key: `visitorid`
- Item key: `itemid`
- Timestamp: millisecond Unix timestamp

This is an implicit-feedback task. The model does not predict ratings; it recommends items based on user behavior.

## ML Task

Recommend top-K items for a user, using only historical interactions available before the evaluation window.

Positive evaluation events:

- `addtocart`
- `transaction`

Views are used as weaker implicit signals during training, but add-to-cart and transaction events define stronger relevance for offline metrics.

## Architecture

```mermaid
flowchart LR
    A[RetailRocket events.csv] --> B[Preprocessing]
    B --> C[Temporal train/val/test split]
    C --> D1[Popularity baseline]
    C --> D2[Item-kNN baseline]
    C --> D3[SVD candidate generator]
    D1 --> E[Candidate pool]
    D2 --> E
    D3 --> E
    E --> F[Learning-to-rank model]
    F --> G[Offline evaluation @K]
    F --> H[Joblib model artifact]
    H --> I[FastAPI recommendation API]
```

## Models

- **Popularity baseline**: recommends globally popular items, excluding items already seen by the user.
- **Item-kNN baseline**: item-item collaborative filtering from co-occurrence in user histories.
- **SVD matrix factorization**: latent-factor candidate generation using sparse user-item interactions.
- **Two-stage ranker**: combines popularity, item-kNN, SVD score, and user-history features in a lightweight ranking model.

## Metrics

Offline ranking metrics are computed on users with positive validation/test events:

- `Precision@K`
- `Recall@K`
- `NDCG@K`
- `MAP@K`
- catalog coverage

No final RetailRocket metrics are committed until the full dataset experiment is actually run.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[train,dev]"
```

Create a tiny local sample for smoke testing:

```bash
make make-sample
make preprocess
make split
make train-baselines
make train-ranker
make evaluate
```

Run the API:

```bash
make serve
```

Request recommendations:

```bash
curl -X POST http://localhost:8000/recommend ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": 1, \"k\": 10}"
```

## Full Dataset Run

Download from Kaggle with KaggleHub:

```bash
make install-train
make download
make preprocess
make split
make train-baselines
make train-ranker
make evaluate
```

MLflow tracking is enabled when `mlflow` is installed:

```bash
make mlflow-ui
```

## Experiment Results

RetailRocket full-run metrics will be added only after running the real pipeline.

| Run | Split | Model | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Coverage | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | temporal validation | popularity | - | - | - | - | - | not run yet |
| 002 | temporal validation | item-kNN | - | - | - | - | - | not run yet |
| 003 | temporal validation | SVD | - | - | - | - | - | not run yet |
| 004 | temporal test | two-stage ranker | - | - | - | - | - | not run yet |

## Repository Layout

```text
configs/                 YAML configuration
data/raw/                raw dataset location, ignored by git
data/processed/          parquet splits, ignored by git
artifacts/               trained model artifacts, ignored by git
reports/                 generated metrics and summaries, ignored by git
src/recsys/api/          FastAPI app and Pydantic schemas
src/recsys/data/         download, preprocessing, sample data, split logic
src/recsys/evaluation/   ranking metrics
src/recsys/models/       recommenders and ranker
src/recsys/pipeline/     training/evaluation orchestration
tests/                   unit and integration tests
```

## Docker

```bash
docker build -t ecommerce-recsys-ranking-service .
docker run --rm -p 8000:8000 ecommerce-recsys-ranking-service
```

## Engineering Decisions

- Temporal split is used to avoid future-interaction leakage.
- Simple baselines are implemented before the ranker.
- Implicit feedback is weighted by event type.
- Model artifacts are serialized as full recommendation pipelines.
- API returns an explicit empty fallback response when no model artifact is available.
- CI keeps checks lightweight: linting plus deterministic tests.

