from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI

from recsys.api.schemas import RecommendationRequest, RecommendationResponse

DEFAULT_MODEL_PATH = Path(os.getenv("RECSYS_MODEL_PATH", "artifacts/ranker.joblib"))

app = FastAPI(title="E-commerce Recommendation Ranking Service", version="0.1.0")
_model = None


def get_model() -> object | None:
    global _model
    if _model is None and DEFAULT_MODEL_PATH.exists():
        _model = joblib.load(DEFAULT_MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True, "model_available": DEFAULT_MODEL_PATH.exists()}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> RecommendationResponse:
    model = get_model()
    if model is None:
        return RecommendationResponse(
            user_id=payload.user_id,
            recommendations=[],
            model_loaded=False,
            fallback_used=True,
        )
    recommendations = model.recommend(payload.user_id, payload.k)
    return RecommendationResponse(
        user_id=payload.user_id,
        recommendations=recommendations,
        model_loaded=True,
        fallback_used=False,
    )
