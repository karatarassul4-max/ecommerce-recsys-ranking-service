from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Visitor/user id from the interaction dataset")
    k: int = Field(default=10, ge=1, le=100)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[int]
    model_loaded: bool
    fallback_used: bool = False
