from fastapi.testclient import TestClient

import recsys.api.main as api_main


def force_missing_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api_main, "DEFAULT_MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api_main, "_model", None)


def test_health_without_model_artifact(tmp_path, monkeypatch) -> None:
    force_missing_model(tmp_path, monkeypatch)
    client = TestClient(api_main.app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_recommend_without_model_returns_explicit_fallback(tmp_path, monkeypatch) -> None:
    force_missing_model(tmp_path, monkeypatch)
    client = TestClient(api_main.app)
    response = client.post("/recommend", json={"user_id": 123, "k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"] == []
    assert payload["model_loaded"] is False
    assert payload["fallback_used"] is True
