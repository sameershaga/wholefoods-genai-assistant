from pathlib import Path

from fastapi.testclient import TestClient

from store_assistant.runtime import LocalSettings, create_local_app


def test_local_runtime_loads_synthetic_data_and_enforces_store_isolation(
    tmp_path: Path,
) -> None:
    app = create_local_app(LocalSettings(state_directory=tmp_path))

    with TestClient(app) as client:
        brooklyn = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer local-brooklyn-token"},
            json={"query": "Do we have oat milk?", "filters": {"sku": "OAT001"}},
        )
        manhattan = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer local-manhattan-token"},
            json={"query": "Do we have oat milk?", "filters": {"sku": "OAT001"}},
        )

    assert brooklyn.status_code == 200
    assert "12 cartons" in brooklyn.json()["answer"]
    assert "3 cartons" not in brooklyn.json()["answer"]
    assert manhattan.status_code == 200
    assert "3 cartons" in manhattan.json()["answer"]
    assert (tmp_path / "requests.jsonl").exists()
    assert (tmp_path / "feedback.sqlite3").exists()


def test_local_runtime_rejects_unknown_token(tmp_path: Path) -> None:
    client = TestClient(create_local_app(LocalSettings(state_directory=tmp_path)))

    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer unknown"},
        json={"query": "Do we have oat milk?"},
    )

    assert response.status_code == 401
