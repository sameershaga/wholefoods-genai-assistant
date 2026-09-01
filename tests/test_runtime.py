from pathlib import Path

from fastapi.testclient import TestClient

from store_assistant.runtime import LocalSettings, create_local_app


def test_local_settings_load_from_environment(tmp_path: Path) -> None:
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_DELIVERY_LOGS_PATH": "fixtures/deliveries.json",
            "STORE_ASSISTANT_STATE_DIRECTORY": str(tmp_path),
            "STORE_ASSISTANT_EMBEDDING_DIMENSIONS": "128",
        }
    )

    assert settings.delivery_logs_path == Path("fixtures/deliveries.json")
    assert settings.state_directory == tmp_path
    assert settings.embedding_dimensions == 128


def test_local_settings_reject_invalid_embedding_dimensions() -> None:
    for value in ("zero", "0", "-1"):
        try:
            LocalSettings.from_environment(
                {"STORE_ASSISTANT_EMBEDDING_DIMENSIONS": value}
            )
        except ValueError as exc:
            assert "STORE_ASSISTANT_EMBEDDING_DIMENSIONS" in str(exc)
        else:
            raise AssertionError(f"expected invalid dimensions {value!r} to fail")


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
