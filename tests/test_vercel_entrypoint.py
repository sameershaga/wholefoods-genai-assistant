"""Integration coverage for the thin Vercel deployment adapter."""

import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_vercel_app() -> FastAPI:
    entrypoint = Path(__file__).parents[1] / "api/index.py"
    spec = importlib.util.spec_from_file_location("vercel_entrypoint", entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert isinstance(module.app, FastAPI)
    return module.app


def test_vercel_adapter_serves_store_scoped_demo_end_to_end() -> None:
    with TestClient(_load_vercel_app()) as client:
        stores = client.get("/api/v1/demo/stores")
        answer = client.post(
            "/api/v1/demo/query",
            json={"store_id": "BROOKLYN-01", "query": "Do we have oat milk?"},
        )

    assert stores.status_code == 200
    assert {store["store_id"] for store in stores.json()} == {
        "BROOKLYN-01",
        "MANHATTAN-01",
        "QUEENS-01",
    }
    assert answer.status_code == 200
    assert answer.json()["store_id"] == "BROOKLYN-01"
    assert answer.json()["citations"]
