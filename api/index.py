"""Thin Vercel adapter for the existing deterministic FastAPI runtime."""

from pathlib import Path

from fastapi import FastAPI

from store_assistant.runtime import LocalSettings, create_local_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Vercel Functions may only write to /tmp, and that storage is ephemeral. The
# public demo deliberately uses local deterministic providers and bundled data;
# no cloud credentials are required or read by this composition root.
demo_app = create_local_app(
    LocalSettings(
        delivery_logs_path=PROJECT_ROOT / "data/delivery_logs.json",
        recipe_path=PROJECT_ROOT / "data/recipes/oat_milk_overnight_oats.html",
        state_directory=Path("/tmp/store-assistant"),
    )
)

app = FastAPI(title="Store Operations Assistant — Vercel Adapter")
app.mount("/api", demo_app)
