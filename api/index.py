"""Thin Vercel adapter for the existing deterministic FastAPI runtime."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

# Vercel installs third-party dependencies from the root pyproject.toml, but the
# function must not depend on the repository itself being installed as a wheel.
# Make the shared src-layout package importable directly from the function bundle.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fastapi import FastAPI  # noqa: E402

from store_assistant.runtime import LocalSettings, create_local_app  # noqa: E402

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
app.include_router(demo_app.router, prefix="/api")
