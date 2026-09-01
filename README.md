# Whole Foods GenAI Assistant

Run the fully offline API from the repository root:

```bash
uvicorn store_assistant.runtime:app --app-dir src
```

The synthetic local identities use bearer tokens `local-brooklyn-token`,
`local-manhattan-token`, and `local-queens-token`. Runtime request logs and
feedback are stored under `.local/`.

## Docker

Build and run the same offline API in a non-root container:

```bash
docker build -t store-assistant .
docker run --rm -p 8000:8000 -v store-assistant-state:/app/.local store-assistant
```

The API is then available at `http://localhost:8000`; the container health check
uses `GET /health`. The named volume preserves request logs and feedback between
container runs. All bundled records are synthetic.
