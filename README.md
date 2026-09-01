# Whole Foods GenAI Assistant

Run the fully offline API from the repository root:

```bash
uvicorn store_assistant.runtime:app --app-dir src
```

The synthetic local identities use bearer tokens `local-brooklyn-token`,
`local-manhattan-token`, and `local-queens-token`. Runtime request logs and
feedback are stored under `.local/`.

Runtime paths and embedding dimensions can be configured with environment
variables documented in `.env.example`. Copying that file to `.env` is useful
for reference, but Uvicorn does not load it automatically; export the values or
start Uvicorn with `--env-file .env` (which requires `python-dotenv`).

## Docker

Build and run the same offline API in a non-root container:

```bash
docker build -t store-assistant .
docker run --rm -p 8000:8000 -v store-assistant-state:/app/.local store-assistant
```

The API is then available at `http://localhost:8000`; the container health check
uses `GET /health`. The named volume preserves request logs and feedback between
container runs. All bundled records are synthetic.

## Evaluation

The small synthetic golden dataset covers all three stores and deliberately uses
the same oat-milk SKU to measure cross-store isolation. Run it with:

```bash
python -m store_assistant.evaluation
```

The JSON report includes retrieval hit rate, correct-store retrieval rate, answer
and citation correctness, average latency, and estimated cost per query. Local
providers have zero configured token cost; hosted-provider evaluations can pass
their actual token prices to the `evaluate` function.
