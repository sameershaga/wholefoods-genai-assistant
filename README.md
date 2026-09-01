# Whole Foods Store Operations Assistant

A production-style reference implementation of a Slack-based GenAI assistant for
store managers querying inventory, deliveries, supplier contracts, and recipes.
It uses synthetic data only and does **not** represent Whole Foods Market
production systems, data, or architecture. The default application runs fully
offline without accounts or paid services.

## Architecture

```text
Slack / FastAPI -> authentication (trusted store context)
  -> query embedding -> filtered vector search (25 candidates)
  -> reranking -> strongest context -> grounded answer with citations
  -> JSONL request telemetry / SQLite feedback
```

Store isolation is enforced before vector search: the authenticated user's
`store_id` is injected by the application service and cannot be overridden by
the caller. Provider protocols isolate auth, embeddings, vector search,
reranking, and answer generation so hosted implementations can replace local
ones without changing orchestration.

## Repository structure

```text
src/store_assistant/
  ingestion/       JSON, PDF, HTML ingestion and normalization
  providers/       embedding, vector-store, reranker, and LLM boundaries
  retrieval.py     filter-first retrieval and reranking
  answering.py     grounded answers and citation enforcement
  services.py      authenticated workflow and telemetry
  api.py/slack.py  FastAPI and signed Slack adapters
  runtime.py       offline composition root
  evaluation.py    golden-dataset evaluation CLI
data/              synthetic records for three stores
evaluation/        golden queries and expectations
tests/             unit and integration tests
```

## Local setup

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
uvicorn store_assistant.runtime:app --app-dir src
```

Check `http://127.0.0.1:8000/health`. Local tokens are
`local-brooklyn-token`, `local-manhattan-token`, and
`local-queens-token`. Brooklyn has 12 synthetic cartons of oat milk,
Manhattan has 3, and Queens has 8.

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H 'Authorization: Bearer local-brooklyn-token' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Do we have oat milk?","filters":{"sku":"OAT001"}}'
```

Use the returned `request_id` for feedback:

```bash
curl -s http://127.0.0.1:8000/v1/feedback \
  -H 'Authorization: Bearer local-brooklyn-token' \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"<request-id>","rating":"up","comment":"Useful"}'
```

## Environment variables

`.env.example` documents local settings and reserved production settings.
Export them or use Uvicorn's `--env-file` option with `python-dotenv`.

| Variable | Purpose | Default/status |
| --- | --- | --- |
| `STORE_ASSISTANT_DELIVERY_LOGS_PATH` | Delivery JSON | `data/delivery_logs.json` |
| `STORE_ASSISTANT_STATE_DIRECTORY` | Logs and feedback | `.local` |
| `STORE_ASSISTANT_EMBEDDING_DIMENSIONS` | Local vector size | `384` |
| `AWS_REGION`, `BEDROCK_EMBEDDING_MODEL_ID` | Titan/Bedrock | reserved |
| `PINECONE_API_KEY`, `PINECONE_INDEX_HOST` | Pinecone | reserved |
| `OKTA_ISSUER`, `OKTA_AUDIENCE` | Okta OIDC | reserved |
| `SLACK_SIGNING_SECRET` | Slack verification | reserved |

Hosted adapters are not yet implemented. Never commit real secrets.

## Slack interface

`SlackCommandHandler` verifies Slack v0 HMAC signatures, rejects requests
older than five minutes, maps a trusted Slack user ID to an access token, and
returns an ephemeral response from `POST /slack/commands`. The local
composition root does not enable the route because it has no signing secret or
user mapping; an integration root passes a configured handler to `create_app`.

```text
/store-assistant Do we have oat milk?
12 cartons of Unsweetened Oat Milk ... [delivery:DLV-BK-1001:0]
Sources: delivery:DLV-BK-1001:0
Request ID: <uuid>
```

## Ingestion pipeline

All ingestors emit a common `DocumentChunk`. Delivery JSON is chunked around
50 tokens, supplier-contract PDF text around 800 tokens, and recipe HTML by
logical sections and steps. Dates become canonical UTC values; SKUs, store IDs,
suppliers, and product categories become consistent filter metadata.
Deterministic IDs support citations, and malformed inputs fail explicitly.

The bundled runtime currently indexes delivery logs. Contract and recipe
ingestors are independently tested but are not yet included at startup.

## Retrieval, answers, and observability

The service embeds each query, applies exact filters (`store_id`,
`product_category`, `sku`, `supplier`, `source_type`) during search,
requests 25 candidates by default, reranks them, and sends only the strongest
context to the answer provider. Local feature-hash embeddings, cosine search,
lexical reranking, and extractive answers are deterministic test substitutes.

The answer service enforces source citations independently of the LLM. JSONL
telemetry records query and identity context, document IDs, retrieval and
reranking scores, model, tokens, estimated cost, latency, and final answer.
Thumbs feedback is persisted in SQLite with per-user isolation.

## Testing and evaluation

```bash
pytest
ruff check .
ruff format --check .
mypy
python -m store_assistant.evaluation
```

Tests cover ingestion, normalization, chunking, store/SKU filtering, retrieval,
reranking, citations, auth propagation, cross-store isolation, feedback, API
errors, Slack signatures, and evaluation. The golden dataset overlaps oat milk
across all three stores and reports retrieval hit rate, correct-store rate,
answer and citation correctness, latency, and estimated cost/query. Offline
token cost is zero; `evaluate` accepts hosted-provider token prices.

## Docker

```bash
docker build -t store-assistant .
docker run --rm -p 8000:8000 \
  -v store-assistant-state:/app/.local store-assistant
```

The Python 3.12 image runs as non-root and health-checks `GET /health`.

## Security considerations

- Trust identity-derived store context, never store IDs from request text.
- Verify Slack signatures on the raw body and enforce replay protection.
- A real Okta adapter must validate signature, issuer, audience, expiry, and
  store claims; mock tokens are development-only.
- Filter vectors before reranking/generation and authorize cited source access.
- Use managed secrets, encryption, telemetry redaction/retention, rate limits,
  dependency scanning, least-privilege IAM, and network controls in production.

## Production deployment approach

Build and scan the container in CI and deploy stateless workers behind TLS.
Implement the existing protocols with Okta OIDC, Amazon Titan/Bedrock,
namespaced Pinecone, and approved hosted LLM/reranking providers selected via
validated environment configuration. Run ingestion as an idempotent background
job, use managed persistence and observability, and add timeouts, retries,
circuit breakers, readiness checks, autoscaling, and cost monitoring.

## Limitations

- Bedrock, Pinecone, Okta, and deployed Slack wiring remain extension points.
- Runtime startup indexes delivery data only.
- The vector index is in-memory and rebuilt at startup.
- Local model substitutes have limited semantic/conversational quality.
- Slack handling is synchronous; production should acknowledge quickly.
- The small synthetic golden set is a regression signal, not real-world proof.
