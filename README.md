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

For a one-command container startup with request logs and feedback persisted in
a named volume, run:

```bash
docker compose up --build -d
docker compose ps
```

The API is available on port `8000` by default. Set `STORE_ASSISTANT_PORT` to
publish a different host port, and use `docker compose down` to stop the service.
The named volume is retained by default; remove it explicitly with
`docker compose down --volumes` only when you intend to discard local state.

## Environment variables

`.env.example` documents local settings and reserved production settings.
Export them or use Uvicorn's `--env-file` option with `python-dotenv`.

| Variable | Purpose | Default/status |
| --- | --- | --- |
| `STORE_ASSISTANT_DELIVERY_LOGS_PATH` | Delivery JSON | `data/delivery_logs.json` |
| `STORE_ASSISTANT_RECIPE_PATH` | Recipe HTML | bundled oat-milk recipe |
| `STORE_ASSISTANT_SUPPLIER_CONTRACT_PATH` | Supplier contract PDF or PDF directory | unset/disabled |
| `STORE_ASSISTANT_STATE_DIRECTORY` | Logs and feedback | `.local` |
| `STORE_ASSISTANT_EMBEDDING_DIMENSIONS` | Local vector size | `384` |
| `STORE_ASSISTANT_EMBEDDING_PROVIDER` | `local` or Amazon `bedrock` | `local` |
| `STORE_ASSISTANT_VECTOR_STORE_PROVIDER` | `local` or `pinecone` | `local` |
| `STORE_ASSISTANT_AUTH_PROVIDER` | `mock` or `okta` | `mock` |
| `STORE_ASSISTANT_LLM_PROVIDER` | `local` or Amazon `bedrock` | `local` |
| `AWS_REGION`, `BEDROCK_EMBEDDING_MODEL_ID` | Titan/Bedrock | reserved |
| `BEDROCK_LLM_MODEL_ID`, `BEDROCK_LLM_MAX_TOKENS`, `BEDROCK_LLM_TEMPERATURE` | Bedrock Converse generation | Nova Lite, `300`, `0` |
| `PINECONE_API_KEY`, `PINECONE_INDEX_HOST`, `PINECONE_NAMESPACE` | Pinecone | unset |
| `OKTA_ISSUER`, `OKTA_AUDIENCE` | Okta token verification | hosted adapter |
| `OKTA_STORE_ID_CLAIM` | Trusted store-scope claim | `store_id` |
| `SLACK_SIGNING_SECRET` | Slack request verification | unset |
| `STORE_ASSISTANT_SLACK_USER_TOKENS` | Slack user-to-token JSON map | unset |

Set the embedding provider to `bedrock`, choose a Titan V2 dimension of 256,
512, or 1024, and install `.[aws]` to let the runtime create its Bedrock client
from the standard AWS credential chain. `AmazonTitanEmbeddingProvider`, `PineconeVectorStore`, and
`OktaOIDCAuthProvider` implement hosted provider boundaries using injected SDK
clients. Set the vector-store provider to `pinecone`, provide its API key and
index host, and install `.[pinecone]` to use an existing Pinecone index whose
dimension matches the embedding provider. The OIDC verifier must validate the JWT signature, issuer, audience,
expiry, and other standard claims before returning identity claims. Credentials,
key caching, retries, and timeouts remain deployment configuration.
Set the auth provider to `okta`, configure issuer, audience, and the trusted
store claim, and install `.[okta]`; the runtime verifies RS256 access tokens
against the issuer's JWKS endpoint. Mock tokens remain the offline default.
Set the LLM provider to `bedrock` and install `.[aws]` to generate grounded
answers with Bedrock Converse. Model token usage flows into the existing request
telemetry and cost calculation, while citations remain service-enforced.
Never commit real secrets.

## Slack interface

`SlackCommandHandler` verifies Slack v0 HMAC signatures, rejects requests
older than five minutes, maps a trusted Slack user ID to an access token, and
returns an ephemeral response from `POST /slack/commands`. The local
composition root enables the route when both `SLACK_SIGNING_SECRET` and
`STORE_ASSISTANT_SLACK_USER_TOKENS` are set. For example, the latter can be
`{"U123":"local-brooklyn-token"}`. Leaving both unset disables the route.

```text
/store-assistant Do we have oat milk?
12 cartons of Unsweetened Oat Milk ... [delivery:DLV-BK-1001:0]
Sources: delivery:DLV-BK-1001:0
Request ID: <uuid>
```

The response includes 👍/👎 buttons. Configure Slack's Interactivity request URL
as `POST /slack/interactions`; signed button actions are persisted to the local
feedback repository under the authorized Slack user ID.

## Ingestion pipeline

All ingestors emit a common `DocumentChunk`. Delivery JSON is chunked around
50 tokens, supplier-contract PDF text around 800 tokens, and recipe HTML by
logical sections and steps. Dates become canonical UTC values; SKUs, store IDs,
suppliers, and product categories become consistent filter metadata.
Deterministic IDs support citations, and malformed inputs fail explicitly.

The bundled runtime indexes delivery logs and the synthetic oat-milk recipe.
Set `STORE_ASSISTANT_SUPPLIER_CONTRACT_PATH` to a synthetic supplier PDF or a
directory of PDFs to index them at startup; leaving it unset keeps contract
indexing disabled. Directory files are processed deterministically by name,
and duplicate contract document IDs are rejected rather than overwritten.

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

The GitHub Actions CI workflow runs these quality gates and the golden evaluation
on every push and pull request using Python 3.12.

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
Deploy the environment-selectable Okta OIDC, Amazon Titan/Bedrock, and
namespaced Pinecone adapters, plus an approved hosted reranking provider, via
validated environment configuration. Run ingestion as an idempotent background
job, use managed persistence and observability, and add timeouts, retries,
circuit breakers, readiness checks, autoscaling, and cost monitoring.

## Limitations

- Hosted provider clients depend on their optional SDK extras and valid cloud
  credentials/configuration.
- The local vector index is in-memory and rebuilt at startup; configured Pinecone
  records are upserted again during each startup.
- Local model substitutes have limited semantic/conversational quality.
- Slack handling is synchronous; production should acknowledge quickly.
- The small synthetic golden set is a regression signal, not real-world proof.
