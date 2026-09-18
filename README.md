# Whole Foods Store Operations GenAI Assistant

> **Synthetic enterprise GenAI/RAG reference implementation.** This independent
> portfolio project uses synthetic data only. It is not a Whole Foods Market
> production system and does not use or describe proprietary infrastructure,
> internal data, confidential architecture, or measured business results.

A locally runnable assistant that lets a store manager ask natural-language
questions about inventory deliveries, supplier contracts, and recipes through
an HTTP API or an optional Slack interface. It demonstrates store-scoped RAG,
retrieve-then-rerank context selection, cited answers, feedback capture, and
swappable local or managed-service providers without requiring cloud accounts
for the default experience.

## Why this project matters

Enterprise RAG is more than connecting a vector database to an LLM. This project
shows the surrounding engineering needed for a credible assistant: identity-derived
data boundaries, metadata-filtered retrieval, citation enforcement, signed Slack
requests, observable provider usage, deterministic evaluation, and an offline path
that makes the complete workflow easy to review. The implementation emphasizes
testable interfaces and explicit failure handling rather than unsupported claims
about production performance.

## Architecture

```mermaid
flowchart TB
    subgraph Query[Query and answer path]
        Clients[Slack slash command or HTTP API] --> API[FastAPI]
        API --> Auth[Authentication and trusted store context]
        Auth --> Router[Query routing]
        Router --> EmbedQ[Query embedding]
        EmbedQ --> Retrieval[Store-scoped metadata retrieval]
        Retrieval --> Vectors[(Vector store)]
        Vectors --> Rerank[Reranking]
        Rerank --> LLM[Grounded answer provider]
        LLM --> Answer[Answer and citations]
        Answer --> Telemetry[JSONL request logging]
        Answer --> Feedback[SQLite thumbs feedback]
    end

    subgraph Ingestion[Ingestion and indexing path]
        Sources[Delivery JSON, supplier PDF, recipe HTML] --> Normalize[Normalization]
        Normalize --> Chunk[Source-aware chunking]
        Chunk --> EmbedD[Document embeddings]
        EmbedD --> Index[Vector indexing with metadata]
        Index --> Vectors
    end

    subgraph Providers[Environment-selectable provider implementations]
        Local[Local offline embeddings, vector search, reranker, and extractive answer]
        Managed[Amazon Bedrock and Titan, Pinecone, Okta-compatible OIDC]
    end

    Providers -. implement provider boundaries .-> Auth
    Providers -. implement provider boundaries .-> EmbedQ
    Providers -. implement provider boundaries .-> Vectors
    Providers -. implement provider boundaries .-> LLM
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

## Three-minute local demo

This path uses only bundled synthetic data and deterministic local providers—no
AWS, Pinecone, Okta, or Slack credentials are required. After completing the
[local setup](#local-setup), start the API:

```bash
uv run uvicorn store_assistant.runtime:app --app-dir src
```

In another terminal, ask the assistant as the synthetic Brooklyn manager:

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H 'Authorization: Bearer local-brooklyn-token' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Do we have oat milk?","filters":{"sku":"OAT001"}}'
```

The response reports Brooklyn's synthetic inventory, identifies the local model,
and includes both a deterministic delivery citation and a `request_id`. That one
request demonstrates the complete flow:

1. FastAPI validates the HTTP request and bearer token.
2. Mock authentication establishes the caller's trusted Brooklyn store context.
3. Query routing identifies the relevant source type and SKU filter.
4. Vector search retrieves candidates with the Brooklyn `store_id` constraint
   already applied.
5. The lexical reranker reduces the candidate set to the best evidence.
6. The local extractive answer provider produces a concise grounded response.
7. The answer service appends the selected document IDs as citations.
8. JSONL telemetry is written and the returned request can receive thumbs-style
   feedback.

Copy the returned `request_id` into this follow-up request:

```bash
curl -s http://127.0.0.1:8000/v1/feedback \
  -H 'Authorization: Bearer local-brooklyn-token' \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"<request-id>","rating":"up","comment":"Useful"}'
```

For an isolation check, repeat the query with `local-manhattan-token`: the same
SKU resolves against Manhattan's synthetic records rather than Brooklyn's. Use
`Ctrl+C` to stop the API. The optional Slack adapter exercises the same service
path with signed requests and interactive feedback; it is not needed for this
offline demo.

## Quickstart

### Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the locked local workflow
- Docker with Compose v2 only if you prefer the container path

Install the application and development dependencies from `uv.lock`:

```bash
uv sync --extra dev
```

No environment file or external credentials are needed for the offline path.
The checked-in defaults use mock authentication and local implementations for
embeddings, vector search, reranking, and answer generation. To customize data
or providers, export selected values documented in [`.env.example`](.env.example)
before startup; do not add credentials to source control.

Start FastAPI locally:

```bash
uv run uvicorn store_assistant.runtime:app --app-dir src
```

Check `http://127.0.0.1:8000/health`, then follow the
[three-minute demo](#three-minute-local-demo). The bundled mock tokens are
`local-brooklyn-token`, `local-manhattan-token`, and
`local-queens-token`.

### Docker Compose

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

## RAG lifecycle

1. **Ingest and normalize.** Source-specific loaders parse delivery JSON,
   supplier-contract PDFs, and recipe HTML into a shared `DocumentChunk`
   model. Dates, SKUs, store IDs, suppliers, and product categories are
   normalized into consistent metadata; malformed inputs fail explicitly.
2. **Chunk and embed.** Delivery records are chunked around 50 tokens,
   contract text around 800 tokens, and recipes by logical sections and steps.
   Each chunk receives a deterministic citation ID and an embedding.
3. **Index metadata with vectors.** Chunk text, vectors, citation IDs, and
   metadata are stored together so authorization and domain filters can be
   applied during vector search, before content reaches reranking or generation.
4. **Authenticate and route.** The application establishes the user's trusted
   store context, optionally infers a source type from the query, embeds the
   query, and constructs exact filters for `store_id`, `product_category`,
   `sku`, `supplier`, and `source_type`.
5. **Retrieve many, then rerank.** Vector search requests up to 25 filtered
   candidates by default. A reranker combines query relevance with the
   candidates and keeps at most five, producing a smaller, higher-signal
   context for the answer provider.
6. **Generate and cite.** The answer provider receives only the selected
   context. The answer service, rather than the provider, appends the selected
   chunks' deterministic IDs as citations to keep sources tied to retrieved
   evidence.
7. **Observe and improve.** JSONL telemetry records identity context, document
   IDs, retrieval and reranking scores, model and token usage, estimated cost,
   latency, and the final answer. Thumbs feedback is stored in SQLite with
   per-user authorization checks.

Store isolation is a data-access boundary, not merely a relevance hint. The
authenticated identity supplies `store_id`; a caller cannot override it with a
different store filter. Applying that metadata constraint inside vector search
prevents another store's chunks from becoming reranking or LLM context, even
when several stores carry the same SKU.

The bundled runtime indexes delivery logs and the synthetic oat-milk recipe.
Set `STORE_ASSISTANT_SUPPLIER_CONTRACT_PATH` to a synthetic supplier PDF or a
directory of PDFs to index them at startup; leaving it unset keeps contract
indexing disabled. Local feature-hash embeddings, cosine search, lexical
reranking, and extractive answers provide deterministic offline substitutes for
the managed provider implementations.

## Engineering quality and evaluation

- **Automated quality gates:** pytest, Ruff linting and formatting, strict mypy,
  and the golden-set evaluation run in
  [GitHub Actions](.github/workflows/ci.yml) on every push and pull request.
- **Layered verification:** unit and integration tests cover ingestion,
  normalization, chunking, provider boundaries, store/SKU filtering, retrieval,
  reranking, citations, authentication propagation, cross-store isolation,
  feedback, API errors, Slack signatures, and evaluation.
- **Reproducible local tooling:** [`uv.lock`](uv.lock) pins the complete dependency
  graph used by the documented `uv sync --extra dev` workflow.
- **Deployable artifact:** the [Dockerfile](Dockerfile) builds a non-root,
  health-checked Python 3.12 runtime, with Docker Compose available for local
  persistence and startup.

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src/store_assistant
uv run evaluate-store-assistant
```

The six bundled synthetic golden cases exercise three store-specific oat-milk
answers, an unknown-SKU abstention, a cross-store access abstention, and a recipe
answer. The evaluation runs those cases through the deterministic local retrieval
and answering pipeline and reports:

| Metric | What it checks |
| --- | --- |
| `retrieval_hit_rate` | Positive cases retrieve at least one expected document. |
| `correct_store_retrieval_rate` | Retrieved context is empty when expected or belongs only to the requested store. |
| `abstention_success_rate` | Expected-abstention cases return no retrieved context. |
| `answer_correctness` | Answers contain required phrases and omit forbidden phrases defined by each case. |
| `citation_correctness` | Positive answers cite expected evidence from the correct store; abstentions cite nothing. |

The report also includes average local latency and estimated cost per query.
Offline token cost is zero; the evaluation API accepts token prices for hosted
answer providers. Perfect scores on this tiny, deterministic synthetic golden
set are regression signals only—not evidence of real-world model accuracy,
generalization, latency, or production performance.

## Docker

```bash
docker build -t store-assistant .
docker run --rm -p 8000:8000 \
  -v store-assistant-state:/app/.local store-assistant
```

The Python 3.12 image runs as non-root and health-checks `GET /health`.

## Security considerations

This repository demonstrates security-conscious boundaries, but it is a reference
application—not a production-certified or fully hardened system.

### Implemented in this reference application

- **Identity-derived store isolation:** the authentication provider supplies a
  normalized `store_id`. The service rejects a conflicting caller-provided store
  filter and injects the trusted scope into vector search before reranking or
  generation.
- **Authentication boundary:** local mode uses explicit mock tokens; the hosted
  Okta-compatible adapter verifies RS256 access tokens against issuer JWKS and
  validates issuer, audience, expiry, and configured identity/store claims.
- **Slack request trust:** slash commands and interactions use constant-time v0
  HMAC verification over the raw body, reject timestamps outside a five-minute
  window, and require an authorized Slack-user-to-access-token mapping.
- **Input validation and limits:** Pydantic forbids unexpected HTTP fields and
  limits query and feedback-comment text to 2,000 characters. Slack additionally
  limits query text to 2,000 characters and request bodies to 16 KiB.
- **Controlled failure responses:** invalid authentication, retrieval input, and
  feedback return client errors; answer-provider failures are translated to
  `502` responses instead of unhandled transport errors.
- **Configuration and persistence boundaries:** provider settings are validated
  at startup, credentials are supplied through environment/SDK configuration,
  feedback is attributed to an authenticated user in SQLite, and append-only
  JSONL telemetry captures request context, selected document IDs and scores,
  model/token/cost fields, latency, and the final answer.

### Required for a real production deployment

Use a managed secret store and key rotation; TLS and encryption at rest; log
redaction, access controls, retention, and audit policies; rate limiting and
abuse protection; least-privilege IAM and network isolation; dependency and
container scanning; and provider timeouts, retries, circuit breakers, and
monitoring. Replace development tokens and local persistence, authorize access
to cited source documents, test tenant isolation end to end, and complete a
formal threat model and privacy review for the actual deployment environment.

## Production deployment approach

Build and scan the container in CI and deploy stateless workers behind TLS.
Deploy the environment-selectable Okta OIDC, Amazon Titan/Bedrock, and
namespaced Pinecone adapters, plus an approved hosted reranking provider, via
validated environment configuration. Run ingestion as an idempotent background
job, use managed persistence and observability, and add timeouts, retries,
circuit breakers, readiness checks, autoscaling, and cost monitoring.

## Portfolio resources

Use these companion guides to explore, present, and discuss the project:

- [Interview guide](docs/INTERVIEW_GUIDE.md) - 30-second and 2-minute explanations, architecture walkthrough, technical Q&A, and production considerations.
- [Design decisions](docs/DESIGN_DECISIONS.md) - architectural choices, tradeoffs, and production evolution.
- [Three-minute demo](docs/DEMO.md) - a practical offline walkthrough for interviews and recruiter demos.
- [Portfolio summary](docs/PORTFOLIO_SUMMARY.md) - recruiter-facing description, resume bullets, technologies, and engineering concepts.
- [GitHub setup](docs/GITHUB_SETUP.md) - recommended repository description, topics, and presentation settings.

## Limitations

- Hosted provider clients depend on their optional SDK extras and valid cloud
  credentials/configuration.
- The local vector index is in-memory and rebuilt at startup; configured Pinecone
  records are upserted again during each startup.
- Local model substitutes have limited semantic/conversational quality.
- Slack handling is synchronous; production should acknowledge quickly.
- The small synthetic golden set is a regression signal, not real-world proof.
