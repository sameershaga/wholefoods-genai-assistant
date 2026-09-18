# Design decisions

This document records the principal choices demonstrated by this synthetic
store-operations assistant. It describes the repository as implemented, not a
production system or a record of Whole Foods architecture.

## RAG rather than direct generation

**Context.** Operational answers should be based on the bundled delivery,
recipe, and optional contract documents rather than on a model's general
knowledge.

**Decision.** Embed each query, retrieve relevant chunks, rerank them, and give
only the selected context to the answer provider.

**Why.** Retrieval makes the evidence explicit, updateable independently of a
model, and available for citation and evaluation. Empty retrieval also gives
the system a clear reason to abstain.

**Tradeoffs.** RAG adds ingestion, indexing, retrieval tuning, and more failure
points. It cannot compensate for missing, stale, or poorly chunked source data.

**Production evolution.** Add governed ingestion, freshness controls, document
versioning, quality monitoring, and broader evaluation before relying on the
answers operationally.

## Store-scoped metadata filtering

**Context.** The corpus contains store-specific records, and callers must not
retrieve another store's content.

**Decision.** Authentication produces a normalized `store_id`. The application
rejects a conflicting caller-supplied store filter and injects the trusted
identity scope into every vector query.

**Why.** Authorization is enforced in the retrieval path rather than left to a
prompt or to post-processing of generated text.

**Tradeoffs.** Every indexed store-specific chunk needs correct metadata, and
the approach depends on trustworthy identity claims and vector-store filtering.

**Production evolution.** Validate isolation with adversarial tests and audits,
apply least-privilege service access, and consider separate indexes or
namespaces where risk requirements justify stronger physical boundaries.

## Filtering before reranking

**Context.** A reranker can inspect candidate text, so filtering candidates
after reranking would expose unauthorized content to another component.

**Decision.** Exact-match metadata filters are passed into vector search. Only
the resulting authorized candidates are converted to reranker inputs.

**Why.** This preserves the store boundary throughout retrieval and avoids
spending reranking work on ineligible documents.

**Tradeoffs.** Strict filters can reduce recall when metadata is incomplete or
incorrect, and hosted stores must implement filter semantics consistently.

**Production evolution.** Enforce metadata schemas during ingestion, monitor
empty-result rates, and test the hosted vector store's filtering behavior.

## Retrieve many, rerank few

**Context.** Vector similarity is useful for candidate discovery, while a
second relevance pass can choose a smaller answer context.

**Decision.** `RetrievalService` retrieves up to 25 candidates by default and
reranks them down to five context chunks.

**Why.** A broader first pass protects recall while the narrower second pass
limits irrelevant context sent to answer generation.

**Tradeoffs.** Larger candidate sets increase vector and reranking work; small
final contexts can omit useful evidence. The repository's lexical reranker is a
deterministic demonstration, not a claim of state-of-the-art ranking quality.

**Production evolution.** Tune both counts on representative queries and
compare lexical, cross-encoder, or managed rerankers using retrieval metrics,
latency, and cost.

## Provider abstractions

**Context.** Local development should not require cloud credentials, while the
architecture should demonstrate replaceable managed integrations.

**Decision.** Protocol boundaries separate authentication, embeddings, vector
storage, reranking, answer generation, request logging, and feedback storage
from their concrete implementations.

**Why.** The pipeline can be tested with deterministic implementations and
configured with hosted adapters without rewriting orchestration logic.

**Tradeoffs.** Interfaces add code and can conceal provider-specific features.
The abstraction is only as portable as the behavior its adapters actually
share.

**Production evolution.** Add contract tests, explicit timeout and retry
policies, capability/version metadata, and operational health signals for each
provider.

## Deterministic local providers

**Context.** The project needs a credential-free path for development, tests,
evaluation, and interviews.

**Decision.** Local mode uses hash-based embeddings, an in-memory vector store,
a lexical reranker, a mock token map, and an extractive answer provider.

**Why.** These components are fast, repeatable, offline, and make failures easy
to reproduce.

**Tradeoffs.** They are intentionally simplified. Their relevance, language
understanding, persistence, and answer quality do not represent managed models
or a production index.

**Production evolution.** Keep deterministic providers for tests while running
separate evaluations against approved hosted providers and production-like
data.

## Pinecone adapter

**Context.** An in-memory index is rebuilt at startup and does not provide a
durable, scalable hosted retrieval service.

**Decision.** `PineconeVectorStore` implements the vector-store boundary using
an injected Pinecone index, exact metadata filters, optional namespace, and
stored chunk text in metadata.

**Why.** It demonstrates how hosted vector search can replace the local store
without changing retrieval orchestration and remains unit-testable without a
live account.

**Tradeoffs.** It requires a pre-existing compatible index, matching embedding
dimensions, credentials, and the optional SDK. Storing text as metadata has
size, governance, and cost implications.

**Production evolution.** Provision indexes through controlled infrastructure,
define namespace and retention policy, validate quotas and filter behavior,
and add retry, timeout, monitoring, and migration procedures.

## Bedrock and Titan adapters

**Context.** The local embedding and extractive answer providers are useful for
repeatability but not representative of hosted semantic models.

**Decision.** Environment-selectable adapters support Amazon Titan Text
Embeddings V2 through Bedrock Runtime and answer generation through Bedrock's
Converse API. SDK clients are injected into the adapters.

**Why.** Injection keeps cloud calls testable, while the shared interfaces keep
provider selection out of the retrieval and answering services.

**Tradeoffs.** Hosted use introduces credentials, regional and model access,
network failures, latency, and usage cost. Embedding dimensions must match the
vector index.

**Production evolution.** Add IAM scoping, model allowlists, quotas, retries,
timeouts, safety policy, cost budgets, and provider-specific quality tests.

## Okta-compatible OIDC abstraction

**Context.** Store scope must come from a verified identity rather than an
untrusted request field.

**Decision.** The hosted authentication path verifies an RS256 access token
against issuer JWKS, issuer, and audience, then maps subject and a configurable
store claim into `UserContext`. Local mode uses a fixed token map.

**Why.** Downstream services receive a small, normalized identity object and do
not need to understand token mechanics.

**Tradeoffs.** Correctness depends on identity-provider claim governance, key
availability, and token verification configuration. The local mock is not a
security substitute.

**Production evolution.** Define claim ownership and authorization policy,
handle key rotation and caching, add role/permission checks where needed, and
integrate security monitoring and access reviews.

## FastAPI

**Context.** The assistant needs a typed HTTP boundary for health, query,
feedback, and optional Slack endpoints.

**Decision.** FastAPI provides request validation, dependency wiring, routing,
and an ASGI application around transport-neutral services.

**Why.** It keeps the API concise and testable while generating a conventional
Python service interface.

**Tradeoffs.** Framework validation does not supply deployment hardening,
capacity management, or provider resilience. The current hosted calls and
Slack handling are synchronous.

**Production evolution.** Run behind managed ingress with TLS, rate limits and
request-size controls; add async/background handling where appropriate,
timeouts, graceful shutdown, and service-level monitoring.

## Slack interface

**Context.** Store users may prefer an existing collaboration surface over a
custom client.

**Decision.** The optional Slack adapter verifies v0 HMAC signatures and a
five-minute timestamp window, maps trusted Slack user IDs to access tokens, and
reuses the same assistant and feedback services as HTTP.

**Why.** Transport-specific trust checks stay at the edge while identity scope,
retrieval, citations, logging, and feedback follow one application path.

**Tradeoffs.** The configured user-to-token JSON map is suitable only for a
demonstration. Synchronous handling can conflict with Slack response deadlines,
and Slack adds another identity-mapping lifecycle.

**Production evolution.** Use secure token storage and managed identity
linking, acknowledge requests quickly and process work asynchronously, rotate
secrets, and audit installation scopes and workspace access.

## Environment-driven configuration

**Context.** Local and hosted adapters require different settings, but secrets
and deployment choices must not be committed to source.

**Decision.** `LocalSettings.from_environment` selects providers, paths, model
settings, state storage, and optional Slack configuration, validating required
combinations at startup.

**Why.** One composition root can support a fully offline default and explicit
hosted alternatives.

**Tradeoffs.** Environment variables are stringly typed, can become numerous,
and do not by themselves provide secret management or configuration history.

**Production evolution.** Integrate a secrets manager and validated deployment
configuration, separate secret from non-secret settings, and record safe
configuration versions for rollback and audit.

## Deterministic synthetic evaluation

**Context.** Retrieval quality, isolation, answer content, citations, and safe
abstention need repeatable checks without external services.

**Decision.** A bundled synthetic golden dataset runs through the local
embedding, vector, routing, reranking, and answering pipeline. It reports
positive retrieval hit rate, abstention success, correct-store retrieval,
answer correctness, citation correctness, latency, and configured estimated
cost.

**Why.** Determinism makes regressions attributable and keeps the evaluation
runnable in any development environment. Separate positive-hit and abstention
metrics prevent an empty answer from looking successful merely because no
document was expected.

**Tradeoffs.** The dataset is small and synthetic, local latency is not hosted
latency, and phrase checks are a limited proxy for answer quality.

**Production evolution.** Build versioned, representative and adversarial test
sets; add human review, slice metrics, hosted-provider runs, thresholds, and
ongoing drift analysis.

## Citations

**Context.** Users and evaluators need to connect an answer to retrieved
evidence, and a model should not control whether source identifiers appear.

**Decision.** `AnswerService` appends the selected chunks' deterministic
document IDs as citations. With no context, it returns an abstention with no
citations.

**Why.** Service-owned citation assembly ties references directly to the
retrieval result and behaves consistently across answer providers.

**Tradeoffs.** A document ID establishes provenance, not that every generated
claim is entailed by that source. Chunk-level citations may also be less usable
than links and highlighted passages.

**Production evolution.** Add resolvable source links, passage highlighting,
document versions, claim-to-evidence checks, and access-controlled source
viewing.

## Request telemetry

**Context.** Debugging a RAG request requires visibility beyond the final text.

**Decision.** `AssistantService` records request and identity IDs, store scope,
selected document IDs and both scores, model, token counts, configured cost
estimate, latency, and final answer in an append-only local JSONL repository.

**Why.** The record connects retrieval and generation behavior for local
diagnostics and evaluation of failures.

**Tradeoffs.** JSONL is a local demonstration sink, not a scalable telemetry
platform. Queries, identities, and answers can be sensitive and require careful
handling.

**Production evolution.** Use structured centralized observability with access
controls, redaction, retention limits, correlation IDs, dashboards, alerts, and
documented privacy policy.

## Feedback collection

**Context.** Explicit user feedback can identify unhelpful answers and link
them to request telemetry.

**Decision.** HTTP and Slack accept thumbs-up/down feedback tied to a
`request_id` and authenticated user. Local mode persists the latest rating and
optional comment per user and request in SQLite.

**Why.** The data model prevents one user's rating from overwriting another's
and supports updating a prior vote while retaining its creation time.

**Tradeoffs.** Thumbs feedback is sparse, subjective, and vulnerable to
selection bias. SQLite is not intended as the enterprise system of record.

**Production evolution.** Move feedback to a governed durable store, define
retention and moderation, join it safely to telemetry, and combine it with
reviewed quality signals rather than treating it as ground truth.
