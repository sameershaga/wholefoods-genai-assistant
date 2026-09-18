# Interview guide

Use this guide to explain what the repository actually demonstrates. It is a
synthetic enterprise GenAI/RAG reference implementation, not a Whole Foods
Market production system or evidence of production results.

## 30-second explanation

“This project is a locally runnable store-operations RAG assistant built with
FastAPI. It retrieves synthetic delivery, recipe, and optional contract data,
but first derives a store boundary from the authenticated user and applies it
inside vector search. It then reranks the authorized candidates, generates a
grounded answer with citations, and records telemetry and feedback. Provider
interfaces make the default deterministic local stack replaceable with Okta,
Amazon Bedrock and Titan, and Pinecone adapters.”

## 2-minute explanation

“The problem is that a useful operational assistant must answer from current
business documents without leaking one store’s data into another store’s
response. Direct generation does not provide either grounding or a reliable
data-access boundary.

“The application exposes a FastAPI query endpoint and an optional Slack
interface. Authentication creates a trusted user context containing the store
identity. A lightweight router can infer the source type, and the retrieval
service embeds the query and searches the vector store with the authenticated
`store_id` plus any allowed domain filters. That filtering happens before any
candidate reaches the reranker or answer provider.

“Retrieval takes up to 25 candidates and reranks them down to at most five.
The answer provider sees only that smaller context. The answer service owns the
citations, tying the response to the selected document IDs, and it abstains
when retrieval returns no evidence. Each completed request records selected
documents, scores, model and token fields, estimated cost, latency, and the
answer; users can also submit thumbs feedback.

“The repository includes deterministic local implementations so the whole path
can be tested offline, plus adapters for Pinecone, Titan embeddings, Bedrock
generation, and Okta-compatible OIDC. A six-case synthetic golden set evaluates
retrieval, correct-store behavior, answers, citations, and abstention separately.
The result is a testable reference architecture, not a claim of deployed scale
or real-world model accuracy.”

## Architecture walkthrough

1. **FastAPI:** `api.py` provides health, query, and feedback endpoints with
   typed Pydantic request and response models. The optional Slack router is
   mounted into the same application only when configured.
2. **Authentication and store identity:** a bearer token is resolved by either
   the fixed local token map or the Okta-compatible OIDC adapter. Both return a
   normalized `UserContext`; store scope comes from that trusted context, not
   from the question or a caller-controlled filter.
3. **Query routing:** `infer_source_type` recognizes inventory, recipe, and
   contract intent. It supplies a source filter only when one was not already
   provided and leaves ambiguous questions unrestricted by source type.
4. **Embeddings:** the query and indexed chunks share an embedding provider.
   Local mode uses deterministic feature hashing; the hosted adapter invokes
   Titan Text Embeddings V2 through Bedrock Runtime.
5. **Metadata-filtered retrieval:** the authenticated service rejects a
   conflicting `store_id`, injects the trusted store, and passes exact filters
   such as SKU, supplier, product category, and source type into vector search.
6. **Vector store:** the in-memory implementation performs local cosine search.
   The Pinecone adapter preserves the vector-store interface, metadata filters,
   optional namespace, and stored chunk text.
7. **Reranking:** vector search returns a broad candidate set; the deterministic
   local lexical reranker scores only those authorized candidates and selects a
   smaller answer context.
8. **Answer generation:** local mode extracts the strongest passage. The
   Bedrock adapter uses the Converse API. Empty context produces an abstention
   rather than asking a provider to improvise.
9. **Citations:** `AnswerService`, not the model, associates the final response
   with the selected chunks’ deterministic document IDs.
10. **Logging:** an append-only JSONL repository records request identity and
    store context, retrieved document IDs and scores, model/token/cost fields,
    latency, and the final answer.
11. **Feedback:** authenticated HTTP users and authorized Slack users can save
    thumbs feedback in SQLite. Repeated feedback from the same user for the same
    request replaces that user’s previous rating.
12. **Slack integration:** signed slash commands use the same transport-neutral
    assistant service. The handler checks Slack’s v0 HMAC, rejects requests
    outside a five-minute window, limits input size, and maps an authorized
    Slack user to an assistant access token. Responses are ephemeral and expose
    thumbs buttons through the interactions endpoint.

## Explain important decisions

### Why RAG?

The answer should be grounded in repository data rather than model memory. RAG
makes evidence inspectable, citeable, independently updateable, and measurable;
it also gives the system a defined abstention path when no evidence is found.

### Why not send all data to the LLM?

Sending everything would weaken the store boundary, increase prompt size and
cost, and introduce irrelevant context. Retrieval selects a small authorized
subset before generation. This repository does not treat a prompt instruction
as authorization.

### Why metadata filtering?

Semantic similarity answers “what is relevant,” not “what is permitted.” Exact
metadata filters enforce trusted store scope and narrow by structured domain
fields that embeddings should not be expected to enforce reliably.

### Why filter before reranking?

A reranker reads candidate content. Filtering later would expose out-of-scope
documents to that component and waste work scoring candidates that can never be
used. Here, only authorized search results become reranker inputs.

### Why retrieve many, then rerank?

The first pass favors recall; the second pass produces a smaller, higher-signal
context. The defaults—up to 25 candidates and five final chunks—are explicit
reference settings that should be tuned on representative production data.

### Why citations?

Citations let a user or evaluator trace an answer to retrieved evidence. The
service derives them from selected document IDs instead of trusting generated
citation text, although production would still need source-document access
authorization and a user verification workflow.

### Why provider abstractions?

Narrow interfaces keep orchestration independent of authentication, embedding,
vector, reranking, and generation vendors. They improve testability and allow
local and hosted implementations, with the tradeoff that contract tests are
needed to catch provider-specific behavior.

### Why deterministic local providers?

They make setup credential-free, tests reproducible, and failures debuggable.
They are intentionally simple substitutes, not proxies for hosted-model quality,
latency, persistence, or scale.

### Why FastAPI?

FastAPI supplies a small typed HTTP boundary, dependency injection, request
validation, generated API documentation, and easy in-process testing. It fits
the service-oriented Python implementation without coupling domain services to
HTTP or Slack.

### Why does store isolation matter?

Operational records are scoped to a store. A relevant but unauthorized answer
is still a security failure. The application therefore derives `store_id` from
identity, rejects conflicting filters, and applies it before content reaches
reranking or generation.

### Why evaluate abstention?

An assistant must know when the indexed evidence cannot support an answer.
Testing only positive questions rewards always answering, which can encourage
hallucination or cross-scope guesses. The golden set includes unknown and
cross-store cases that should retrieve nothing and cite nothing.

### Why separate retrieval and generation evaluation?

Retrieval can fail even when an answer sounds plausible, and generation can
misuse good context. Separate retrieval-hit, store-isolation, answer-content,
and citation metrics localize regressions and prevent one aggregate score from
hiding the failing stage.

## Likely interviewer questions

1. **What makes this RAG rather than document search?** The system retrieves
   and reranks chunks, then passes the selected evidence to an answer provider;
   it returns a generated or extractive answer plus service-owned citations.
2. **How do local embeddings work?** The local provider uses deterministic
   feature hashing into a configurable vector dimension. It is repeatable and
   offline, but intentionally less semantic than a learned embedding model.
3. **How is Pinecone used?** `PineconeVectorStore` adapts an injected index for
   upsert and query, passes exact metadata filters and an optional namespace,
   and reconstructs search results from stored metadata. It requires an
   existing index with dimensions matching the embedding provider.
4. **What do Bedrock and Titan do here?** Titan Text Embeddings V2 is the
   optional embedding adapter; Bedrock Converse is the optional answer adapter.
   Both sit behind injected, narrow interfaces so their error and response
   handling can be unit-tested without live cloud calls.
5. **Why include reranking after vector similarity?** Vector similarity is a
   candidate-discovery signal. Reranking uses the query and candidate text to
   choose a smaller context, balancing first-pass recall against prompt noise.
6. **Can a client request another store’s records?** No. The service rejects a
   conflicting caller-supplied store filter and overwrites the effective scope
   with the normalized store from authenticated identity.
7. **Is metadata filtering enough for multitenancy?** It is the isolation
   mechanism demonstrated here, but production risk requirements may warrant
   stronger namespace, index, account, or infrastructure separation plus
   adversarial isolation tests and audits.
8. **How does Okta authentication work?** The hosted adapter uses an injected
   verifier for RS256 access tokens and requires issuer, audience, subject, and
   a configurable store claim. Runtime wiring uses issuer JWKS. Claim governance,
   key caching/rotation, and broader authorization policy still need production
   design.
9. **How is Slack secured?** The handler verifies the raw-body v0 HMAC in
   constant time, enforces a five-minute timestamp tolerance and body/query
   limits, and accepts only Slack users in an explicit user-to-token mapping.
   That mapping is demonstration-grade and should become managed identity
   federation or an approved authorization service.
10. **What is observable?** Completed requests log user/store context, selected
    document IDs, retrieval and reranking scores, model, token counts, estimated
    cost, latency, and final answer. Production needs managed telemetry,
    redaction, retention, alerting, and access control.
11. **What does evaluation measure?** The six synthetic cases report retrieval
    hit rate, correct-store retrieval, abstention, answer-content checks,
    citation correctness, average local latency, and configured estimated cost.
    These are regression checks, not real-world benchmarks.
12. **How are failures handled?** Invalid authentication and retrieval input
    become client errors, and answer-provider failures become HTTP `502`
    responses. Providers validate malformed responses. Production still needs
    explicit timeouts, retries, circuit breakers, idempotency where applicable,
    and operational alerting.
13. **How would this scale?** Move ingestion to idempotent background jobs, use
    a durable vector service and managed telemetry/feedback stores, run stateless
    API workers behind a load balancer, cache safely, and autoscale based on
    measured demand. The repository itself makes no scale claim.
14. **What blocks production deployment?** The synthetic data, local providers,
    startup indexing, simple identity mapping, synchronous Slack handling, and
    local JSONL/SQLite persistence are reference choices. Governance, security,
    resilience, deployment, and quality work remains.
15. **How would you control cost?** Measure tokens and provider calls, tune
    candidate and context counts, set model and token limits, cache only where
    authorization and freshness permit, enforce quotas/budgets, and evaluate
    cheaper models. The local configured token cost is zero and is not a cloud
    cost estimate.
16. **How does the system mitigate hallucination?** It restricts generation to
    retrieved store-scoped context, abstains on empty retrieval, uses
    deterministic evidence citations, and evaluates required and forbidden
    answer content. It does not eliminate hallucination; production needs larger
    evaluations, model safety controls, monitoring, and user verification.
17. **What happens when data changes?** The local runtime rebuilds its in-memory
    index at startup. A production system needs governed incremental ingestion,
    document versioning, freshness objectives, deletion propagation, and
    repeatable reindexing or migration procedures.
18. **Why keep retrieval and transport separate?** HTTP and Slack both call the
    same `AssistantService`, so authentication scope, retrieval, answering, and
    telemetry have one implementation and can be tested without a network.

## Production discussion

Before real enterprise use, replace mock tokens and the static Slack mapping
with governed identity and authorization; define claim ownership and roles;
complete a threat model, privacy review, and end-to-end tenant-isolation tests;
and use managed secrets, TLS, encryption, least-privilege IAM, and network
controls. Logs and feedback need access controls, redaction, retention, audit,
and deletion policies.

Move ingestion out of API startup into idempotent jobs with source ownership,
schema validation, versioning, freshness monitoring, and deletion propagation.
Provision durable vector infrastructure and managed operational storage. Add
provider timeouts, retries, backoff, circuit breakers, quotas, readiness checks,
rate limits, abuse protection, alerts, and disaster-recovery procedures. Slack
should acknowledge within its required window and complete longer work
asynchronously.

Finally, evaluate representative, approved data and realistic adversarial
queries. Tune retrieval and reranking, compare hosted providers, test prompt
injection and cross-store attacks, establish human review and escalation, track
quality/latency/cost by version, and define release thresholds before enabling
operational decisions.

## Limitations and honest scope

- All bundled business data and golden cases are synthetic.
- This is an independent portfolio reference implementation, not Whole Foods
  Market production work and not a description of proprietary systems.
- The local embedding, vector store, reranker, authentication, and extractive
  answer components optimize for deterministic demonstration, not production
  quality or capacity.
- Hosted adapters are implemented and tested through injected clients, but the
  repository does not demonstrate a live managed deployment.
- The small golden set supports regression testing only. It does not establish
  business impact, generalization, production accuracy, scale, latency, cost,
  uptime, or a security certification.

In an interview, describe the architectural choices and code that can be shown.
Do not present this repository as evidence of production deployment, employer
experience, proprietary knowledge, or measured operational outcomes.
