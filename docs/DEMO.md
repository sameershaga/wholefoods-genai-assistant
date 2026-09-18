# Three-Minute Demo Guide

This script demonstrates the repository's deterministic, synthetic offline path.
It requires no AWS, Pinecone, Okta, or Slack credentials. Keep two terminals open
and the repository's `README.md` ready at the architecture diagram.

## Before the demo

From the repository root, install the locked application and development
dependencies:

```bash
uv sync --extra dev
```

In terminal 1, start the API:

```bash
uv run uvicorn store_assistant.runtime:app --app-dir src
```

Wait for `Application startup complete`. In terminal 2, optionally verify that
the service is ready:

```bash
curl -s http://127.0.0.1:8000/health
```

## 0:00-0:30 — State the problem

Say:

> This is a synthetic enterprise RAG reference implementation for store
> operations. A manager can ask about deliveries, supplier contracts, or
> recipes without sending an entire corpus to a model. The interesting part is
> the system around generation: authenticated store scope, filtered retrieval,
> reranking, citations, telemetry, feedback, and deterministic evaluation. It
> is an independent portfolio project, not a Whole Foods production system.

## 0:30-1:00 — Walk through the architecture

Show the Mermaid diagram in the README and trace the top path from left to
right. Say:

> FastAPI accepts an HTTP or optional Slack request. Authentication establishes
> a trusted store identity, then query routing adds domain filters. The query is
> embedded and the vector store applies the store metadata filter before any
> result reaches the reranker or answer provider. The service attaches citations
> from the selected chunks and records request telemetry; users can submit
> thumbs feedback. Provider boundaries allow local components to be replaced by
> Bedrock, Titan, Pinecone, and Okta-compatible implementations.

Emphasize that filtering is an isolation boundary, while reranking improves the
relevance of only the already-authorized candidates.

## 1:00-1:45 — Run a local query

In terminal 2, run:

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H 'Authorization: Bearer local-brooklyn-token' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Do we have oat milk?","filters":{"sku":"OAT001"}}'
```

Point out the synthetic Brooklyn result: 12 cartons, the
`delivery:DLV-BK-1001:0` citation, the `local-extractive-v1` model, and the
`request_id`. Say:

> The caller supplied a product filter, but not a store ID. The service derived
> Brooklyn from the authenticated token, retrieved Brooklyn evidence, reranked
> it, generated a grounded response, and returned a stable source identifier.

## 1:45-2:15 — Show store isolation

Repeat the same query with a different identity:

```bash
curl -s http://127.0.0.1:8000/v1/query \
  -H 'Authorization: Bearer local-manhattan-token' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Do we have oat milk?","filters":{"sku":"OAT001"}}'
```

Point out that Manhattan returns 3 cartons and the distinct
`delivery:DLV-MN-1001:0` citation. Say:

> The question and SKU are identical; only the authenticated identity changed.
> Store scope is injected before vector search, so Brooklyn evidence cannot
> become Manhattan's reranking or generation context.

## 2:15-2:40 — Run evaluation and tests

Run the deterministic evaluation:

```bash
uv run evaluate-store-assistant
```

The checked-in six-case synthetic dataset evaluates retrieval, correct-store
retrieval, citation correctness, answer correctness, and abstention. Results
are reproducible local fixture results, not production measurements or business
impact. If time allows, show the test suite command:

```bash
uv run pytest
```

## 2:40-3:00 — Close with production evolution

Say:

> The repository demonstrates the architecture and its replaceable boundaries,
> not a production deployment. A real rollout would require enterprise identity
> and authorization policy, managed secrets, durable telemetry and feedback,
> provider retry and timeout policies, rate limits, monitoring and alerting,
> data governance, adversarial security testing, and evaluation on approved
> representative data. The local providers make the design inspectable and
> testable without claiming production scale, latency, cost, or impact.

Stop terminal 1 with `Ctrl+C` after the demo.

## If something fails live

- **Dependency installation fails:** explain that `uv sync --extra dev` installs
  from the checked-in lockfile, then show the README architecture and tests
  while using the expected responses below as a walkthrough.
- **The API does not start:** check that port 8000 is free. If it is occupied,
  add `--port 8001` to the Uvicorn command and replace `8000` with `8001` in
  both curl commands.
- **A curl command cannot connect:** show terminal 1 to confirm startup, then run
  the health request. Explain the request path from authentication through
  retrieval while the service restarts.
- **Terminal output is hard to read:** focus on the four response fields:
  `answer`, `citations`, `model`, and `request_id`. Brooklyn should reference 12
  cartons and `DLV-BK-1001`; Manhattan should reference 3 cartons and
  `DLV-MN-1001`.
- **Evaluation or tests fail:** do not claim they pass. Show the failing output,
  explain that the local suite is deterministic, and discuss how the failure
  would block promotion until diagnosed.
- **Time runs short:** run the two query commands only. Together they demonstrate
  the end-to-end RAG path, citations, and identity-derived isolation.
