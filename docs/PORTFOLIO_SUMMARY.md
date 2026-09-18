# Portfolio Summary

> This is an independent portfolio project built with synthetic data. It is not a
> Whole Foods Market production system and does not represent proprietary work,
> infrastructure, results, or operational experience.

## One-line project description

A locally runnable enterprise RAG reference implementation for store-operations
questions, with identity-derived store isolation, filtered retrieval, reranking,
citations, evaluation, and swappable local or hosted providers.

## GitHub project description

Synthetic store-operations GenAI assistant demonstrating FastAPI, store-scoped RAG,
retrieve-then-rerank context selection, cited answers, provider abstractions, signed
Slack requests, feedback capture, and deterministic offline evaluation.

## LinkedIn project description

Built a synthetic enterprise GenAI/RAG reference implementation for store-operations
questions. The project uses authenticated store identity to constrain vector retrieval
before reranking and answer generation, then returns source citations and records request
telemetry and user feedback. A FastAPI service and optional signed Slack interface share
the same application workflow. Provider boundaries support deterministic local operation
as well as adapters for Amazon Bedrock/Titan, Pinecone, and Okta-compatible OIDC, while a
golden dataset evaluates retrieval, answer behavior, citations, isolation, and abstention.

## Resume bullets

- Designed a FastAPI-based enterprise RAG reference architecture with provider boundaries
  for authentication, embeddings, vector search, reranking, and answer generation,
  supporting a deterministic offline workflow and optional hosted-service adapters.
- Enforced store/tenant isolation by deriving `store_id` from authenticated identity and
  applying it as a metadata filter during vector retrieval, before reranking or generation;
  returned service-enforced citations with grounded answers.
- Built synthetic evaluation and automated tests covering retrieval, generation behavior,
  citations, abstention, and cross-store isolation, with request telemetry and thumbs-style
  feedback capture for operational review.

## Technologies demonstrated

- Python 3.12, FastAPI, Uvicorn, Pydantic, pytest, Ruff, and mypy
- Retrieval-augmented generation, embeddings, vector search, lexical reranking, and citations
- Local deterministic providers and adapters for Amazon Bedrock/Titan and Pinecone
- Mock authentication and an Okta-compatible OIDC adapter
- Slack slash commands, request-signature verification, and interactive feedback
- JSONL request telemetry, SQLite feedback storage, Docker Compose, and `uv`

## Engineering concepts demonstrated

- Identity-derived authorization boundaries and store-scoped metadata filtering
- Filter-first, retrieve-many/rerank-few retrieval orchestration
- Dependency inversion through provider protocols and composition at runtime
- Source-aware ingestion and normalization for JSON, PDF, and HTML inputs
- Grounded answer construction, citation enforcement, and explicit abstention
- Deterministic local development and reproducible synthetic evaluation
- Separation of HTTP and Slack adapters from the shared application service
- Configuration through environment variables and explicit provider selection
- Structured request telemetry, usage accounting, and user-feedback persistence
- Unit and integration testing of application, provider, and isolation behavior

## Recruiter keywords

Enterprise RAG, generative AI, LLM, FastAPI, Python, vector search, embeddings, Pinecone,
Amazon Bedrock, Amazon Titan, reranking, citations, multi-tenant isolation, OIDC, Okta,
Slack integration, observability, evaluation, hallucination mitigation, provider abstraction,
synthetic data, pytest, mypy, Ruff, Docker.

## What makes this project technically interesting

The project treats RAG as an application and security architecture rather than a single LLM
call. Store scope comes from authenticated identity instead of caller-supplied filters, and
the boundary is applied during vector search so out-of-scope chunks do not enter reranking or
generation. Retrieval deliberately gathers a wider candidate set and reranks it into a small,
high-signal context. Citations are attached by the service from selected chunks rather than
trusted to free-form model output.

The same orchestration can run entirely offline with deterministic components or use hosted
adapters without changing the core workflow. That makes behavior testable without cloud
credentials while still exposing realistic integration boundaries. The evaluation suite also
keeps retrieval and answer behavior visible, including whether the assistant abstains when the
available evidence cannot support an answer.

## Honest scope statement

This repository is a synthetic, locally runnable reference implementation created to
demonstrate architecture and engineering decisions. Its sample store, inventory, supplier,
contract, and recipe data are synthetic. It does not contain Whole Foods Market systems or
data, and it does not establish production scale, latency, cost, security certification,
business impact, or real-world deployment experience. The managed-service integrations are
adapter implementations and configuration paths; the default demonstration uses local
providers and requires no AWS, Pinecone, Okta, or Slack credentials.
