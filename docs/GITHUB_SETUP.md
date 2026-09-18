# GitHub repository setup

GitHub's **About** panel and repository topics are configured in the GitHub UI,
not in tracked files. Use the following metadata to make the project easier to
understand and discover without overstating its scope.

## About panel

**Description**

> Synthetic enterprise RAG reference implementation for store operations, with
> store-scoped retrieval, citations, evaluation, and an offline demo.

Leave the **Website** field empty unless this repository later gains a maintained
project page or live demo. Do not link to an unrelated personal site as though it
were a deployment of this application.

## Topics

```text
generative-ai
rag
llm
fastapi
python
enterprise-ai
retrieval-augmented-generation
aws-bedrock
pinecone
slack
```

The final three topics describe optional, implemented integrations; the default
demo remains fully local and does not require those services.

## Presentation checklist

- Keep the repository public if it is intended to serve as a portfolio sample.
- Enable the **Releases**, **Packages**, and **Deployments** panels only when the
  repository actually uses them.
- Pin the repository on the owner's profile and use the same synthetic-reference
  wording in any profile or portfolio description.
- Confirm that the default branch has the `CI` workflow passing before sharing
  the repository.
- Avoid adding claims about production use, proprietary Whole Foods systems, or
  measured business impact; this project uses synthetic data and is independent
  of Whole Foods Market.
