# Whole Foods GenAI Assistant

Run the fully offline API from the repository root:

```bash
uvicorn store_assistant.runtime:app --app-dir src
```

The synthetic local identities use bearer tokens `local-brooklyn-token`,
`local-manhattan-token`, and `local-queens-token`. Runtime request logs and
feedback are stored under `.local/`.
