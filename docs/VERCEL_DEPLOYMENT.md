# Vercel deployment

This guide deploys the repository as a deterministic public portfolio demo through
Vercel's GitHub integration. It targets a Vercel Hobby account and does not claim
that a deployment has already succeeded.

## Prerequisites

- A GitHub account containing this repository, with the intended production code
  on `main`.
- A Vercel Hobby account authorized to read that GitHub repository.
- No AWS, Pinecone, Okta, Slack, or other hosted-service account is needed for the
  deterministic public demo.

The demo uses only bundled synthetic data and local deterministic providers. **No
environment variables are required.** In particular, do not set
`STORE_ASSISTANT_API_URL` in Vercel: the browser should use same-origin `/v1/*`
requests, which `next.config.ts` rewrites to the bundled Python function.

## Import the GitHub repository

1. In the [Vercel dashboard](https://vercel.com/new), choose **Add New → Project**.
2. Connect GitHub if necessary, then select **Import** beside this repository.
3. Choose the personal Hobby scope.
4. Before deploying, confirm these project settings:

   | Setting | Value |
   | --- | --- |
   | Framework Preset | `Next.js` (normally auto-detected) |
   | Root Directory | repository root (`.`; leave the field empty) |
   | Build Command | `npm run build` (detected from `package.json`) |
   | Output Directory | Next.js default (do not override) |
   | Install Command | Vercel default (do not override) |
   | Environment Variables | none for deterministic demo mode |

The repository root is essential: the Next.js app, `api/index.py`, `pyproject.toml`,
`uv.lock`, `src/`, and bundled `data/` must share the project root. Vercel installs
Node dependencies from `package-lock.json` and the Python function dependencies
from the root Python project metadata. Locally, `npm ci` remains the reproducible
frontend installation check.

Vercel's Python runtime recognizes the ASGI `app` exported by `api/index.py`.
That adapter adds the bundled `src` directory to Python's import path before
importing `store_assistant.runtime`; it does not depend on the repository package
being installed as a wheel. It includes the existing FastAPI router with an `/api`
prefix so Vercel can discover concrete full-path routes for the Python function.
In production, the complete request path is:

```text
browser /v1/demo/*
  -> Next.js rewrite /api/v1/demo/*
  -> api/index.py concrete FastAPI route /api/v1/demo/*
  -> existing route handler and dependencies
```

The adapter pins bundled data paths relative to the repository, so its delivery
JSON and recipe HTML do not depend on the function's working directory.

## First deployment and verification

Choose **Deploy** and wait for both the build and deployment to finish. This action
performs the first real deployment validation; the repository itself does not
assert that Vercel has already deployed it successfully.

After Vercel supplies a deployment URL, verify it in this order:

1. Open the deployment URL and confirm the homepage and assistant console render.
2. Open `https://<deployment-host>/v1/demo/stores`. Expect JSON listing exactly
   `BROOKLYN-01`, `MANHATTAN-01`, and `QUEENS-01`. Mock access tokens must not be
   present in the response.
3. In the console, select **Brooklyn**, ask `Do we have oat milk?`, and confirm the
   response is scoped to `BROOKLYN-01`, reports the bundled Brooklyn result, and
   includes citations.
4. Select **Manhattan**, ask the same question, and confirm the result and citations
   change to Manhattan's synthetic records. A Brooklyn citation must not appear in
   the Manhattan response, or vice versa.
5. Submit thumbs-up or thumbs-down feedback and confirm the UI reports that it was
   saved. This verifies the request path only; it does not prove durable storage.

For a direct API smoke test, substitute the deployment host:

```bash
curl -fsS https://<deployment-host>/v1/demo/stores

curl -fsS https://<deployment-host>/v1/demo/query \
  -H 'Content-Type: application/json' \
  -d '{"store_id":"BROOKLYN-01","query":"Do we have oat milk?"}'

curl -fsS https://<deployment-host>/v1/demo/query \
  -H 'Content-Type: application/json' \
  -d '{"store_id":"MANHATTAN-01","query":"Do we have oat milk?"}'
```

## Serverless state limitations

The Vercel adapter writes request telemetry and feedback under
`/tmp/store-assistant`. Vercel Functions have a read-only deployed filesystem and
writable temporary scratch space. This state is **ephemeral and non-durable**:

- a cold start may begin without records from an earlier instance;
- concurrent function instances do not share their SQLite or JSONL files;
- redeployments do not migrate or preserve those files; and
- Vercel may recycle an instance and discard its temporary files at any time.

Therefore feedback acknowledgement means only that the active function instance
accepted the record. Do not use the demo as a feedback system of record, an audit
log, or production telemetry store. The demo intentionally does not add an
external database.

## Preview and production deployments

Once Git integration is connected, pushes to non-production branches and pull
requests create Preview Deployments. Use their unique URLs to repeat the smoke
tests above without changing production. A push or merge to the configured
production branch (`main`) creates a Production Deployment automatically. Review
the project's **Settings → Git** page if the production branch differs from
`main`.

To recover from a bad deployment, open the project **Deployments** list and promote
a previously working deployment or redeploy a known-good commit. Promotion or
redeployment changes what is served; it does not restore ephemeral `/tmp` feedback
or telemetry. Consult Vercel's current dashboard prompts before confirming the
operation.

## Troubleshooting

- **Framework detected as Other:** set the Framework Preset to **Next.js** and
  redeploy. Do not select FastAPI as the project-level preset; FastAPI is the
  Python function inside the same Next.js project.
- **Frontend builds but `/v1/demo/stores` returns 404:** confirm the Root Directory
  is the repository root, `api/index.py` is present in the deployed commit, and no
  `STORE_ASSISTANT_API_URL` variable overrides same-origin routing. Check the
  function logs for `/api/index.py`.
- **`ModuleNotFoundError: store_assistant`:** confirm the deployed commit contains
  the `api/index.py` source-path bootstrap and the root contains `src/store_assistant`.
- **Python dependency import failure:** confirm `pyproject.toml` and `uv.lock` are at
  the configured Root Directory and inspect the deployment's Python build logs.
- **Bundled data file error:** confirm `data/delivery_logs.json` and
  `data/recipes/oat_milk_overnight_oats.html` are tracked and included in the
  deployment.
- **Homepage works locally but queries fail locally:** Vercel-only rewriting is
  inactive outside Vercel. Start FastAPI separately and set
  `STORE_ASSISTANT_API_URL=http://127.0.0.1:8000` before starting Next.js.
- **Feedback disappears:** this is expected serverless behavior; see the `/tmp`
  limitations above.
- **A build setting was changed:** settings changes apply on the next deployment;
  redeploy after correcting the setting.

For platform behavior that may change, refer to Vercel's current documentation for
[build configuration](https://vercel.com/docs/builds/configure-a-build),
[Python Functions](https://vercel.com/docs/functions/runtimes/python), and
[Git-based deployments](https://vercel.com/docs/deployments/git).
