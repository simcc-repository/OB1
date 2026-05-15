# SIMCC fork patches for OB1 dashboard

Fork: `simcc-repository/OB1` branch `simcc-main` (off `NateBJones-Projects/OB1` `main`).

## Why

Deploy the OB1 Open Brain dashboard against our self-hosted OpenBrain
(REST at `http://192.168.50.148:8001`) instead of the upstream-assumed
Supabase Edge Function backend.

## Patches

| File | Change | Reason |
|---|---|---|
| `next.config.ts` | Add `output: "standalone"` | Required for the multi-stage Docker build to copy `.next/standalone` and serve via `node server.js`. Without this, the standalone bundle isn't emitted. |
| `Dockerfile` (new) | Multi-stage Node 20 alpine build | Upstream ships no Dockerfile; this image is built locally on the Docker host via Portainer Git stack. `NEXT_PUBLIC_API_URL` is a build arg because Next.js inlines `NEXT_PUBLIC_*` vars into the client bundle at build time. |
| `docker-compose.yml` (new) | Wires the image with env-templated build args + runtime env | Portainer Git stack root for the dashboard. Defaults to host port `3003` (3000 / 3001 / 3002 / 3005 / 3010 already taken on .148). |

## Authentication note

Upstream dashboard validates the user-entered "API key" by `GET /health` with
header `x-brain-key: <key>`. Our self-hosted OpenBrain (`pgvector/pgvector:pg17`
container, **not** Supabase) ignores `x-brain-key` entirely — `/health` always
returns 200. So any non-empty string works as the login key. The dashboard's
login is therefore a cookie gate, not real auth.

If real auth becomes needed, the OpenBrain REST source would need to enforce
`x-brain-key`. That's an upstream OpenBrain patch, not a dashboard patch.

## Applying upstream updates

```bash
cd OB1
git fetch upstream
git checkout simcc-main
git rebase upstream/main
# resolve conflicts if upstream changes next.config.ts; verify standalone still set.
git push --force-with-lease origin simcc-main
```

Then in Portainer → openbrain-dashboard stack → Pull and redeploy.
