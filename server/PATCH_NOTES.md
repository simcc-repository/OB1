# SIMCC patches to server/index.ts

Fork: `simcc-repository/OB1` branch `simcc-main`.

## Why

Route OB1's `open-brain-mcp` Edge Function through our self-hosted LiteLLM
gateway (`http://litellm:4000/v1`) instead of OpenRouter's cloud endpoint.
LiteLLM serves the same OpenAI-compatible API but lets us:

- Use **`bge-m3`** (1024-dim, local on gx10-3 LM Studio) for embeddings —
  free and private vs OpenRouter's paid `text-embedding-3-small` (1536-dim).
- Use **`gemma-4-e4b-it`** (local on gx10-3) for metadata extraction — free
  vs OpenRouter's `gpt-4o-mini`.
- Trace every call in Langfuse for prompt + cost analysis.

## Patches

1. **`OPENROUTER_BASE`** is now `Deno.env.get("OPENROUTER_BASE_URL")` (default
   unchanged: `https://openrouter.ai/api/v1`). Lets us point at LiteLLM by
   setting that env var.
2. **`OPENROUTER_EMBED_MODEL`** is now `Deno.env.get("OPENROUTER_EMBED_MODEL")`
   (default unchanged: `openai/text-embedding-3-small`). Lets us swap to
   `bge-m3`.
3. **`OPENROUTER_LLM_MODEL`** is now `Deno.env.get("OPENROUTER_LLM_MODEL")`
   (default unchanged: `openai/gpt-4o-mini`). Lets us swap to `gemma-4-e4b-it`.
4. **`extractMetadata` JSON parsing** strips optional markdown code fences
   (```json … ```) before `JSON.parse`. Claude and Gemma via openai-compatible
   proxies often wrap JSON in fences even when `response_format: json_object`
   is requested; without this the catch fell back to default tags.

## Schema impact

The thoughts table uses `vector(1024)` instead of upstream's `vector(1536)` to
match bge-m3 dimensions. Apply the OB1 bootstrap SQL with `1536` → `1024`
substituted (see Phase B notes in /home/adren/.claude/projects/...).

## LiteLLM compatibility note

LiteLLM's `additional_drop_params: ["response_format"]` for the gemma model
entry is required — LM Studio's OpenAI shim only accepts
`response_format.type` ∈ {"json_schema", "text"}, not "json_object". Already
configured in `/home/simcc/litellm-langfuse/litellm-config.yaml` on .148.

## Applying upstream updates

```bash
cd OB1
git fetch upstream
git checkout simcc-main
git rebase upstream/main
# Resolve conflicts if server/index.ts changes; verify the three env reads + fence strip remain.
git push --force-with-lease origin simcc-main
```

Then redeploy on .148: copy `server/index.ts` to
`/home/simcc/openbrain-supabase/supabase/docker/volumes/functions/open-brain-mcp/index.ts`
and `docker restart supabase-edge-functions`.
