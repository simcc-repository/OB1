/**
 * memory-consolidation-worker — nightly LLM-driven memory curation.
 *
 * Four jobs, all archive-first (snapshots into thoughts_archive, never hard-loses data):
 *   1. Merge near-duplicate thoughts (embedding pairs >= SIM_THRESHOLD, LLM-confirmed)
 *   2. Consolidate clusters of redundant thoughts into one summary thought
 *   3. Resolve conflicts: same subject + contradicting values -> newest truth supersedes
 *   4. Archive stale, low-value thoughts (old + unimportant + LLM-confirmed transient)
 *
 * Query params:
 *   ?dry_run=true    — judge and log proposals, write NOTHING to thoughts
 *   ?max_actions=N   — cap merge/supersede actions per run (default 5)
 *
 * Auth: x-brain-key header / ?key= / Authorization: Bearer <MCP_ACCESS_KEY>
 *
 * Every decision (including keep_both) is recorded in curation_log with run_id.
 */

import { createClient } from "@supabase/supabase-js";

// ── Environment ─────────────────────────────────────────────────────────────

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY") ?? "";
const OPENROUTER_API_KEY = Deno.env.get("OPENROUTER_API_KEY") ?? "";
const OPENROUTER_BASE = (Deno.env.get("OPENROUTER_BASE_URL") ?? "https://openrouter.ai/api/v1").replace(/\/+$/, "");
const LLM_MODEL = Deno.env.get("ENTITY_CLASSIFIER_MODEL") ?? Deno.env.get("OPENROUTER_LLM_MODEL") ?? "openai/gpt-4o-mini";
const EMBED_MODEL = Deno.env.get("OPENROUTER_EMBED_MODEL") ?? "openai/text-embedding-3-small";

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// ── Tunables ────────────────────────────────────────────────────────────────

/** Embedding cosine similarity floor for pair candidates. */
const SIM_THRESHOLD = 0.80;
/** Max similar pairs pulled per run. */
const PAIR_SCAN_LIMIT = 40;
/** Confidence floor to apply a merge/supersede verdict. */
const ACTION_CONFIDENCE = 0.8;
/** Confidence floor to archive a stale thought (extra conservative). */
const STALE_CONFIDENCE = 0.85;
/** Stale candidates: importance <= this … */
const STALE_MAX_IMPORTANCE = 2;
/** … and older than this many days. */
const STALE_MIN_AGE_DAYS = 30;
/** Max stale archives per run. */
const STALE_CAP = 5;
/** Wall-clock budget (platform kills at 150s). */
const INVOCATION_BUDGET_MS = 140_000;
const FETCH_TIMEOUT_MS = 60_000;

// ── Helpers ─────────────────────────────────────────────────────────────────

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, x-brain-key",
  "Content-Type": "application/json",
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data, null, 2), { status, headers: CORS_HEADERS });
}

function isAuthorized(req: Request): boolean {
  const url = new URL(req.url);
  const key =
    req.headers.get("x-brain-key")?.trim() ||
    url.searchParams.get("key")?.trim() ||
    (req.headers.get("authorization") ?? "").replace(/^Bearer\s+/i, "").trim();
  return key === MCP_ACCESS_KEY;
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs = FETCH_TIMEOUT_MS): Promise<Response> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error(`fetch timeout after ${timeoutMs}ms`)), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(timer);
  }
}

function stripCodeFences(text: string): string {
  const stripped = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const a = stripped.indexOf("{");
  const b = stripped.lastIndexOf("}");
  return a >= 0 && b > a ? stripped.slice(a, b + 1) : stripped;
}

// ── LiteLLM/Langfuse trace attribution ──────────────────────────────────────

const TRACE_USER = Deno.env.get("OPENBRAIN_TRACE_USER") ?? "openbrain:owner=adren";

function traceFields(runId: string, step: string, generationName: string, extra: Record<string, unknown> = {}) {
  return {
    user: TRACE_USER,
    metadata: {
      trace_id: crypto.randomUUID(),
      session_id: `openbrain:curation=${runId}`,
      trace_name: `openbrain:${step}`,
      tags: ["openbrain", `step:${step}`, "source:curation-worker"],
      trace_metadata: { run_id: runId, ...extra },
      generation_name: generationName,
    },
  };
}

// ── LLM calls ───────────────────────────────────────────────────────────────

async function llmJson(runId: string, step: string, system: string, user: string, extra: Record<string, unknown>): Promise<Record<string, unknown> | null> {
  const response = await fetchWithTimeout(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: LLM_MODEL,
      temperature: 0.1,
      response_format: { type: "json_object" },
      chat_template_kwargs: { enable_thinking: false },
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      ...traceFields(runId, step, step, extra),
    }),
  });
  if (!response.ok) throw new Error(`LLM call failed (${response.status}): ${await response.text()}`);
  const d = await response.json();
  const raw: string = d?.choices?.[0]?.message?.content ?? "";
  try {
    const parsed = JSON.parse(stripCodeFences(raw));
    return typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : null;
  } catch {
    return null;
  }
}

async function getEmbedding(runId: string, text: string): Promise<number[]> {
  const r = await fetchWithTimeout(`${OPENROUTER_BASE}/embeddings`, {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: EMBED_MODEL,
      input: text,
      ...traceFields(runId, "curation_reembed", "embed_thought"),
    }),
  });
  if (!r.ok) throw new Error(`Embeddings failed: ${r.status} ${await r.text()}`);
  return (await r.json()).data[0].embedding;
}

// ── Judge prompts ───────────────────────────────────────────────────────────

const JUDGE_SYSTEM = `You curate a personal memory store. You are given a small cluster of stored thoughts with high semantic similarity. Decide ONE action:

- "merge": the thoughts state the SAME fact(s) redundantly. Produce one consolidated text that preserves EVERY distinct detail (IDs, credentials, dates, numbers) with no loss. Differences in phrasing only.
- "supersede": the thoughts cover the SAME subject but CONTRADICT each other (e.g. an old password vs a new password, an outdated config vs its replacement). The most recent thought wins.
- "keep_both": the thoughts are about DIFFERENT subjects/instances despite similar wording (e.g. credentials for two different machines, two different people). When in doubt, choose keep_both.

CRITICAL: credentials, hostnames, serial numbers, or model numbers that differ mean DIFFERENT instances -> keep_both. Never merge distinct instances.

Return STRICT JSON only:
{"action":"merge","merged_content":"...","confidence":0.0-1.0,"reason":"..."}
{"action":"supersede","keeper_id":"<uuid of the thought that should survive>","confidence":0.0-1.0,"reason":"..."}
{"action":"keep_both","confidence":0.0-1.0,"reason":"..."}`;

const STALE_SYSTEM = `You curate a personal memory store. You are given one old, low-importance stored thought. Decide whether it is STALE — transient/obsolete content with no future retrieval value (e.g. one-off smoke tests, "testing 123" notes, temporary status updates long past, abandoned drafts).

Facts, credentials, reference data, decisions, and anything a future reader might search for are NOT stale, regardless of age.

Return STRICT JSON only:
{"archive":true|false,"confidence":0.0-1.0,"reason":"..."}`;

// ── Union-find for clustering pairs ─────────────────────────────────────────

function buildClusters(pairs: Array<{ id_a: string; id_b: string }>): string[][] {
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    if (!parent.has(x)) parent.set(x, x);
    let r = x;
    while (parent.get(r) !== r) r = parent.get(r)!;
    parent.set(x, r);
    return r;
  };
  for (const p of pairs) {
    const ra = find(p.id_a), rb = find(p.id_b);
    if (ra !== rb) parent.set(ra, rb);
  }
  const groups = new Map<string, string[]>();
  for (const k of parent.keys()) {
    const r = find(k);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r)!.push(k);
  }
  return [...groups.values()].filter((g) => g.length >= 2);
}

// ── Apply helpers (shared by merge + supersede) ─────────────────────────────

type ThoughtRow = Record<string, unknown> & { id: string };

async function archiveAndRemove(
  losers: ThoughtRow[],
  reason: string,
  archivedBy: string,
  replacedBy: string | null,
): Promise<string | null> {
  const archiveRows = losers.map((t) => ({
    ...t,
    archive_reason: reason,
    archived_by: archivedBy,
    replaced_by: replacedBy,
  }));
  const { error: archErr } = await supabase.from("thoughts_archive").insert(archiveRows);
  if (archErr) return `archive failed: ${archErr.message}`;
  const { error: delErr } = await supabase.from("thoughts").delete().in("id", losers.map((l) => l.id));
  if (delErr) return `delete failed after archive: ${delErr.message}`;
  return null;
}

function unionMetadata(keeper: ThoughtRow, losers: ThoughtRow[]): Record<string, unknown> {
  const keeperMeta = (keeper.metadata as Record<string, unknown>) || {};
  const merged: Record<string, unknown> = { ...keeperMeta };
  for (const key of ["topics", "people", "action_items"]) {
    const set = new Set<string>(Array.isArray(keeperMeta[key]) ? keeperMeta[key] as string[] : []);
    for (const l of losers) {
      const lm = (l.metadata as Record<string, unknown>) || {};
      if (Array.isArray(lm[key])) for (const v of lm[key] as string[]) set.add(v);
    }
    if (set.size) merged[key] = [...set];
  }
  merged.merged_from = [
    ...(Array.isArray(keeperMeta.merged_from) ? keeperMeta.merged_from as string[] : []),
    ...losers.map((l) => l.id),
  ];
  return merged;
}

// ── Main handler ────────────────────────────────────────────────────────────

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS_HEADERS });
  if (!MCP_ACCESS_KEY) return json({ error: "Service misconfigured: auth key not set" }, 503);
  if (!isAuthorized(req)) return json({ error: "Unauthorized" }, 401);
  if (!OPENROUTER_API_KEY) return json({ error: "No LLM API key configured" }, 503);

  const url = new URL(req.url);
  const dryRun = url.searchParams.get("dry_run") === "true";
  const maxActions = Math.min(Math.max(parseInt(url.searchParams.get("max_actions") ?? "5", 10) || 5, 1), 20);

  const runId = crypto.randomUUID();
  const startTime = Date.now();
  const summary = {
    run_id: runId,
    dry_run: dryRun,
    clusters_scanned: 0,
    merges: 0,
    supersedes: 0,
    keep_both: 0,
    stale_archived: 0,
    stale_kept: 0,
    errors: [] as string[],
    actions: [] as Record<string, unknown>[],
  };

  const log = async (action: string, keptId: string | null, removedIds: string[], reason: string, confidence: number) => {
    await supabase.from("curation_log").insert({
      run_id: runId, action, kept_id: keptId, removed_ids: removedIds,
      reason: reason.slice(0, 500), confidence, dry_run: dryRun,
    });
  };

  // ── Phase 1: similar-pair scan + clustering ──
  const { data: pairs, error: pairErr } = await supabase.rpc("find_similar_pairs", {
    p_threshold: SIM_THRESHOLD, p_limit: PAIR_SCAN_LIMIT,
  });
  if (pairErr) return json({ error: `pair scan failed: ${pairErr.message}` }, 500);

  const clusters = buildClusters((pairs || []) as Array<{ id_a: string; id_b: string }>);
  summary.clusters_scanned = clusters.length;

  let actionsTaken = 0;
  for (const cluster of clusters) {
    if (actionsTaken >= maxActions) break;
    if (Date.now() - startTime >= INVOCATION_BUDGET_MS) { summary.errors.push("wall-clock budget hit"); break; }

    const { data: rows, error: rowErr } = await supabase
      .from("thoughts").select("*").in("id", cluster);
    if (rowErr || !rows || rows.length < 2) continue;
    const thoughts = rows as ThoughtRow[];

    const clusterDesc = thoughts.map((t) =>
      `--- thought id=${t.id} created=${t.created_at} importance=${t.importance} ---\n${String(t.content).slice(0, 1500)}`
    ).join("\n\n");

    let verdict: Record<string, unknown> | null = null;
    try {
      verdict = await llmJson(runId, "curate_judge", JUDGE_SYSTEM,
        `Cluster of ${thoughts.length} similar thoughts:\n\n${clusterDesc}`,
        { cluster_ids: cluster });
    } catch (err) {
      summary.errors.push(`judge failed for cluster ${cluster[0].slice(0, 8)}: ${(err as Error).message.slice(0, 100)}`);
      continue;
    }

    const action = String(verdict?.action ?? "keep_both");
    const confidence = Number(verdict?.confidence ?? 0);
    const reason = String(verdict?.reason ?? "no verdict");

    if (action === "merge" && confidence >= ACTION_CONFIDENCE && typeof verdict?.merged_content === "string" && (verdict.merged_content as string).trim()) {
      // keeper = newest thought in cluster
      const sorted = [...thoughts].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
      const keeper = sorted[0];
      const losers = sorted.slice(1);
      const mergedContent = (verdict.merged_content as string).trim();
      summary.actions.push({ action: "merge", keeper: keeper.id, losers: losers.map((l) => l.id), confidence, reason });
      if (!dryRun) {
        try {
          const embedding = await getEmbedding(runId, mergedContent);
          const archErr = await archiveAndRemove(losers, "merged", "worker:dedup", keeper.id);
          if (archErr) { summary.errors.push(archErr); continue; }
          const { error: updErr } = await supabase.from("thoughts").update({
            content: mergedContent,
            embedding,
            metadata: unionMetadata(keeper, losers),
            importance: Math.max(...thoughts.map((t) => Number(t.importance ?? 3))),
          }).eq("id", keeper.id);
          if (updErr) { summary.errors.push(`keeper update failed: ${updErr.message}`); continue; }
        } catch (err) {
          summary.errors.push(`merge apply failed: ${(err as Error).message.slice(0, 150)}`);
          continue;
        }
      }
      await log("merge", keeper.id, losers.map((l) => l.id), reason, confidence);
      summary.merges++;
      actionsTaken++;
    } else if (action === "supersede" && confidence >= ACTION_CONFIDENCE && typeof verdict?.keeper_id === "string" && cluster.includes(verdict.keeper_id as string)) {
      const keeperId = verdict.keeper_id as string;
      const losers = thoughts.filter((t) => t.id !== keeperId);
      summary.actions.push({ action: "supersede", keeper: keeperId, losers: losers.map((l) => l.id), confidence, reason });
      if (!dryRun) {
        const archErr = await archiveAndRemove(losers, "superseded", "worker:supersede", keeperId);
        if (archErr) { summary.errors.push(archErr); continue; }
      }
      await log("supersede", keeperId, losers.map((l) => l.id), reason, confidence);
      summary.supersedes++;
      actionsTaken++;
    } else {
      summary.keep_both++;
      summary.actions.push({ action: "keep_both", ids: cluster, confidence, reason });
      await log("keep_both", null, cluster, reason, confidence);
    }
  }

  // ── Phase 2: stale-content pass ──
  const staleCutoff = new Date(Date.now() - STALE_MIN_AGE_DAYS * 86_400_000).toISOString();
  const { data: staleCandidates } = await supabase
    .from("thoughts")
    .select("*")
    .lte("importance", STALE_MAX_IMPORTANCE)
    .lt("created_at", staleCutoff)
    .or("status.is.null,status.not.in.(new,active)")
    .order("created_at", { ascending: true })
    .limit(STALE_CAP * 2);

  let staleActions = 0;
  for (const t of (staleCandidates || []) as ThoughtRow[]) {
    if (staleActions >= STALE_CAP) break;
    if (Date.now() - startTime >= INVOCATION_BUDGET_MS) { summary.errors.push("wall-clock budget hit (stale pass)"); break; }

    let verdict: Record<string, unknown> | null = null;
    try {
      verdict = await llmJson(runId, "curate_stale", STALE_SYSTEM,
        `Thought (created ${t.created_at}, importance ${t.importance}):\n\n${String(t.content).slice(0, 1500)}`,
        { thought_id: t.id });
    } catch (err) {
      summary.errors.push(`stale judge failed for ${t.id.slice(0, 8)}: ${(err as Error).message.slice(0, 100)}`);
      continue;
    }

    const archive = verdict?.archive === true;
    const confidence = Number(verdict?.confidence ?? 0);
    const reason = String(verdict?.reason ?? "no verdict");

    if (archive && confidence >= STALE_CONFIDENCE) {
      summary.actions.push({ action: "archive_stale", id: t.id, confidence, reason });
      if (!dryRun) {
        const archErr = await archiveAndRemove([t], "stale", "worker:stale", null);
        if (archErr) { summary.errors.push(archErr); continue; }
      }
      await log("archive_stale", null, [t.id], reason, confidence);
      summary.stale_archived++;
      staleActions++;
    } else {
      summary.stale_kept++;
    }
  }

  (summary as Record<string, unknown>).elapsed_ms = Date.now() - startTime;
  return json(summary);
});
