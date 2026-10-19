import "jsr:@supabase/functions-js/edge-runtime.d.ts";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPTransport } from "@hono/mcp";
import { Hono } from "hono";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const OPENROUTER_API_KEY = Deno.env.get("OPENROUTER_API_KEY")!;
const MCP_ACCESS_KEY = Deno.env.get("MCP_ACCESS_KEY")!;

const OPENROUTER_BASE = (Deno.env.get("OPENROUTER_BASE_URL") ?? "https://openrouter.ai/api/v1").replace(/\/+$/, "");
const OPENROUTER_EMBED_MODEL = Deno.env.get("OPENROUTER_EMBED_MODEL") ?? "openai/text-embedding-3-small";
const OPENROUTER_LLM_MODEL = Deno.env.get("OPENROUTER_LLM_MODEL") ?? "openai/gpt-4o-mini";
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

type ThoughtMatch = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
  created_at: string;
};

type ThoughtRecord = {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at?: string | null;
};

const CITATION_BASE_URL =
  Deno.env.get("OPEN_BRAIN_CITATION_BASE_URL") || "https://openbrain.local/thoughts";

function thoughtTitle(content: string, createdAt?: string): string {
  const firstLine = content.replace(/\s+/g, " ").trim().slice(0, 80);
  const datePrefix = createdAt ? new Date(createdAt).toLocaleDateString() : "Open Brain";
  return firstLine ? `${datePrefix} - ${firstLine}` : `${datePrefix} thought`;
}

function thoughtUrl(id: string): string {
  return `${CITATION_BASE_URL.replace(/\/$/, "")}/${id}`;
}

// --- LiteLLM/Langfuse trace attribution ---
// Adds `user` + `metadata` to every LLM request body so the LiteLLM proxy can
// group, label, and back-link traces in the `openbrain` Langfuse project.

const TRACE_USER = Deno.env.get("OPENBRAIN_TRACE_USER") ?? "openbrain:owner=adren";

type TraceCtx = {
  sessionId: string;
  traceName: string;
  step: string;
  source: string;
  traceMetadata: Record<string, unknown>;
  generationName?: string;
};

function traceFields(ctx: TraceCtx) {
  return {
    user: TRACE_USER,
    metadata: {
      trace_id: crypto.randomUUID(),
      session_id: ctx.sessionId,
      trace_name: ctx.traceName,
      tags: ["openbrain", `step:${ctx.step}`, `source:${ctx.source}`],
      trace_metadata: ctx.traceMetadata,
      ...(ctx.generationName ? { generation_name: ctx.generationName } : {}),
    },
  };
}

async function getEmbedding(text: string, ctx: TraceCtx): Promise<number[]> {
  const r = await fetch(`${OPENROUTER_BASE}/embeddings`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENROUTER_EMBED_MODEL,
      input: text,
      ...traceFields(ctx),
    }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => "");
    throw new Error(`OpenRouter embeddings failed: ${r.status} ${msg}`);
  }
  const d = await r.json();
  return d.data[0].embedding;
}

async function extractMetadata(text: string, ctx: TraceCtx): Promise<Record<string, unknown>> {
  const r = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: OPENROUTER_LLM_MODEL,
      temperature: 0.2,
      response_format: { type: "json_object" },
      // Disable thinking for Qwen3/3.6 reasoning models routed via LiteLLM/vLLM.
      // Harmless no-op on models that don't honor it (haiku, gpt-4o-mini, etc.).
      chat_template_kwargs: { enable_thinking: false },
      messages: [
        {
          role: "system",
          content: `Extract metadata from the user's captured thought. Return JSON with:
- "people": array of people mentioned (empty if none)
- "action_items": array of implied to-dos (empty if none)
- "dates_mentioned": array of dates YYYY-MM-DD (empty if none)
- "topics": array of 1-3 short topic tags (always at least one)
- "type": one of "observation", "task", "idea", "reference", "person_note"
Only extract what's explicitly there.`,
        },
        { role: "user", content: text },
      ],
      ...traceFields(ctx),
    }),
  });
  const d = await r.json();
  try {
    // Some LLMs (notably Claude/Gemma via openai-compatible proxies) wrap JSON
    // in markdown fences even when response_format: json_object is requested.
    const raw: string = d.choices[0].message.content ?? "{}";
    const stripped = raw.trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/```\s*$/i, "")
      .trim();
    const firstBrace = stripped.indexOf("{");
    const lastBrace = stripped.lastIndexOf("}");
    const jsonSlice = firstBrace >= 0 && lastBrace > firstBrace
      ? stripped.slice(firstBrace, lastBrace + 1)
      : stripped;
    return JSON.parse(jsonSlice);
  } catch {
    return { topics: ["uncategorized"], type: "observation" };
  }
}

// --- MCP Server Setup ---

const server = new McpServer({
  name: "open-brain",
  version: "1.0.0",
});

// ChatGPT compatibility: restricted connector surfaces, company knowledge, and deep
// research look for exact read-only `search` and `fetch` tool shapes.
server.registerTool(
  "search",
  {
    title: "Search Open Brain",
    description:
      "Search Open Brain memories by meaning. Use this read-only compatibility tool when ChatGPT needs search/fetch-style access to stored thoughts.",
    annotations: {
      readOnlyHint: true,
    },
    inputSchema: {
      query: z.string().describe("The search query to run against Open Brain thoughts"),
    },
  },
  async ({ query }) => {
    try {
      const qEmb = await getEmbedding(query, {
        sessionId: `openbrain:search=${crypto.randomUUID()}`,
        traceName: "openbrain:embed_query",
        step: "embed_query",
        source: "search",
        traceMetadata: { kind: "search_query" },
        generationName: "embed_query",
      });
      const { data, error } = await supabase.rpc("match_thoughts", {
        query_embedding: qEmb,
        // 0.35 (not upstream's 0.5): bge-m3 cosine scores for short queries
        // land in a lower band (~0.35-0.65) than OpenAI embeddings; at 0.5
        // relevant matches routinely score just under the cutoff and vanish.
        match_threshold: 0.35,
        match_count: 10,
        filter: {},
      });

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Search error: ${error.message}` }],
          isError: true,
        };
      }

      const results = ((data || []) as ThoughtMatch[]).map((t) => ({
        id: t.id,
        title: thoughtTitle(t.content, t.created_at),
        url: thoughtUrl(t.id),
      }));

      return {
        content: [{ type: "text" as const, text: JSON.stringify({ results }) }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

server.registerTool(
  "fetch",
  {
    title: "Fetch Open Brain Thought",
    description:
      "Fetch one Open Brain thought by ID after using search. Use this read-only compatibility tool to retrieve the full text and metadata for citation.",
    annotations: {
      readOnlyHint: true,
    },
    inputSchema: {
      id: z.string().describe("The Open Brain thought ID returned by the search tool"),
    },
  },
  async ({ id }) => {
    try {
      const { data, error } = await supabase
        .from("thoughts")
        .select("id, content, metadata, created_at, updated_at")
        .eq("id", id)
        .single();

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Fetch error: ${error.message}` }],
          isError: true,
        };
      }

      const thought = data as ThoughtRecord;
      const document = {
        id: thought.id,
        title: thoughtTitle(thought.content, thought.created_at),
        text: thought.content,
        url: thoughtUrl(thought.id),
        metadata: {
          ...thought.metadata,
          created_at: thought.created_at,
          updated_at: thought.updated_at,
        },
      };

      return {
        content: [{ type: "text" as const, text: JSON.stringify(document) }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 1: Semantic Search
server.registerTool(
  "search_thoughts",
  {
    title: "Search Thoughts",
    description:
      "Search captured thoughts by meaning. Use this when the user asks about a topic, person, or idea they've previously captured.",
    annotations: {
      readOnlyHint: true,
    },
    inputSchema: {
      query: z.string().describe("What to search for"),
      limit: z.number().optional().default(10),
      // 0.35 (not upstream's 0.5): bge-m3 scores land in a lower band than
      // OpenAI embeddings; 0.5 filtered out genuinely relevant matches.
      threshold: z.number().optional().default(0.35),
      recency_weight: z.number().min(0).max(1).optional().default(0)
        .describe("0 = pure similarity (default). >0 blends in recency: score = sim*(1-w) + exp(-age/half_life)*w"),
      half_life_days: z.number().positive().optional().default(90)
        .describe("Recency decay half-life in days (only used when recency_weight > 0)"),
    },
  },
  async ({ query, limit, threshold, recency_weight, half_life_days }) => {
    try {
      const qEmb = await getEmbedding(query, {
        sessionId: `openbrain:search=${crypto.randomUUID()}`,
        traceName: "openbrain:embed_query",
        step: "embed_query",
        source: "search",
        traceMetadata: { kind: "search_query" },
        generationName: "embed_query",
      });
      // recency_weight > 0 opts into the recency-boosted RPC (installed
      // 2026-06-12, 1024-dim bge-m3 variant). Default 0 keeps behavior
      // byte-identical to plain match_thoughts.
      const { data, error } = recency_weight > 0
        ? await supabase.rpc("match_thoughts_recency", {
            query_embedding: qEmb,
            match_threshold: threshold,
            match_count: limit,
            filter: {},
            recency_weight,
            half_life_days,
          })
        : await supabase.rpc("match_thoughts", {
            query_embedding: qEmb,
            match_threshold: threshold,
            match_count: limit,
            filter: {},
          });

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Search error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || data.length === 0) {
        return {
          content: [{ type: "text" as const, text: `No thoughts found matching "${query}".` }],
        };
      }

      const results = data.map(
        (
          t: ThoughtMatch,
          i: number
        ) => {
          const m = t.metadata || {};
          const parts = [
            `--- Result ${i + 1} (${(t.similarity * 100).toFixed(1)}% match) ---`,
            `Captured: ${new Date(t.created_at).toLocaleDateString()}`,
            `Type: ${m.type || "unknown"}`,
          ];
          if (Array.isArray(m.topics) && m.topics.length)
            parts.push(`Topics: ${(m.topics as string[]).join(", ")}`);
          if (Array.isArray(m.people) && m.people.length)
            parts.push(`People: ${(m.people as string[]).join(", ")}`);
          if (Array.isArray(m.action_items) && m.action_items.length)
            parts.push(`Actions: ${(m.action_items as string[]).join("; ")}`);
          parts.push(`\n${t.content}`);
          return parts.join("\n");
        }
      );

      return {
        content: [
          {
            type: "text" as const,
            text: `Found ${data.length} thought(s):\n\n${results.join("\n\n")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 2: List Recent
server.registerTool(
  "list_thoughts",
  {
    title: "List Recent Thoughts",
    description:
      "List recently captured thoughts with optional filters by type, topic, person, or time range.",
    annotations: {
      readOnlyHint: true,
    },
    inputSchema: {
      limit: z.number().optional().default(10),
      type: z.string().optional().describe("Filter by type: observation, task, idea, reference, person_note"),
      topic: z.string().optional().describe("Filter by topic tag"),
      person: z.string().optional().describe("Filter by person mentioned"),
      days: z.number().optional().describe("Only thoughts from the last N days"),
    },
  },
  async ({ limit, type, topic, person, days }) => {
    try {
      let q = supabase
        .from("thoughts")
        .select("content, metadata, created_at")
        .order("created_at", { ascending: false })
        .limit(limit);

      if (type) q = q.contains("metadata", { type });
      if (topic) q = q.contains("metadata", { topics: [topic] });
      if (person) q = q.contains("metadata", { people: [person] });
      if (days) {
        const since = new Date();
        since.setDate(since.getDate() - days);
        q = q.gte("created_at", since.toISOString());
      }

      const { data, error } = await q;

      if (error) {
        return {
          content: [{ type: "text" as const, text: `Error: ${error.message}` }],
          isError: true,
        };
      }

      if (!data || !data.length) {
        return { content: [{ type: "text" as const, text: "No thoughts found." }] };
      }

      const results = data.map(
        (
          t: { content: string; metadata: Record<string, unknown>; created_at: string },
          i: number
        ) => {
          const m = t.metadata || {};
          const tags = Array.isArray(m.topics) ? (m.topics as string[]).join(", ") : "";
          return `${i + 1}. [${new Date(t.created_at).toLocaleDateString()}] (${m.type || "??"}${tags ? " - " + tags : ""})\n   ${t.content}`;
        }
      );

      return {
        content: [
          {
            type: "text" as const,
            text: `${data.length} recent thought(s):\n\n${results.join("\n\n")}`,
          },
        ],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 3: Stats
server.registerTool(
  "thought_stats",
  {
    title: "Thought Statistics",
    description: "Get a summary of all captured thoughts: totals, types, top topics, and people.",
    annotations: {
      readOnlyHint: true,
    },
    inputSchema: {},
  },
  async () => {
    try {
      const { count } = await supabase
        .from("thoughts")
        .select("*", { count: "exact", head: true });

      const { data } = await supabase
        .from("thoughts")
        .select("metadata, created_at")
        .order("created_at", { ascending: false });

      const types: Record<string, number> = {};
      const topics: Record<string, number> = {};
      const people: Record<string, number> = {};

      for (const r of data || []) {
        const m = (r.metadata || {}) as Record<string, unknown>;
        if (m.type) types[m.type as string] = (types[m.type as string] || 0) + 1;
        if (Array.isArray(m.topics))
          for (const t of m.topics) topics[t as string] = (topics[t as string] || 0) + 1;
        if (Array.isArray(m.people))
          for (const p of m.people) people[p as string] = (people[p as string] || 0) + 1;
      }

      const sort = (o: Record<string, number>): [string, number][] =>
        Object.entries(o)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10);

      const lines: string[] = [
        `Total thoughts: ${count}`,
        `Date range: ${
          data?.length
            ? new Date(data[data.length - 1].created_at).toLocaleDateString() +
              " → " +
              new Date(data[0].created_at).toLocaleDateString()
            : "N/A"
        }`,
        "",
        "Types:",
        ...sort(types).map(([k, v]) => `  ${k}: ${v}`),
      ];

      if (Object.keys(topics).length) {
        lines.push("", "Top topics:");
        for (const [k, v] of sort(topics)) lines.push(`  ${k}: ${v}`);
      }

      if (Object.keys(people).length) {
        lines.push("", "People mentioned:");
        for (const [k, v] of sort(people)) lines.push(`  ${k}: ${v}`);
      }

      return { content: [{ type: "text" as const, text: lines.join("\n") }] };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 4: Capture Thought
server.registerTool(
  "capture_thought",
  {
    title: "Capture Thought",
    description:
      "Save a new thought to the Open Brain. Generates an embedding and extracts metadata automatically. Use this when the user wants to save something to their brain directly from any AI client — notes, insights, decisions, or migrated content from other systems.",
    annotations: {
      readOnlyHint: false,
      openWorldHint: false,
      destructiveHint: false,
      idempotentHint: false,
    },
    inputSchema: {
      content: z.string().describe("The thought to capture — a clear, standalone statement that will make sense when retrieved later by any AI"),
    },
  },
  async ({ content }) => {
    try {
      const captureId = crypto.randomUUID();
      const sessionId = `openbrain:thought=${captureId}`;
      const [embedding, metadata] = await Promise.all([
        getEmbedding(content, {
          sessionId,
          traceName: "openbrain:embed_thought",
          step: "embed",
          source: "mcp",
          traceMetadata: { capture_id: captureId, source: "mcp" },
          generationName: "embed_thought",
        }),
        extractMetadata(content, {
          sessionId,
          traceName: "openbrain:extract_metadata",
          step: "extract_metadata",
          source: "mcp",
          traceMetadata: { capture_id: captureId, source: "mcp" },
          generationName: "extract_metadata",
        }),
      ]);

      const { data: upsertResult, error: upsertError } = await supabase.rpc("upsert_thought", {
        p_content: content,
        p_payload: { metadata: { ...metadata, source: "mcp", trace_session: captureId } },
      });

      if (upsertError) {
        return {
          content: [{ type: "text" as const, text: `Failed to capture: ${upsertError.message}` }],
          isError: true,
        };
      }

      const thoughtId = upsertResult?.id;
      const { error: embError } = await supabase
        .from("thoughts")
        .update({ embedding })
        .eq("id", thoughtId);

      if (embError) {
        return {
          content: [{ type: "text" as const, text: `Failed to save embedding: ${embError.message}` }],
          isError: true,
        };
      }

      const meta = metadata as Record<string, unknown>;
      let confirmation = `Captured as ${meta.type || "thought"}`;
      if (Array.isArray(meta.topics) && meta.topics.length)
        confirmation += ` — ${(meta.topics as string[]).join(", ")}`;
      if (Array.isArray(meta.people) && meta.people.length)
        confirmation += ` | People: ${(meta.people as string[]).join(", ")}`;
      if (Array.isArray(meta.action_items) && meta.action_items.length)
        confirmation += ` | Actions: ${(meta.action_items as string[]).join("; ")}`;

      return {
        content: [{ type: "text" as const, text: confirmation }],
      };
    } catch (err: unknown) {
      return {
        content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }],
        isError: true,
      };
    }
  }
);

// Tool 5: Update Thought
server.registerTool(
  "update_thought",
  {
    title: "Update Thought",
    description:
      "Update a stored thought by ID. Provide one or more of: content, type, importance, metadata. Re-embeds automatically if content changes.",
    annotations: {
      readOnlyHint: false,
      openWorldHint: false,
      destructiveHint: true,
      idempotentHint: true,
    },
    inputSchema: {
      id: z.string().uuid().describe("UUID of the thought to update"),
      content: z.string().optional().describe("New content (triggers re-embedding)"),
      type: z.string().optional().describe("Thought type: observation, task, idea, reference, person_note"),
      importance: z.number().int().min(1).max(5).optional().describe("Importance 1-5"),
      metadata: z.record(z.string(), z.unknown()).optional().describe("Metadata fields to merge over existing"),
    },
  },
  async ({ id, content, type, importance, metadata }) => {
    try {
      const { data: existing, error: fetchErr } = await supabase
        .from("thoughts")
        .select("id, content, metadata")
        .eq("id", id)
        .single();
      if (fetchErr || !existing) {
        return { content: [{ type: "text" as const, text: `Thought ${id} not found` }], isError: true };
      }

      const mergedMeta = {
        ...(existing.metadata as Record<string, unknown> || {}),
        ...(metadata || {}),
      };
      if (type) mergedMeta.type = type;

      const update: Record<string, unknown> = { metadata: mergedMeta };
      const changed: string[] = ["metadata"];
      if (content !== undefined) {
        update.content = content;
        update.embedding = await getEmbedding(content, {
          sessionId: `openbrain:thought=${id}`,
          traceName: "openbrain:embed_thought",
          step: "embed",
          source: "mcp",
          traceMetadata: { thought_id: id, source: "mcp" },
          generationName: "embed_thought",
        });
        changed.push("content (re-embedded)");
      }
      if (type !== undefined) { update.type = type; changed.push(`type=${type}`); }
      if (importance !== undefined) { update.importance = importance; changed.push(`importance=${importance}`); }

      const { error: updErr } = await supabase.from("thoughts").update(update).eq("id", id);
      if (updErr) {
        return { content: [{ type: "text" as const, text: `Update failed: ${updErr.message}` }], isError: true };
      }
      return { content: [{ type: "text" as const, text: `Updated ${id}: ${changed.join(", ")}` }] };
    } catch (err: unknown) {
      return { content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }], isError: true };
    }
  }
);

// Tool 6: Delete Thought
server.registerTool(
  "delete_thought",
  {
    title: "Delete Thought",
    description:
      "Delete a thought by ID. Cascades to thought_entities and entity_extraction_queue. Irreversible.",
    annotations: {
      readOnlyHint: false,
      openWorldHint: false,
      destructiveHint: true,
      idempotentHint: true,
    },
    inputSchema: {
      id: z.string().uuid().describe("UUID of the thought to delete"),
    },
  },
  async ({ id }) => {
    try {
      const { data: existing } = await supabase
        .from("thoughts")
        .select("id, content")
        .eq("id", id)
        .single();
      if (!existing) {
        return { content: [{ type: "text" as const, text: `Thought ${id} not found` }], isError: true };
      }

      const { error: delErr } = await supabase.from("thoughts").delete().eq("id", id);
      if (delErr) {
        return { content: [{ type: "text" as const, text: `Delete failed: ${delErr.message}` }], isError: true };
      }
      const c = String(existing.content);
      const preview = c.slice(0, 60) + (c.length > 60 ? "..." : "");
      return { content: [{ type: "text" as const, text: `Deleted ${id} ("${preview}")` }] };
    } catch (err: unknown) {
      return { content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }], isError: true };
    }
  }
);

// Tool 7: Merge Thoughts
server.registerTool(
  "merge_thoughts",
  {
    title: "Merge Thoughts",
    description:
      "Merge near-duplicate thoughts into one. Keeps `keep_id`, archives the others into thoughts_archive (recoverable), unions their topics/people/action_items metadata onto the keeper, and optionally replaces the keeper's content (triggers re-embedding and entity re-extraction). Use after spotting duplicates via search_thoughts or the dashboard duplicates view.",
    annotations: {
      readOnlyHint: false,
      openWorldHint: false,
      destructiveHint: true,
      idempotentHint: false,
    },
    inputSchema: {
      keep_id: z.string().uuid().describe("UUID of the thought to keep"),
      merge_ids: z.array(z.string().uuid()).min(1).max(10).describe("UUIDs of thoughts to merge into the keeper (will be archived)"),
      merged_content: z.string().optional().describe("Optional consolidated content for the keeper (re-embeds if provided)"),
    },
  },
  async ({ keep_id, merge_ids, merged_content }) => {
    try {
      if (merge_ids.includes(keep_id)) {
        return { content: [{ type: "text" as const, text: "keep_id must not appear in merge_ids" }], isError: true };
      }

      const allIds = [keep_id, ...merge_ids];
      const { data: rows, error: fetchErr } = await supabase
        .from("thoughts")
        .select("*")
        .in("id", allIds);
      if (fetchErr) {
        return { content: [{ type: "text" as const, text: `Fetch failed: ${fetchErr.message}` }], isError: true };
      }
      const byId = new Map((rows || []).map((r: Record<string, unknown>) => [r.id as string, r]));
      const missing = allIds.filter((id) => !byId.has(id));
      if (missing.length) {
        return { content: [{ type: "text" as const, text: `Thought(s) not found: ${missing.join(", ")}` }], isError: true };
      }

      const keeper = byId.get(keep_id) as Record<string, unknown>;
      const losers = merge_ids.map((id) => byId.get(id) as Record<string, unknown>);

      // 1. Snapshot losers into the archive (recoverable)
      const archiveRows = losers.map((t) => ({
        ...t,
        archive_reason: "merged",
        archived_by: "mcp:merge_thoughts",
        replaced_by: keep_id,
      }));
      const { error: archErr } = await supabase.from("thoughts_archive").insert(archiveRows);
      if (archErr) {
        return { content: [{ type: "text" as const, text: `Archive failed (nothing merged): ${archErr.message}` }], isError: true };
      }

      // 2. Union list-valued metadata onto the keeper
      const keeperMeta = (keeper.metadata as Record<string, unknown>) || {};
      const mergedMeta: Record<string, unknown> = { ...keeperMeta };
      for (const key of ["topics", "people", "action_items"]) {
        const merged = new Set<string>(
          Array.isArray(keeperMeta[key]) ? (keeperMeta[key] as string[]) : [],
        );
        for (const l of losers) {
          const lm = (l.metadata as Record<string, unknown>) || {};
          if (Array.isArray(lm[key])) for (const v of lm[key] as string[]) merged.add(v);
        }
        if (merged.size) mergedMeta[key] = [...merged];
      }
      mergedMeta.merged_from = [
        ...(Array.isArray(keeperMeta.merged_from) ? (keeperMeta.merged_from as string[]) : []),
        ...merge_ids,
      ];

      const maxImportance = Math.max(
        Number(keeper.importance ?? 3),
        ...losers.map((l) => Number(l.importance ?? 3)),
      );

      const update: Record<string, unknown> = { metadata: mergedMeta, importance: maxImportance };
      if (merged_content !== undefined && merged_content.trim()) {
        update.content = merged_content.trim();
        update.embedding = await getEmbedding(merged_content.trim(), {
          sessionId: `openbrain:thought=${keep_id}`,
          traceName: "openbrain:embed_thought",
          step: "embed",
          source: "mcp",
          traceMetadata: { thought_id: keep_id, source: "mcp", kind: "merge_reembed" },
          generationName: "embed_thought",
        });
      }
      const { error: updErr } = await supabase.from("thoughts").update(update).eq("id", keep_id);
      if (updErr) {
        return { content: [{ type: "text" as const, text: `Keeper update failed: ${updErr.message}` }], isError: true };
      }

      // 3. Delete losers (graph links cascade; keeper re-extracts on next hourly run if content changed)
      const { error: delErr } = await supabase.from("thoughts").delete().in("id", merge_ids);
      if (delErr) {
        return { content: [{ type: "text" as const, text: `Delete failed after archive — losers still live. ${delErr.message}` }], isError: true };
      }

      // 4. Audit log
      await supabase.from("curation_log").insert({
        run_id: crypto.randomUUID(),
        action: "merge",
        kept_id: keep_id,
        removed_ids: merge_ids,
        reason: "manual merge via MCP",
        confidence: 1.0,
        dry_run: false,
      });

      return {
        content: [{
          type: "text" as const,
          text: `Merged ${merge_ids.length} thought(s) into ${keep_id}` +
            (merged_content ? " (content replaced, re-embedded)" : "") +
            `. Archived originals are recoverable from thoughts_archive.`,
        }],
      };
    } catch (err: unknown) {
      return { content: [{ type: "text" as const, text: `Error: ${(err as Error).message}` }], isError: true };
    }
  }
);

// --- Hono App with Auth + CORS ---

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-brain-key, accept, mcp-session-id, mcp-protocol-version, last-event-id",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS, DELETE",
};

// JSON-RPC error code for unauthorized requests.
// Per the JSON-RPC 2.0 spec, the range -32099 to -32000 is reserved for
// implementation-defined server errors. -32001 is the conventional
// "Unauthorized" code used by MCP clients/servers in the wild.
//
// Why a JSON-RPC envelope (HTTP 200) instead of a bare HTTP 401?
// Strict MCP hosts (Codex CLI, Claude Code) treat bare HTTP 4xx responses
// as transport-level failures and tear the connection down rather than
// surfacing the failure to the application layer. Wrapping the auth
// rejection in a JSON-RPC error keeps the connection alive and lets
// clients recover (e.g. prompt the user for a new key, refetch a stale
// cache) instead of dying.
const JSON_RPC_UNAUTHORIZED_CODE = -32001;
const UNAUTHORIZED_MESSAGE = "Unauthorized: missing or invalid authentication.";

/**
 * Read the request body as text without consuming the original request's
 * body stream for downstream handlers. Returns null on bodyless methods
 * or read failure. Safe here because it is only called on the auth-reject
 * path, which returns immediately and never reaches the transport.
 */
async function readBodyText(req: Request): Promise<string | null> {
  if (req.method === "GET" || req.method === "HEAD" || req.method === "DELETE") {
    return null;
  }
  try {
    return await req.text();
  } catch {
    return null;
  }
}

/**
 * Best-effort extraction of the JSON-RPC `id` from a raw request body.
 * Returns null when the body is missing, not JSON, or not a JSON-RPC
 * shape with an id. Per the JSON-RPC 2.0 spec, id may be a string,
 * number, or null — we preserve any of those; anything else becomes null.
 */
function extractJsonRpcId(bodyText: string | null): string | number | null {
  if (!bodyText) return null;
  try {
    const parsed = JSON.parse(bodyText);
    if (parsed && typeof parsed === "object" && "id" in parsed) {
      const id = (parsed as { id: unknown }).id;
      if (typeof id === "string" || typeof id === "number" || id === null) {
        return id;
      }
    }
  } catch {
    // fall through — malformed body
  }
  return null;
}

/**
 * Build a JSON-RPC 2.0 error envelope response for auth failures.
 * Returns HTTP 200 — the JSON-RPC layer expresses the error so that
 * strict MCP clients keep the connection alive instead of treating
 * the failure as a transport-level fault.
 */
function unauthorizedResponse(id: string | number | null): Response {
  const body = {
    jsonrpc: "2.0",
    error: {
      code: JSON_RPC_UNAUTHORIZED_CODE,
      message: UNAUTHORIZED_MESSAGE,
    },
    id,
  };
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders,
    },
  });
}

const app = new Hono();

// CORS preflight — required for browser/Electron-based clients (Claude Desktop, claude.ai)
app.options("*", (c) => {
  return c.text("ok", 200, corsHeaders);
});

app.all("*", async (c) => {
  // Accept access key via header OR URL query parameter
  const provided = c.req.header("x-brain-key") || new URL(c.req.url).searchParams.get("key");
  if (!provided || provided !== MCP_ACCESS_KEY) {
    // Return a JSON-RPC 2.0 error envelope (HTTP 200) instead of a bare
    // HTTP 401 so strict MCP hosts treat this as an application-level
    // error rather than a transport fault and keep the connection alive.
    // Best-effort echo of the inbound request id keeps the response
    // correlated; malformed/missing bodies fall back to id: null.
    const bodyText = await readBodyText(c.req.raw);
    const id = extractJsonRpcId(bodyText);
    return unauthorizedResponse(id);
  }

  // Fix: Claude Desktop connectors don't send the Accept header that
  // StreamableHTTPTransport requires. Build a patched request if missing.
  // See: https://github.com/NateBJones-Projects/OB1/issues/33
  if (!c.req.header("accept")?.includes("text/event-stream")) {
    const headers = new Headers(c.req.raw.headers);
    headers.set("Accept", "application/json, text/event-stream");
    const patched = new Request(c.req.raw.url, {
      method: c.req.raw.method,
      headers,
      body: c.req.raw.body,
      // @ts-ignore -- duplex required for streaming body in Deno
      duplex: "half",
    });
    Object.defineProperty(c.req, "raw", { value: patched, writable: true });
  }

  const transport = new StreamableHTTPTransport();
  await server.connect(transport);
  return transport.handleRequest(c);
});

Deno.serve(app.fetch);
