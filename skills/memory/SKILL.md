# Unified Memory Retrieval System

## Concept & Vision

A single entry point that queries all five memory stores simultaneously, ranks results by weighted relevance, deduplicates near-identical content, and returns a unified ranked list with full source attribution. The goal is zero-friction memory access — one call, all stores, ranked intelligently.

The system prioritizes **ground truth** (KG) over **verbatim** (Chroma) over **contextual** (session), with explicit source labels on every result so the agent can reason about provenance.

---

## Architecture

```
unified_memory_search(query, limit, tiers)
         │
         ├──▶ Store 1: KG (sqlite)        ──▶ query_kg()
         ├──▶ Store 2: ChromaDB (vector)   ──▶ query_chroma()
         ├──▶ Store 3: Session DB (FTS5)   ──▶ query_session()
         ├──▶ Store 4: Wiki Notes (fs)     ──▶ query_notes()
         └──▶ Store 5: Self-improving      ──▶ query_corrections()
                        │
                   parallel (ThreadPoolExecutor, 5 workers, 8s timeout)
                        │
                   fuse_and_rank()
                        │
                   deduplicate (first-100-chars key)
                        │
                   return top-N UnifiedMemoryResult
```

---

## Memory Stores

### Store 1: MemPalace KG — `~/.mempalace/knowledge_graph.sqlite3`

**Schema:**
- `entities(id, name, type, properties, created_at)` — 470 entities
- `triples(id, subject, predicate, object, valid_from, valid_to, confidence, importance_score, mention_count, ...)` — **6094 active triples**
- `entity_attrs(entity, entity_rank, triple_count)` — PageRank-style centrality per entity
- `predicate_weights(predicate, weight)` — per-predicate importance multipliers

**Query method:** SQLite with dynamic OR-based LIKE clauses per query term. Joined against `entity_attrs` for centrality boost and `predicate_weights` for predicate importance.

**Scoring:** `importance_score × confidence × (1 + entity_rank × 0.5) × match_quality × predicate_weight`
- `importance_score`: per-triple LLM-judged salience (0-1)
- `confidence`: triple confidence (default 1.0)
- `entity_rank`: PageRank centrality (0-1, multiplicative boost up to 1.5×)
- `match_quality`: query-term overlap with `[subject predicate object]` (0.5-1.0)
- `predicate_weight`: from `predicate_weights` table (default 1.0)

**Weight:** 1.0 (highest — structured ground truth)

**Sample output:**
```
[research_topic] research_session → llama.cpp AMD GPU optimization techniques 2026
score=0.657, metadata: {importance_score: 0.65, entity_rank: 0.023, confidence: 1.0}
```

---

### Store 2: MemPalace ChromaDB — `~/.hermes/workspace/palace/chroma.sqlite3`

**Schema:**
- `collections(id, name)` — single `hinksbot` collection
- `embeddings(id, collection, vector_id, seq_id)` — vector records
- Chroma client stores: document text + metadata (wing, room, hall, date, topic, ...)

**Query method:** `chromadb.PersistentClient` → `get_collection("hinksbot")` → `query(query_texts=[query], n_results=...)`

**Scoring:** `similarity = max(0, 1.0 - distance / 2.0)` — L2 distance mapped to 0-1 similarity.

**Weight:** 0.95 (nearly full weight — verbatim semantic memory)

---

### Store 3: Session DB — `~/.hermes/state.db`

**Schema:**
- `sessions(id, source, model, started_at, ended_at, ...)` — 592 sessions
- `messages(id, session_id, role, content, timestamp, ...)` — 23222 messages
- `messages_fts` — FTS5 virtual table over `messages.content`

**Query method:** FTS5 `MATCH` query (with LIKE fallback) joining `messages` → `sessions` for recency. Deduplicated per session.

**Scoring:** `match_quality + recency_boost`, capped at 0.85
- `match_quality`: 0.4 base + up to 0.3 for keyword overlap
- `recency_boost`: +0.3 for sessions <7 days old, linearly decaying to 0

**Weight:** 0.7 (moderate — chat history is valuable but noisy)

---

### Store 4: Filesystem Notes — `~/.hermes/workspace/llm-wiki/wiki/`

**Contents:** ~50+ markdown pages organized as:
- `wiki/concepts/*.md` — curated concept notes
- `wiki/entities/*.md` — entity definitions
- `wiki/summaries/*.md` — task/project summaries
- `wiki/syntheses/*.md` — auto-generated syntheses

**Query method:** `Path.rglob("*.md")` with BM25-style keyword overlap scoring.

**Scoring:** `match_quality × path_weight`, capped at 0.7
- `path_weight`: concepts=0.9, entities=0.85, summaries=0.8, syntheses=0.75

**Weight:** 0.5 (lower — keyword-only, no semantic understanding)

---

### Store 5: Self-improving — `~/.hermes/self-improving/`

**Contents:**
- `corrections.md` — structured error/fix log with timestamps
- `domains/ERRORS.md` — categorized error patterns
- `domains/LEARNINGS.md` — accumulated domain insights
- `domains/*.md` — domain-specific learnings (model-management, cost-optimization, etc.)

**Query method:** File glob over `*.md` with regex snippet extraction.

**Scoring:** 0.4-0.85
- Exact phrase match: 0.6-0.7
- Pattern match (Fix/Cause/Symptom keywords): up to 0.85
- Term overlap: 0.4-0.7

**Weight:** 0.8 (high — corrections and learnings are high-value operational knowledge)

---

## API

```python
from unified_memory_retriever import unified_memory_search, UnifiedMemoryResult

# Basic usage
results: list[UnifiedMemoryResult] = unified_memory_search(
    query="distributed training GPU optimization",
    limit=10,
    tiers=None  # None = all 5 stores
)

# Each result:
result.content       # str — the retrieved content
result.source        # Literal["kg", "chroma", "session", "notes", "corrections"]
result.source_detail # str — predicate, room/wing, session_id, or filepath
result.score         # float — 0-1, weighted relevance
result.metadata      # dict — extra fields (timestamp, entity, etc.)
```

**CLI:**
```bash
python3 ~/.hermes/scripts/unified-memory-retriever.py "your query" --limit 10
python3 ~/.hermes/scripts/unified-memory-retriever.py --status
python3 ~/.hermes/scripts/unified-memory-retriever.py "GPU" --tiers kg,chroma --json
```

---

## Score Fusion Algorithm

1. **Parallel query**: All 5 stores queried simultaneously via `ThreadPoolExecutor` (5 workers, 8s timeout each).
2. **Weighted score**: Each store's raw score multiplied by its weight (KG=1.0, Chroma=0.95, Session=0.7, Notes=0.5, Corrections=0.8).
3. **Deduplication**: Group by first 100 chars of content, keep highest-scoring per group.
4. **Sort**: Descending by weighted score.
5. **Limit**: Return top N (default 10).

---

## MCP Integration

The tool is registered as `unified_memory_search` in the `memory` toolset via `tools/unified_memory_tool.py`.

**Tool schema:**
```json
{
  "name": "unified_memory_search",
  "description": "Search all memory stores simultaneously...",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "minLength": 1},
      "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
      "tiers": {"type": "array", "items": {"enum": ["kg", "chroma", "session", "notes", "corrections"]}}
    },
    "required": ["query"]
  }
}
```

**Usage in agent:**
```
tool: unified_memory_search
args: {query: "MCP server restart fix", limit: 10}
```

---

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Audit all 5 stores (schema, paths, query methods) | ✅ Complete |
| 2 | Write unified-memory-retriever.py with all 5 query functions | ✅ Complete |
| 3 | Add parallel execution, scoring, deduplication, fusion | ✅ Complete |
| 4 | CLI with --status, --tiers, --json flags | ✅ Complete |
| 5 | MCP tool registration (unified_memory_tool.py) | ✅ Complete |
| 6 | SKILL.md documentation | ✅ Complete |

---

## Files

| File | Purpose |
|------|---------|
| `~/.hermes/scripts/unified-memory-retriever.py` | Main module — all 5 query functions + fusion + CLI |
| `~/.hermes/hermes-agent/tools/unified_memory_tool.py` | MCP tool registration wrapper |

---

## Usage Examples

```bash
# Search everything, return top 10
python3 unified-memory-retriever.py "llama.cpp ROCm optimization"

# KG only — structured facts
python3 unified-memory-retriever.py "hacksbot AMD GPU" --tiers kg

# Sessions + Corrections — recent operational context
python3 unified-memory-retriever.py "MCP server restart" --tiers session,corrections

# Verbose JSON output for scripting
python3 unified-memory-retriever.py "distributed inference" --json --limit 5

# Check store availability
python3 unified-memory-retriever.py --status
```

---

## Design Rationale

**Why per-term OR instead of full query LIKE for KG?**
- Multi-word queries like "hinksbot language" would fail with `%hinksbot language%` (no single field contains both terms)
- Per-term OR ensures at least one term matches, then Python-level overlap scoring ranks the results

**Why deduplicate after weighting?**
- Different stores may contain the same fact expressed differently (e.g., Chroma has the verbatim, KG has the triple, session has the conversation)
- Deduplication before weighting would lose store-specific metadata; doing it after preserves the original score in metadata

**Why recency boost only for sessions?**
- Session messages are inherently time-sensitive; recent conversations are more actionable
- KG triples and Chroma documents are intentionally more permanent; recency is already implicit in their creation timestamps in metadata
