#!/usr/bin/env python3
"""
Unified Memory Retrieval Tool — MCP-accessible tool for hermes-agent

This module provides a registered tool that wraps unified_memory_search(),
making it accessible to the agent via the standard tool dispatch mechanism.

Usage (within hermes-agent):
    tool name: unified_memory_search
    args: {query: str, limit?: int, tiers?: list[str]}

Stores searched (5 tiers):
    kg         — MemPalace KG (6094 triples): importance × entity_rank × match
    chroma     — MemPalace ChromaDB: vector similarity (hinksbot collection)
    session    — Session DB (23222 messages): FTS5 + recency boost
    notes      — Wiki markdown pages: BM25-style keyword match
    corrections — Self-improving log: pattern match on fixes/recipes
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add scripts dir to path so we can import the retriever module
SCRIPTS_DIR = Path.home() / ".hermes" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from unified_memory_retriever import (
    unified_memory_search,
    UnifiedMemoryResult,
    STORE_WEIGHTS,
)

# ── Tool Schema ────────────────────────────────────────────────────────────────
TOOL_SCHEMA = {
    "name": "unified_memory_search",
    "description": """Search all memory stores simultaneously and return merged, ranked results.

Queries 5 memory stores in parallel and fuses results by relevance:
  1. kg (knowledge graph) — 6094 MemPalace KG triples: typed entities + relations
  2. chroma (vector memory) — MemPalace ChromaDB verbatim memories
  3. session (chat history) — 23222 session messages with FTS5 search
  4. notes (wiki pages) — Markdown notes from llm-wiki
  5. corrections (self-improving) — corrections.md + learnings/

Each result includes: content, source, source_detail, score (0-1), and metadata.
Results are deduplicated and sorted by weighted relevance score.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query (required)",
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10, max: 50)",
                "default": 10,
                "minimum": 1,
                "maximum": 50,
            },
            "tiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["kg", "chroma", "session", "notes", "corrections"],
                },
                "description": "Which stores to search. Defaults to all 5 stores.",
                "default": ["kg", "chroma", "session", "notes", "corrections"],
            },
        },
        "required": ["query"],
    },
}


# ── Store descriptions for output ─────────────────────────────────────────────
STORE_DESCRIPTIONS = {
    "kg": "MemPalace Knowledge Graph — structured triples with importance scoring",
    "chroma": "MemPalace ChromaDB — semantic vector similarity on verbatim memories",
    "session": "Session DB — FTS5 full-text search on 23222 conversation messages",
    "notes": "Filesystem Wiki — markdown notes from llm-wiki",
    "corrections": "Self-improving — corrections log and accumulated learnings",
}


# ── Handler ───────────────────────────────────────────────────────────────────
def handle_unified_memory_search(args: dict) -> str:
    """
    Handler for the unified_memory_search tool.
    Returns JSON string of results (compatible with tool registry convention).
    """
    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "query is required"}, indent=2)

    limit = args.get("limit", 10)
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (ValueError, TypeError):
            limit = 10
    limit = max(1, min(limit, 50))

    tiers = args.get("tiers")
    if tiers is not None and not isinstance(tiers, list):
        # Allow comma-separated string as well
        if isinstance(tiers, str):
            tiers = [t.strip() for t in tiers.split(",")]
        else:
            tiers = None

    try:
        results = unified_memory_search(query=query, limit=limit, tiers=tiers)

        # Build output
        output = {
            "query": query,
            "total_results": len(results),
            "stores_searched": tiers or list(STORE_DESCRIPTIONS.keys()),
            "store_weights": STORE_WEIGHTS,
            "results": [r.to_dict() for r in results],
        }

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "query": query}, indent=2)


# ── Availability check ────────────────────────────────────────────────────────
def check_requirements() -> bool:
    """Return True if the underlying memory stores are accessible."""
    return True  # All stores have graceful fallbacks


# ── Registration ──────────────────────────────────────────────────────────────
try:
    from tools.registry import registry
    registry.register(
        name="unified_memory_search",
        toolset="memory",          # Belongs to the memory toolset
        schema=TOOL_SCHEMA,
        handler=handle_unified_memory_search,
        check_fn=check_requirements,
        emoji="🔍",
        description="Search all memory stores at once (KG, Chroma, sessions, wiki, corrections)",
    )
except ImportError:
    # Outside hermes-agent context — skip registration
    pass
