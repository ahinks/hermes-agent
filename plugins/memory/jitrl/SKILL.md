# JitRL Memory Provider Plugin

## What it does

Before each LLM call, `jitrl.prefetch()` retrieves relevant past experiences AND additional context from 5 sources, then injects them as a formatted text block into the user message. This gives the agent situational awareness from memory on every single turn.

```
prefetch(query)  →  returns formatted str (not list[dict])
                     injected as <memory-context> in user message
```

Example context block that reaches the LLM on every turn:
```
[JITRL-EXPERIENCES]
  1. [sessions][success] Task: Fix 36 OpenClaw skills with duplicate YAML keys
    Approach: Use Python yaml library with duplicate-key detection
    Result: OK success
  ...

[RUNTIME-SELF-MOD] adjustments for this turn:
  [RUNTIME-SELF-MOD] type=context_patch confidence=0.81
    Trigger: api_error
    Guidance: API call failed. Implement exponential backoff with 3 retries.

[STRATEGY] [REASONING-STRATEGY] code/debug
    Why: Multi-step implementation task. Use chain-of-thought with tool verification.

[CRITIQUE] [RUNTIME-CRITIQUE] status=pending
    Verdict: Agent used wrong tool for file operations
    Suggestions:
      Use patch() for file edits instead of terminal sed/awk
      ...

[REAL-TIME-KNOWLEDGE] Live sources (refreshed ~10min):
  [hn] Laws of Software Engineering
       https://lawsofsoftwareengineering.com
  [hn] Anthropic says OpenClaw-style Claude CLI usage is allowed again
       https://docs.openclaw.ai/providers/anthropic
  ...
```

## The 4 New Capabilities

### 1. Runtime Self-Modifier
**What:** Detects errors mid-session and immediately patches the agent's context for the NEXT turn within the same session. Changes behavior THIS session, not just future ones.

**How:** `on_turn_start()` detects failure signals in the user's message. `sync_turn()` fires `runtime-self-modifier.py` on errors. The script generates a context patch written to `runtime-patches.jsonl`. `prefetch()` reads it back and formats it as advisory context.

**File:** `~/.hermes/scripts/runtime-self-modifier.py`
**Trigger:** Automatically via JitRL on errors; also via `python3 scripts/runtime-self-modifier.py --error TEXT --query TEXT --session ID`

### 2. Reasoning Strategy Selector (SELF-DISCOVER pattern)
**What:** Classifies the incoming query type and pre-loads the optimal reasoning strategy. The agent knows HOW to reason before it starts reasoning.

**How:** `reasoning-strategy-selector.py` runs every 15min via cron and also proactively on complex tasks. Decision written to `reasoning-strategy.jsonl`. `prefetch()` reads the matching strategy and formats it as `[STRATEGY]` context.

**Strategy types:** `code/debug` (chain-of-thought + tool verification), `research/analysis` (tree-of-thought + evidence synthesis), `creative` (divergent exploration + convergence), `factual/lookup` (direct retrieval + verification), `planning` (hierarchical decomposition + dependency analysis)

**File:** `~/.hermes/scripts/reasoning-strategy-selector.py`
**Trigger:** `python3 scripts/reasoning-strategy-selector.py --query "implement a new cron job"` or via cron every 15min

### 3. Real-Time Knowledge Injector
**What:** Live HN stories, arxiv cs.AI papers, and model releases refreshed every 10 minutes. The agent has access to the latest information on every turn.

**How:** `real-time-knowledge-injector.py` fetches HN top 30 stories, arxiv cs.AI papers, and HuggingFace trending models every 10 minutes via cron. Stores in SQLite FTS5 cache. Writes hot topics summary to `research-memory.json`. `prefetch()` reads it and formats as `[REAL-TIME-KNOWLEDGE]` context. Age-limited to 2 hours.

**Files:**
- `~/.hermes/scripts/real-time-knowledge-injector.py`
- `~/.hermes/workspace/realtime-knowledge.db` (SQLite FTS5, 75 records)
- `~/.hermes/workspace/research-memory.json` (hot topics summary)

**CLI:**
```bash
python3 scripts/real-time-knowledge-injector.py           # refresh all sources
python3 scripts/real-time-knowledge-injector.py --stats   # print cache stats
python3 scripts/real-time-knowledge-injector.py --query "llm inference"  # search cache
```

### 4. Runtime Multi-Agent Critique
**What:** On major failures, spawns a second LLM agent to critique the first agent's approach. The critique is written to `runtime-critique.jsonl` and picked up by `prefetch()` as `[CRITIQUE]` context for the next turn.

**How:** `multi-agent-critique.py` is triggered by `sync_turn()` on severe errors (syntax errors, complete wrong approach, repeated failures). It calls the llama.cpp inference API at localhost:8080 as the critic agent. Writes `{status: "pending", verdict, suggestions, ...}` to `runtime-critique.jsonl`. Prefetch formats it. Deduped by session.

**File:** `~/.hermes/scripts/multi-agent-critique.py`
**Trigger:** `python3 scripts/multi-agent-critique.py --failure "TypeError" --context "implementing X" --session SESSION_ID`

## Architecture

```
prefetch(query)
  → retrieve()              BM25 + KG entity expansion + UCB1
  → get_and_clear_patches()  Runtime Self-Modifier patches
  → get_current_reasoning_strategy()  SELF-DISCOVER strategy
  → get_pending_critique()   Multi-Agent Critique
  → _fmt_research_memory()   Real-Time Knowledge (HN/arxiv/models)
  → format as \n\n-joined string  ← MUST BE str, not list[dict]
  → injected as <memory-context> in user message

sync_turn(user, asst)
  → classify outcome (success/error)
  → extract tools used, skill domain
  → _append_entry()          → jitrl-experiences.jsonl
  → if error: Popen runtime-self-modifier.py (async)
  → if severe error: Popen multi-agent-critique.py (async)

on_turn_start(turn_num, message)
  → detect failure signals in message
  → write signal to runtime-patches.jsonl
  → _PATCH_APPLIED = False (resets per new user message)

on_session_end(messages)
  → extract final task and outcome
  → _append_entry()
```

## Experience store format

Each line in `jitrl-experiences.jsonl`:
```json
{
  "timestamp": "2026-04-21T15:30:00+00:00",
  "task": "Fix YAML parse error in skill frontmatter",
  "approach": "Use yaml.SafeLoader with duplicate key detection",
  "outcome": "success",
  "skill_domain": "skill_maintenance",
  "error_pattern": "yaml.scanner.ScannerError: duplicate key",
  "session_id": "abc123",
  "importance": 0.7
}
```

## Backfill sources

1. Past sessions in `~/.hermes/sessions/` (JSON logs)
2. Corrections store at `~/.hermes/workspace/corrections-store.jsonl`
3. Research findings in `~/.hermes/workspace/max-research/stream-*-*/report.json`
4. EPG entries from MemPalace (via MCP)
5. Manual entries from SI ticker corrections phase

## CLI

```bash
# Search experiences from the command line
python3 ~/.hermes/hermes-agent/plugins/memory/jitrl/__init__.py

# Real-time knowledge injector
python3 ~/.hermes/scripts/real-time-knowledge-injector.py --stats
python3 ~/.hermes/scripts/real-time-knowledge-injector.py --query "speculative decoding"

# Reasoning strategy selector
python3 ~/.hermes/scripts/reasoning-strategy-selector.py --query "implement a web scraper"

# Runtime self-modifier (diagnose mode)
python3 ~/.hermes/scripts/runtime-self-modifier.py --error "SyntaxError" --query "fix YAML" --session SESSION --diagnose

# Multi-agent critique
python3 ~/.hermes/scripts/multi-agent-critique.py --failure "wrong approach" --context "building X" --session SESSION
```

## Activation

In `~/.hermes/config.yaml`:
```yaml
memory:
  provider: jitrl
```

## Files

- `plugins/memory/jitrl/__init__.py` — main provider (MemoryProvider implementation, 23KB)
- `plugins/memory/jitrl/plugin.yaml` — plugin metadata
- `workspace/jitrl-experiences.jsonl` — experience store (created on first write)
- `workspace/runtime-patches.jsonl` — runtime self-modification patches
- `workspace/reasoning-strategy.jsonl` — reasoning strategy decisions
- `workspace/runtime-critique.jsonl` — pending multi-agent critiques
- `workspace/realtime-knowledge.db` — SQLite FTS5 live knowledge cache
- `workspace/research-memory.json` — hot topics summary
- `scripts/runtime-self-modifier.py` — runtime patch generator
- `scripts/reasoning-strategy-selector.py` — SELF-DISCOVER strategy selector
- `scripts/real-time-knowledge-injector.py` — live HN/arxiv/model monitor
- `scripts/multi-agent-critique.py` — runtime failure critique

## Critical implementation notes

**Return type is str, NOT list[dict]:** The base MemoryProvider defines `prefetch() -> str`. JitRL was originally returning `list[dict]` which caused `if result and result.strip()` to throw `AttributeError` on every turn. Fixed to return formatted str.

**BM25 is O(n) not O(n^2):** Each document is scored independently. The original implementation accumulated scores across all docs in a nested loop. Fixed.

**KG neighbor lookup uses `triples` table:** The entity_attrs table does not exist. Neighbor lookup uses `subject`/`object` columns from the `triples` table.

**Patches are cleared after reading:** `_PATCH_APPLIED` guard prevents the same patch from being injected twice in one prefetch cycle. File is atomically cleared after reading.
