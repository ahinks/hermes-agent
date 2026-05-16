"""
JitRL memory plugin -- Just-in-Time experience recall for HinksBot.

JitRL = "Just-in-Time Reinforcement Learning". Before every LLM call, this
provider retrieves the most relevant past experiences from the local JSONL
experience store and injects them into the conversation context.

Runtime Self-Modification: on_turn_start() tracks in-turn errors and writes
patches to runtime-patches.jsonl. prefetch() reads them back and formats them
as advisory context for the next LLM call -- changing this session's behavior
immediately, not just future sessions.

Files
  ~/.hermes/workspace/jitrl-experiences.jsonl  -- experience store
  ~/.hermes/workspace/runtime-patches.jsonl     -- runtime self-modification patches
  ~/.hermes/workspace/reasoning-strategy.jsonl -- per-query reasoning strategy
  ~/.hermes/workspace/runtime-critique.jsonl    -- pending runtime critique
  ~/.mempalace/knowledge_graph.sqlite3         -- KG for entity-aware retrieval
  ~/.hermes/scripts/runtime-self-modifier.py    -- generates runtime context patches
  ~/.hermes/scripts/reasoning-strategy-selector.py -- selects reasoning approach
  ~/.hermes/scripts/real-time-knowledge-injector.py -- live HN/arxiv/model monitoring
  ~/.hermes/scripts/multi-agent-critique.py     -- runtime failure critique
  ~/.hermes/scripts/si-outcome-tracker.py       -- measures SI loop effectiveness
  ~/.hermes/scripts/session-error-aggregator.py -- cross-session error patterns
  ~/.hermes/scripts/belief-revision.py          -- detects belief contradictions
  ~/.hermes/scripts/model-release-monitor.py    -- AI lab changelog monitoring
"""
from __future__ import annotations

import hashlib, json, math, os, re, sqlite3, struct, subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Coordinator CLI path for MemRL outcome logging
_COORDINATOR_CLI = Path.home() / ".hermes" / "scripts" / "continuous-learning-coordinator.py"

# --------------------------------------------------------------------------
# Runtime Self-Modifier -- module-level state shared by on_turn_start,
# sync_turn, and prefetch within a single agent session.
# --------------------------------------------------------------------------
_RUNTIME_PATCH_FILE   = Path.home() / ".hermes" / "workspace" / "runtime-patches.jsonl"
_REASONING_STRATEGY_FILE = Path.home() / ".hermes" / "workspace" / "reasoning-strategy.jsonl"
_SI_OUTCOMES_FILE = Path.home() / ".hermes" / "workspace" / "si-outcomes" / "si-outcomes-latest.json"
_BELIEF_REVISIONS_FILE = Path.home() / ".hermes" / "workspace" / "belief-revisions.json"
_ERROR_AGGREGATOR_FILE = Path.home() / ".hermes" / "workspace" / "session-error-aggregator.json"
_MODEL_RELEASES_FILE = Path.home() / ".hermes" / "workspace" / "model-releases-detected.json"
_CRITIQUE_FILE        = Path.home() / ".hermes" / "workspace" / "runtime-critique.jsonl"

_RUNTIME_ERRORS: list[dict] = []        # errors accumulated this turn
_PATCH_APPLIED: bool = False           # guard against double-inject per prefetch cycle


def track_error(error_info: dict) -> None:
    """Called from on_turn_start() when a tool call fails mid-turn."""
    global _RUNTIME_ERRORS
    _RUNTIME_ERRORS.append(error_info)


def get_and_clear_patches() -> list[dict]:
    """Pop pending runtime patches from disk (called once per prefetch)."""
    global _RUNTIME_ERRORS, _PATCH_APPLIED
    if _PATCH_APPLIED:
        return []
    _PATCH_APPLIED = True
    _RUNTIME_ERRORS = []
    if not _RUNTIME_PATCH_FILE.exists():
        return []
    patches = []
    try:
        text = _RUNTIME_PATCH_FILE.read_text(errors="replace")
        _RUNTIME_PATCH_FILE.write_text("")   # atomic clear
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                patches.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return patches


def get_current_reasoning_strategy(query: str) -> dict:
    """Read the last reasoning strategy decision matching this query."""
    if not _REASONING_STRATEGY_FILE.exists():
        return {}
    q_lower = query.lower()
    try:
        for line in reversed(_REASONING_STRATEGY_FILE.read_text(errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                eq = entry.get("query", "").lower()
                if eq and (eq in q_lower or q_lower in eq):
                    return entry
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return {}


def get_pending_critique() -> dict | None:
    """Return the most recent pending critique, if any."""
    if not _CRITIQUE_FILE.exists():
        return None
    try:
        for line in reversed(_CRITIQUE_FILE.read_text(errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") == "pending":
                    return entry
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_HERE       = Path(__file__).parent
_STORE_PATH = Path.home() / ".hermes" / "workspace" / "jitrl-experiences.jsonl"
_KG_PATH    = Path.home() / ".mempalace" / "knowledge_graph.sqlite3"
_EXPERIENCES: list[dict] = []
_CACHE_LOADED = False
_DOC_FREQS: dict[str, int] = {}
_AVGDL: float = 1.0
_N_DOCS: int = 0
_DIRTY: bool = False   # set True when new entries are appended

# Retrieval weights — loaded from jitrl_weights.json on init, updated by coordinator
_WEIGHTS_FILE = Path.home() / ".hermes" / "scripts" / "jitrl_weights.json"
_DEFAULT_WEIGHTS = {"bm25": 0.7, "kg": 0.2, "ucb1": 0.1}
_RETRIEVAL_WEIGHTS: dict[str, float] = _DEFAULT_WEIGHTS.copy()

# MemRL outcome tracking — which experiences were retrieved this session
_RETRIEVED_EXPERIENCES: list[dict] = []   # [{exp_text, exp_id, retrieved_at}]
_COORDINATOR_AVAILABLE: bool | None = None


def _coordinator_available() -> bool:
    """Check if the coordinator CLI is reachable (cached)."""
    global _COORDINATOR_AVAILABLE
    if _COORDINATOR_AVAILABLE is None:
        try:
            result = subprocess.run(
                ["python3", str(_COORDINATOR_CLI), "stats", "--days", "0"],
                capture_output=True, timeout=10,
                cwd=str(Path.home()),
            )
            _COORDINATOR_AVAILABLE = result.returncode == 0
        except Exception:
            _COORDINATOR_AVAILABLE = False
    return _COORDINATOR_AVAILABLE


def _log_retrieval_to_coordinator(experiences: list[dict], query: str) -> None:
    """Log which experiences were retrieved so the coordinator can track outcome later."""
    if not experiences or not _coordinator_available():
        return
    for exp in experiences:
        exp_text = exp.get("task", "") + " " + exp.get("approach", "")
        exp_id = hashlib.md5(exp_text[:200].encode()).hexdigest()[:12]
        _RETRIEVED_EXPERIENCES.append({
            "exp_id": exp_id,
            "exp_text": exp_text[:300],
            "query": query[:200],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "outcome": None,   # filled in by _log_outcome_to_coordinator
        })


def _log_outcome_to_coordinator(outcome: str, session_id: str = "") -> None:
    """Log outcome for all experiences retrieved this session. Fires the MemRL feedback loop."""
    global _RETRIEVED_EXPERIENCES
    if not _RETRIEVED_EXPERIENCES or not _coordinator_available():
        _RETRIEVED_EXPERIENCES = []
        return
    for ret in _RETRIEVED_EXPERIENCES:
        if ret.get("outcome") is not None:
            continue   # already logged
        ret["outcome"] = outcome
        ret["session_id"] = session_id
        try:
            subprocess.run(
                [
                    "python3", str(_COORDINATOR_CLI), "log-outcome",
                    "-k", ret["exp_text"],
                    "-q", ret["query"],
                    "-o", outcome,
                    "-s", session_id or ret.get("session_id", ""),
                ],
                capture_output=True, timeout=10,
                cwd=str(Path.home()),
            )
        except Exception:
            pass
    _RETRIEVED_EXPERIENCES = []
    return

def _load_weights() -> dict[str, float]:
    """Load retrieval weights from jitrl_weights.json. Returns defaults if absent."""
    if not _WEIGHTS_FILE.exists():
        return _DEFAULT_WEIGHTS.copy()
    try:
        data = json.loads(_WEIGHTS_FILE.read_text())
        w = {
            "bm25": float(data.get("bm25", _DEFAULT_WEIGHTS["bm25"])),
            "kg":   float(data.get("kg",   _DEFAULT_WEIGHTS["kg"])),
            "ucb1": float(data.get("ucb1", _DEFAULT_WEIGHTS["ucb1"])),
        }
        # Normalise so they sum to 1.0
        total = w["bm25"] + w["kg"] + w["ucb1"]
        if total > 0:
            w["bm25"] /= total
            w["kg"]   /= total
            w["ucb1"] /= total
        return w
    except Exception:
        return _DEFAULT_WEIGHTS.copy()


# --------------------------------------------------------------------------
# BM25 (lightweight, no external deps)
# --------------------------------------------------------------------------
def _bm25_score(
    q_tokens: list[str], doc_tokens: list[str], doc_len: int,
    avgdl: float, n_docs: int, doc_freqs: dict[str, int],
) -> float:
    """BM25 score for ONE document."""
    if not doc_tokens:
        return 0.0
    k1, b = 1.5, 0.75
    score = 0.0
    for token in q_tokens:
        df = doc_freqs.get(token, 0)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        tf = doc_tokens.count(token)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / max(1, avgdl)))
        score += idf * tf_norm
    return score


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w{2,}\b', text.lower())


def _load_store() -> list[dict]:
    """Load + tokenize experience store. Cached after first read."""
    global _EXPERIENCES, _CACHE_LOADED, _DOC_FREQS, _AVGDL, _N_DOCS, _DIRTY
    if _CACHE_LOADED and not _DIRTY:
        return _EXPERIENCES

    _EXPERIENCES = []
    if not _STORE_PATH.exists():
        _CACHE_LOADED = True
        return _EXPERIENCES

    # Pass 1: load
    with open(_STORE_PATH, "r", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                entry = json.loads(raw)
                entry.setdefault("task", "")
                entry.setdefault("approach", "")
                entry.setdefault("outcome", "")
                entry.setdefault("skill_domain", "")
                entry.setdefault("error_pattern", "")
                entry.setdefault("timestamp", "")
                entry.setdefault("session_id", "")
                entry.setdefault("importance", 1.0)
                text = entry.get("task", "") + " " + entry.get("approach", "")
                entry["_tokens"] = _tokenize(text)
                _EXPERIENCES.append(entry)
            except json.JSONDecodeError:
                pass

    # Pass 2: BM25 globals
    _N_DOCS = len(_EXPERIENCES)
    _DOC_FREQS = {}
    total_len = 0
    for exp in _EXPERIENCES:
        tokens = exp.get("_tokens", [])
        total_len += len(tokens)
        seen = set()
        for t in tokens:
            if t not in seen:
                _DOC_FREQS[t] = _DOC_FREQS.get(t, 0) + 1
                seen.add(t)
    _AVGDL = total_len / max(1, _N_DOCS)

    _CACHE_LOADED = True
    _DIRTY = False
    return _EXPERIENCES


def _kg_neighbors(entity: str, radius: int = 2) -> list[str]:
    """Return entity surface-forms reachable within *radius* hops via triples table."""
    if not _KG_PATH.exists():
        return []
    conn = sqlite3.connect(str(_KG_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    frontier, visited, neighbors = {entity}, {entity}, []
    for _ in range(radius):
        next_frontier = set()
        placeholders = ",".join("?" * len(frontier))
        cur.execute(
            f"SELECT subject, object FROM triples "
            f"WHERE subject IN ({placeholders}) OR object IN ({placeholders})",
            [*frontier, *frontier],
        )
        for row in cur.fetchall():
            for node in (row["subject"], row["object"]):
                if node not in visited and node != entity:
                    neighbors.append(node)
                    next_frontier.add(node)
                    visited.add(node)
        frontier = next_frontier
        if not frontier:
            break
    conn.close()
    return list(set(neighbors))


# --------------------------------------------------------------------------
# Core retrieval
# --------------------------------------------------------------------------
def retrieve(query: str, limit: int = 5) -> list[dict]:
    """Return the *limit* best experiences for *query*, ranked by combined score."""
    experiences = _load_store()
    if not experiences:
        return []

    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    n_docs = _N_DOCS
    avgdl  = _AVGDL
    doc_freqs = _DOC_FREQS
    UCB1_BASE = 1.0
    scores: list[tuple[float, int]] = []

    # Reload weights when the store is dirty (new entries appended) so that
    # coordinator-updated weights take effect on the NEXT retrieval after a batch
    global _RETRIEVAL_WEIGHTS
    if _DIRTY:
        _RETRIEVAL_WEIGHTS = _load_weights()

    w_bm25 = _RETRIEVAL_WEIGHTS["bm25"]
    w_kg   = _RETRIEVAL_WEIGHTS["kg"]
    w_ucb1 = _RETRIEVAL_WEIGHTS["ucb1"]

    for idx, exp in enumerate(experiences):
        doc_tokens = exp.get("_tokens", [])
        doc_len = len(doc_tokens)
        bm25 = _bm25_score(q_tokens, doc_tokens, doc_len, avgdl, n_docs, doc_freqs)

        # KG entity expansion
        kg_boost = 0.0
        for qt in q_tokens[:3]:
            neighbors = _kg_neighbors(qt)
            if neighbors:
                kg_boost += sum(
                    0.05 for nt in doc_tokens if nt in neighbors
                )

        # UCB1 bonus
        n = idx + 1
        ucb1 = UCB1_BASE + math.sqrt(2 * math.log(n_docs) / n)

        combined = bm25 * w_bm25 + kg_boost * w_kg + ucb1 * w_ucb1
        scores.append((combined, idx))

    scores.sort(reverse=True)
    top = [experiences[idx] for _, idx in scores[:limit]]
    # MemRL: log retrieval so coordinator can track outcome after session
    _log_retrieval_to_coordinator(top, query)
    return top


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def _fmt_exp(exp: dict) -> str:
    """Format a single experience as a readable text block."""
    domain  = exp.get("skill_domain", "general")
    outcome = exp.get("outcome", "unknown")
    icon = {"success": "OK", "error": "FAIL", "partial": "WARN"}.get(outcome, "?")
    body = exp.get("approach", "") or exp.get("task", "")
    task_short = exp.get("task", "")[:120]
    return (
        f"[{domain}][{outcome}] Task: {task_short}\n"
        f"  Approach: {body[:250]}\n"
        f"  Result: {icon} {outcome}"
    )


def _fmt_patch(patch: dict) -> str:
    """Format a runtime self-modification patch as advisory context."""
    ptype   = patch.get("type", "advisory")
    trigger = patch.get("trigger", "?")
    guidance = patch.get("guidance", patch.get("patch", ""))
    confidence = patch.get("confidence", 0.0)
    return (
        f"[RUNTIME-SELF-MOD] type={ptype} confidence={confidence:.2f}\n"
        f"  Trigger: {trigger}\n"
        f"  Guidance: {guidance[:400]}"
    )


def _fmt_si_outcomes() -> str:
    """Format the latest SI outcome summary."""
    try:
        if not _SI_OUTCOMES_FILE.exists():
            return ""
        with open(_SI_OUTCOMES_FILE) as f:
            data = json.load(f)
        snap = data.get("current_snapshot", {})
        trend = data.get("trend_summary", {})
        lines = ["[SI-OUTCOMES]"]
        err_trend = trend.get("error_rate_trend", "?")
        err_pct = trend.get("error_rate_change_pct", 0)
        filing_trend = trend.get("filing_rate_trend", "?")
        snap_14d = snap.get("sessions_14d", {})
        kg_stats = snap.get("kg_stats", {})
        lines.append(f"  Error rate trend: {err_trend} ({err_pct:+.1f}%)")
        lines.append(f"  Filing rate trend: {filing_trend}")
        lines.append(f"  14d sessions: {snap_14d.get('total_sessions', 0)}")
        lines.append(f"  Error rate: {snap_14d.get('error_rate', 0):.1%}")
        lines.append(f"  KG filing: {kg_stats.get('filing_rate_per_day', 0):.0f}/day")
        lines.append(f"  Total experiences: {snap.get('retrieval_stats', {}).get('total_experiences', 0)}")
        return "\n".join(lines)
    except Exception:
        return ""


def _fmt_belief_revisions() -> str:
    """Format pending belief revisions from the belief-revision diagnostic."""
    try:
        if not _BELIEF_REVISIONS_FILE.exists():
            return ""
        with open(_BELIEF_REVISIONS_FILE) as f:
            data = json.load(f)
        total = data.get("total_tan_beliefs", 0)
        fluid = data.get("fluid_domain_beliefs", 0)
        if total == 0:
            return ""
        lines = ["[BELIEF-REVISION] fluid domain beliefs requiring monitoring:"]
        lines.append(f"  {fluid}/{total} TAN beliefs are in fluid domains (models, GPU, drivers, etc.)")
        for b in data.get("fluid_beliefs_sample", [])[:3]:
            lines.append(f"  [{b.get('confidence', 0):.2f}] {b.get('subject', '')} {b.get('predicate', '')} {b.get('object', '')}")
        return "\n".join(lines)
    except Exception:
        return ""


def _fmt_session_errors() -> str:
    """Format cross-session error patterns for awareness."""
    try:
        if not _ERROR_AGGREGATOR_FILE.exists():
            return ""
        with open(_ERROR_AGGREGATOR_FILE) as f:
            data = json.load(f)
        auto = data.get("auto_triggers", [])
        if not auto:
            return ""
        lines = ["[SESSION-ERROR-PATTERNS] top cross-session error categories:"]
        for t in auto[:5]:
            lines.append(f"  {t['category']}: {t['session_count']} sessions (auto-triggered)")
        return "\n".join(lines)
    except Exception:
        return ""


def _fmt_model_releases() -> str:
    """Format newly detected AI model releases."""
    try:
        if not _MODEL_RELEASES_FILE.exists():
            return ""
        with open(_MODEL_RELEASES_FILE) as f:
            data = json.load(f)
        releases = data.get("releases", [])
        if not releases:
            return ""
        lines = [f"[MODEL-RELEASES] {len(releases)} new AI model releases detected:"]
        for r in releases[:6]:
            src = r.get("source", "?").upper()[:6]
            desc = r.get("description", "")[:100]
            lines.append(f"  [{src}] {desc}")
        return "\n".join(lines)
    except Exception:
        return ""


def _fmt_reasoning_strategy(strategy: dict) -> str:
    """Format a reasoning strategy selection."""
    stype  = strategy.get("strategy_type", "?")
    reason = strategy.get("reasoning", "")[:200]
    return (
        f"[REASONING-STRATEGY] {stype}\n"
        f"  Why: {reason}"
    )


def _fmt_critique(critique: dict) -> str:
    """Format a pending multi-agent critique."""
    status = critique.get("status", "?")
    verdict = critique.get("verdict", "")[:200]
    suggestions = "\n  ".join(critique.get("suggestions", [])[:3])
    return (
        f"[RUNTIME-CRITIQUE] status={status}\n"
        f"  Verdict: {verdict}\n"
        f"  Suggestions:\n  {suggestions}"
    )


def _fmt_research_memory(query: str) -> str:
    """Load and format hot topics from research-memory.json.

    Accepts two formats:
      - legacy: {"status": "ok", "hot_topics": [...], "updated_at": "...", ...}
      - injector: {"updated": "...", "hot_topics": {"hn": [...], "arxiv": [...], "models": [...]}, ...}
    """
    research_file = Path.home() / ".hermes" / "workspace" / "research-memory.json"
    if not research_file.exists():
        return ""
    try:
        age_secs = (datetime.now() - datetime.fromtimestamp(research_file.stat().st_mtime)).total_seconds()
        if age_secs > 7200:
            return ""
        rm = json.loads(research_file.read_text())

        # Determine which format
        updated_at = rm.get("updated_at") or rm.get("updated", "")
        hot = rm.get("hot_topics", [])
        trending = rm.get("trending_keywords", [])
        summary = rm.get("summary", {})

        # Normalize hot_topics to flat list
        items: list[tuple[str, str, str]] = []   # (source, title, url)
        if isinstance(hot, list):
            for h in hot[:10]:
                src = h.get("source", "hn")
                title = h.get("title", "")
                url = h.get("url", "")
                if title:
                    items.append((src, title, url))
        elif isinstance(hot, dict):
            for src, records in hot.items():
                if not isinstance(records, list):
                    continue
                for r in records[:5]:
                    title = r.get("title", "") or r.get("name", "")
                    url = r.get("url", "") or r.get("link", "")
                    if title:
                        items.append((src, title, url))

        if not items:
            return ""

        lines = ["[REAL-TIME-KNOWLEDGE] Live sources (refreshed ~10min):"]
        seen_titles = set()
        for src, title, url in items:
            if title in seen_titles:
                continue
            seen_titles.add(title)
            short_src = src[:4] if src else "misc"
            lines.append(f"  [{short_src}] {title[:70]}")
            if url:
                lines.append(f"         {url[:70]}")

        if trending and len(trending) > 0:
            keywords = trending[:8] if isinstance(trending, list) else []
            if keywords:
                lines.append(f"  trending: {', '.join(str(k) for k in keywords[:8])}")

        return "\n".join(lines)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# MemoryProvider interface
# --------------------------------------------------------------------------
class JitRLMemoryProvider:
    """Hermes Agent MemoryProvider -- Just-in-Time experience recall.

    Registered in ~/.hermes/config.yaml as memory.provider: jitrl.
    prefetch() returns a formatted string injected into the user message
    block before every LLM call (NOT persisted to session history).
    """

    store_path = _STORE_PATH

    def __init__(self):
        self._turn_count = 0
        self._session_id = ""

    @property
    def name(self) -> str:
        return "jitrl"

    def initialize(self, session_id: str = "", **kwargs) -> None:
        global _RETRIEVAL_WEIGHTS
        self._session_id = session_id
        # Warm the store and load weights from jitrl_weights.json
        _load_store()
        _RETRIEVAL_WEIGHTS = _load_weights()

    # -- MemoryProvider contract --------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Called before each LLM turn. Returns formatted text injected as context.

        Returns a string (NOT list[dict]) so MemoryManager.prefetch_all()
        can join it directly. Content includes:
          1. Retrieved JitRL experiences (BM25 + KG + UCB1)
          2. Runtime self-modification patches (from on_turn_start() errors)
          3. Reasoning strategy (from reasoning-strategy-selector.py)
          4. Pending multi-agent critique (from multi-agent-critique.py)
          5. Research memory (hot topics from research-memory.json)
        """
        global _PATCH_APPLIED
        _PATCH_APPLIED = False   # reset per new user message (new prefetch cycle)
        self._turn_count += 1

        parts: list[str] = []

        # 1. Retrieved JitRL experiences
        experiences = retrieve(query, limit=5)
        if experiences:
            exp_lines = ["[JITRL-EXPERIENCES]"]
            for i, exp in enumerate(experiences, 1):
                exp_lines.append(f"  {i}. {_fmt_exp(exp)}")
            parts.append("\n".join(exp_lines))

        # 2. Runtime self-modification patches
        patches = get_and_clear_patches()
        if patches:
            patch_lines = ["[RUNTIME-SELF-MOD] adjustments for this turn:"]
            for p in patches:
                patch_lines.append(f"  {_fmt_patch(p)}")
            parts.append("\n".join(patch_lines))

        # 3. Reasoning strategy
        strategy = get_current_reasoning_strategy(query)
        if strategy:
            parts.append("[STRATEGY] " + _fmt_reasoning_strategy(strategy))

        # 4. Pending runtime critique
        critique = get_pending_critique()
        if critique:
            parts.append("[CRITIQUE] " + _fmt_critique(critique))

        # 5. Research memory (live HN/arxiv/model releases -- refreshed every 10 min)
        research_ctx = _fmt_research_memory(query)
        if research_ctx:
            parts.append(research_ctx)

        # 6. SI outcome tracker (error rate, filing rate, trend)
        si_outcomes = _fmt_si_outcomes()
        if si_outcomes:
            parts.append(si_outcomes)

        # 7. Belief revision (fluid domain claims that may need updating)
        belief_ctx = _fmt_belief_revisions()
        if belief_ctx:
            parts.append(belief_ctx)

        # 8. Cross-session error patterns (aggregated from 30d of sessions)
        error_patterns = _fmt_session_errors()
        if error_patterns:
            parts.append(error_patterns)

        # 9. Newly detected AI model releases
        model_releases = _fmt_model_releases()
        if model_releases:
            parts.append(model_releases)

        return "\n\n".join(parts)

    def on_turn_start(
        self, turn_number: int, message: str, **kwargs
    ) -> None:
        """Called at the start of each turn.

        Detects error keywords in the incoming user message and triggers
        the runtime-self-modifier.py script to generate a context patch
        for this turn -- immediately, before the LLM call fires.
        """
        global _RUNTIME_ERRORS, _PATCH_APPLIED
        msg_lower = message.lower()

        # Check for failure language suggesting this turn may struggle
        failure_signals = [
            "error", "failed", "not working", "broken", "crash",
            "exception", "traceback", "does not work", "cannot",
            "unable to", "stuck",
        ]
        for sig in failure_signals:
            if sig in msg_lower:
                # Lightweight: write the signal to the patch file so the
                # runtime-self-modifier script can pick it up asynchronously.
                # The synchronous path is: on_turn_start writes sig → prefetch
                # reads it → LLM gets advisory context.
                try:
                    _RUNTIME_PATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
                    with open(_RUNTIME_PATCH_FILE, "a", errors="replace") as fh:
                        fh.write(json.dumps({
                            "type": "failure_signal",
                            "trigger": sig,
                            "query_hint": message[:200],
                            "turn": turn_number,
                            "session_id": self._session_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                break

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Called after each turn. Extracts outcome and appends to experience store.

        If the assistant returned an error, also fires runtime-self-modifier.py
        to generate a patch for the NEXT turn within this same session.
        """
        global _DIRTY

        # Determine outcome
        outcome = "success"
        error_pattern = ""
        content_lower = (user_content + " " + assistant_content).lower()
        if "error" in content_lower or "traceback" in content_lower or "failed" in content_lower:
            outcome = "error"
            for line in (user_content + "\n" + assistant_content).splitlines():
                if any(k in line.lower() for k in ["error", "traceback", "exception"]):
                    error_pattern = line.strip()[:100]
                    break

        # Extract tool names from assistant content
        tool_names: list[str] = []
        for m in re.findall(r'("name"\s*:\s*"([^"]+)")', assistant_content):
            tool_names.append(m[1])
        if not tool_names:
            tool_names = re.findall(r'\b(terminal|read_file|write_file|patch|search_files|delegate_task|cronjob|skill_view|memory)\b', assistant_content)

        skill_domain = "general"
        if tool_names:
            domain_map = {
                "terminal": "shell", "read_file": "code", "write_file": "code",
                "patch": "code", "skill_view": "skill", "skill_manage": "skill",
                "search_files": "research", "delegate_task": "agent",
                "cronjob": "system", "memory": "memory",
            }
            skill_domain = domain_map.get(tool_names[0], "tool")

        task = user_content[:300] if user_content else " ".join(tool_names[:3])[:100]
        approach = f"Tools: {', '.join(tool_names[:5])}"

        entry = {
            "task": task,
            "approach": approach,
            "outcome": outcome,
            "skill_domain": skill_domain,
            "error_pattern": error_pattern,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or f"turn_{self._turn_count}",
            "importance": 0.7 if tool_names else 0.5,
        }
        _append_entry(entry)

        # MemRL: log turn outcome to coordinator — closes the retrieval-outcome feedback loop
        _log_outcome_to_coordinator(outcome, session_id or self._session_id)

        # If error: trigger runtime-self-modifier asynchronously for next turn
        if outcome == "error" and error_pattern:
            import subprocess
            try:
                subprocess.Popen(
                    [
                        "python3",
                        str(Path.home() / ".hermes" / "scripts" / "runtime-self-modifier.py"),
                        "--error", error_pattern,
                        "--query", task,
                        "--session", session_id or "",
                    ],
                    cwd=str(Path.home() / ".hermes"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    def on_session_end(self, messages: list[dict]) -> None:
        """Called at session end. Extracts final outcome and appends to store."""
        if not messages:
            return
        outcome = "success"
        for msg in reversed(messages):
            content: str = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            if "error" in content.lower() or "traceback" in content.lower():
                outcome = "error"
                break
        task = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content: str = msg.get("content", "") or ""
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                if len(content) >= 20:
                    task = content[:300]
                    break
        if task:
            entry = {
                "task": task,
                "approach": "Session ended",
                "outcome": outcome,
                "skill_domain": "conversation",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "importance": 0.6,
            }
            _append_entry(entry)
            _log_outcome_to_coordinator(
                outcome,
                f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            )

    # -- optional hooks (noop for now) --------------------------------------
    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def get_tool_schemas(self) -> list[dict]:
        return []

    def shutdown(self) -> None:
        pass


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _append_entry(entry: dict) -> None:
    """Append a JSONL entry to the experience store."""
    global _DIRTY
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STORE_PATH, "a", errors="replace") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    entry["_tokens"] = _tokenize(entry.get("task", "") + " " + entry.get("approach", ""))
    _EXPERIENCES.append(entry)
    _DIRTY = True


# --------------------------------------------------------------------------
# Plugin discovery
# --------------------------------------------------------------------------
def load() -> JitRLMemoryProvider:
    return JitRLMemoryProvider()
