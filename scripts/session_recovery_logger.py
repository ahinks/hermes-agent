"""
Session Recovery Logger for HinksBot.

Records every session failure for post-mortem analysis and recovery.
Logs to ~/.hermes/workspace/logs/session-recovery/.

Used by run_agent.py's AIAgent._log_session_failure() method.
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ─── Enums ────────────────────────────────────────────────────────────────────


class SessionEndState(Enum):
    """Valid end states for a session."""

    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"
    COMPRESSION_EXHAUSTED = "compression_exhausted"
    RATE_LIMIT_EXHAUSTED = "rate_limit_exhausted"
    AUTH_FAILURE = "***"


class ErrorSource(Enum):
    """Source/location that initiated an error."""

    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    AUTH_FAILURE = "***"
    TOOL_ERROR = "tool_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


# ─── Logger ───────────────────────────────────────────────────────────────────


class SessionRecoveryLogger:
    """
    Records session failures to disk for post-mortem analysis.

    Log format (one JSON object per line, newline-delimited JSON, .jsonl):
        ~/.hermes/workspace/logs/session-recovery/sessions.jsonl
    """

    SESSION_LOG_DIR = Path.home() / ".hermes" / "workspace" / "logs" / "session-recovery"
    SESSION_LOG_FILE = SESSION_LOG_DIR / "sessions.jsonl"

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else self.SESSION_LOG_DIR
        self.log_file = self.log_dir / "sessions.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def log_failure(
        self,
        session_id: str,
        end_state: SessionEndState,
        error_source: ErrorSource,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        recovery_action: Optional[str] = None,
        last_tool_name: Optional[str] = None,
        api_call_count: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_seconds: float = 0.0,
        notes: str = "",
    ) -> None:
        """
        Append a session failure record to the log.

        Args:
            session_id:       Unique session identifier
            end_state:        SessionEndState enum value
            error_source:     ErrorSource enum value
            error_message:    Human-readable error message
            error_code:       Machine-readable error code (e.g. "rate_limit", "context_overflow")
            provider:         API provider name (e.g. "openai", "anthropic")
            model:            Model name used in this session
            recovery_action: What recovery was attempted (e.g. "compress", "retry", "abort")
            last_tool_name:   Last tool called before failure
            api_call_count:   Number of API calls made
            tokens_in:        Total input tokens used
            tokens_out:       Total output tokens used
            duration_seconds: How long the session ran
            notes:            Free-form notes
        """
        now = datetime.now(timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "end_state": end_state.value if isinstance(end_state, SessionEndState) else end_state,
            "error_source": error_source.value if isinstance(error_source, ErrorSource) else error_source,
            "error_message": error_message,
            "error_code": error_code,
            "provider": provider,
            "model": model,
            "recovery_action": recovery_action,
            "last_tool_name": last_tool_name,
            "api_call_count": api_call_count,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration_seconds": duration_seconds,
            "notes": notes,
            "stack_trace": traceback.format_exc() if error_message else None,
            "system_state": self._get_system_state(),
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _get_system_state(self) -> dict:
        """Capture system state at time of failure."""
        import shutil

        state = {}
        try:
            state["memory_percent"] = self._memory_usage_percent()
        except Exception:
            state["memory_percent"] = None

        try:
            state["disk_free_gb"] = shutil.disk_usage("/").free / (1024**3)
        except Exception:
            state["disk_free_gb"] = None

        return state

    @staticmethod
    def _memory_usage_percent() -> float:
        """Return memory usage as a percentage (0-100)."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback: read /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            avail_kb = int(line.split()[1])
                        elif line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                    return round((1 - avail_kb / total_kb) * 100, 1)
            except Exception:
                return 0.0

    def get_recent_failures(self, limit: int = 50) -> list[dict]:
        """Read recent failure records from the log."""
        if not self.log_file.exists():
            return []

        records = []
        try:
            with open(self.log_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return []

        return records[-limit:]
