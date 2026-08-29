"""The human slot: possession, a bounded input queue, and the sidecar recorder."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from biofoundry.types import AgentAction


class HumanSlot:
    """One possessable agent slot with a bounded, ordered input queue."""

    def __init__(self, agent_id: str, queue_limit: int):
        self.agent_id = agent_id
        self.queue_limit = queue_limit
        self.owner: str | None = None  # connection id of the possessing client
        self._queue: list[tuple[AgentAction, str]] = []
        self.input_event = asyncio.Event()

    @property
    def joined(self) -> bool:
        return self.owner is not None

    @property
    def is_empty(self) -> bool:
        return not self._queue

    @property
    def pending(self) -> int:
        return len(self._queue)

    def join(self, connection_id: str) -> tuple[bool, str]:
        if self.owner is not None and self.owner != connection_id:
            return False, "slot already possessed by another player"
        self.owner = connection_id
        return True, ""

    def release(self, connection_id: str | None = None) -> bool:
        """Release possession; ``None`` forces release (e.g. server shutdown)."""

        if self.owner is None:
            return False
        if connection_id is not None and self.owner != connection_id:
            return False
        self.owner = None
        self._queue.clear()
        self.input_event.clear()
        return True

    def submit(
        self, connection_id: str, action: AgentAction, request_id: str
    ) -> tuple[bool, str]:
        if self.owner is None:
            return False, "join before sending human_action"
        if self.owner != connection_id:
            return False, "slot is possessed by another connection"
        if len(self._queue) >= self.queue_limit:
            return False, f"input queue full ({self.queue_limit})"
        self._queue.append((action, request_id))
        self.input_event.set()
        return True, ""

    def pop(self) -> tuple[AgentAction, str] | None:
        if not self._queue:
            return None
        item = self._queue.pop(0)
        if not self._queue:
            self.input_event.clear()
        return item


class HumanSidecar:
    """Append-only JSONL of human/session provenance beside the canonical trace.

    The canonical trace stays byte-compatible with every existing replay and
    analysis tool; this file records which actions came from a person, plus
    control commands, joins, and the session settings. Lines are flushed
    immediately — the volume is human-scale.
    """

    def __init__(self, trace_path: str | Path, session_metadata: dict[str, Any]):
        trace_path = Path(trace_path)
        self.path = trace_path.with_name(trace_path.name + ".human.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self.write({"type": "session", **session_metadata})

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def write(self, record: dict[str, Any]) -> None:
        if self._handle.closed:
            return
        stored = {"wall_time": self._now(), **record}
        self._handle.write(json.dumps(stored, sort_keys=True, separators=(",", ":")))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()
