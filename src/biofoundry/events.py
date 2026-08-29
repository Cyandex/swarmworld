"""Versioned JSONL recording and stable state digests."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .types import PROTOCOL_VERSION, WorldEvent


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def digest_snapshot(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()


class EventRecorder:
    def __init__(
        self,
        path: str | Path,
        metadata: dict[str, Any],
        *,
        compression: str = "none",
        deduplicate_prompts: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if compression not in {"none", "gzip"}:
            raise ValueError("compression must be 'none' or 'gzip'")
        self.handle = (
            gzip.open(self.path, "wt", encoding="utf-8")
            if compression == "gzip"
            else self.path.open("w", encoding="utf-8")
        )
        self.deduplicate_prompts = deduplicate_prompts
        self._written_templates: set[str] = set()
        self.write_record(
            {
                "type": "header",
                "protocol": PROTOCOL_VERSION,
                "metadata": metadata,
            }
        )

    def write_record(self, record: dict[str, Any]) -> None:
        stored = dict(record)
        templates = stored.pop("_trace_templates", None)
        if self.deduplicate_prompts and isinstance(templates, dict):
            replacements: list[tuple[str, str]] = []
            for name, value in sorted(templates.items()):
                if not isinstance(value, str) or not value:
                    continue
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
                template_id = f"{name}_{digest}"
                if template_id not in self._written_templates:
                    self._write_raw(
                        {"type": "trace_template", "template_id": template_id, "text": value}
                    )
                    self._written_templates.add(template_id)
                replacements.append((value, f"{{{{trace_template:{template_id}}}}}"))
            stored = _replace_strings(stored, replacements)
        self._write_raw(stored)

    def _write_raw(self, record: dict[str, Any]) -> None:
        self.handle.write(canonical_json(record))
        self.handle.write("\n")

    def write_events(self, events: Iterable[WorldEvent]) -> None:
        for event in events:
            self.write_record({"type": "event", **event.as_dict()})

    def write_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        state_digest: str | None = None,
    ) -> str:
        digest = digest_snapshot(snapshot)
        record = {"type": "snapshot", "digest": digest, "snapshot": snapshot}
        if state_digest is not None:
            record["state_digest"] = state_digest
        self.write_record(record)
        self.handle.flush()
        return state_digest or digest

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            self.handle.close()

    def __enter__(self) -> EventRecorder:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _replace_strings(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for before, after in replacements:
            value = value.replace(before, after)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    return value


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    templates: dict[str, str] = {}
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("type") == "trace_template":
                    templates[str(record["template_id"])] = str(record["text"])
                    yield record
                    continue
                replacements = [
                    (f"{{{{trace_template:{template_id}}}}}", text)
                    for template_id, text in templates.items()
                ]
                yield _replace_strings(record, replacements)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
