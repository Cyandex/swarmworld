"""Bounded private memories and an append-only cultural archive."""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MemoryRecord:
    record_id: str
    tick: int
    kind: str
    content: str
    salience: float = 0.5
    causal_parents: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentMemory:
    def __init__(self, capacity: int) -> None:
        self.working: deque[MemoryRecord] = deque(maxlen=max(4, min(capacity, 16)))
        self.episodic: deque[MemoryRecord] = deque(maxlen=capacity)
        self.notebook: deque[MemoryRecord] = deque(maxlen=max(capacity, 32))
        self.last_retrieval: dict[str, Any] = {}
        self.retrieval_stats: dict[str, dict[str, int]] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Deterministic lexical normalization suitable for IDs and scientific text."""

        return re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", text.casefold())

    def remember(self, record: MemoryRecord, notebook: bool = False) -> None:
        self.working.append(record)
        self.episodic.append(record)
        if notebook:
            self.notebook.append(record)

    def retrieve(
        self,
        query_terms: set[str],
        limit: int = 8,
        *,
        adaptive: bool = False,
        feedback_weight: float = 0.45,
        exploration_weight: float = 0.12,
    ) -> list[MemoryRecord]:
        records_by_id: dict[str, MemoryRecord] = {}
        for record in list(self.episodic) + list(self.notebook):
            records_by_id[record.record_id] = record
        records = list(records_by_id.values())
        normalized_query = set(self.tokenize(" ".join(sorted(query_terms))))
        tokenized = {record.record_id: self.tokenize(record.content) for record in records}
        document_frequency: dict[str, int] = {}
        for terms in tokenized.values():
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        average_length = (
            sum(len(terms) for terms in tokenized.values()) / max(1, len(tokenized))
        )
        scored: list[tuple[float, MemoryRecord, dict[str, Any]]] = []
        for record in records:
            terms = tokenized[record.record_id]
            counts = {term: terms.count(term) for term in normalized_query if term in terms}
            lexical = 0.0
            for term, frequency in counts.items():
                df = document_frequency.get(term, 0)
                inverse = math.log(1.0 + (len(records) - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * len(terms) / max(1.0, average_length)
                )
                lexical += inverse * frequency * 2.2 / denominator
            stats = self.retrieval_stats.get(record.record_id, {})
            selected = int(stats.get("selected", 0))
            used = int(stats.get("used", 0))
            outcome_attempts = int(stats.get("outcome_attempts", 0))
            successful_outcomes = int(stats.get("successful_outcomes", 0))
            citation_rate = used / max(1, selected)
            causal_success_rate = successful_outcomes / max(1, outcome_attempts)
            # Self-selection is weak evidence. A later successful action carrying the
            # record as a causal parent supplies the other half of the general signal.
            feedback = 0.5 * citation_rate + 0.5 * (
                causal_success_rate if outcome_attempts else 0.0
            )
            total_selections = sum(
                int(item.get("selected", 0))
                for item in self.retrieval_stats.values()
            )
            exploration = math.sqrt(
                math.log1p(total_selections + len(records)) / (selected + 1)
            )
            adaptive_bonus = (
                feedback_weight * feedback + exploration_weight * exploration
                if adaptive
                else 0.0
            )
            score = (
                record.salience
                + 0.22 * lexical
                + adaptive_bonus
                + 1e-9 * record.tick
            )
            scored.append(
                (
                    score,
                    record,
                    {
                        "record_id": record.record_id,
                        "score": round(score, 8),
                        "lexical_score": round(lexical, 8),
                        "feedback_rate": round(feedback, 8),
                        "citation_rate": round(citation_rate, 8),
                        "causal_success_rate": round(causal_success_rate, 8),
                        "causal_outcome_attempts": outcome_attempts,
                        "exploration_bonus": round(exploration, 8),
                        "adaptive_bonus": round(adaptive_bonus, 8),
                        "matched_terms": sorted(counts),
                        "salience": record.salience,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        seen: set[str] = set()
        result: list[MemoryRecord] = []
        selected_diagnostics: list[dict[str, Any]] = []
        for _, record, diagnostic in scored:
            if record.record_id in seen:
                continue
            seen.add(record.record_id)
            result.append(record)
            selected_diagnostics.append(diagnostic)
            stats = self.retrieval_stats.setdefault(record.record_id, {})
            stats.setdefault("selected", 0)
            stats.setdefault("used", 0)
            stats.setdefault("outcome_attempts", 0)
            stats.setdefault("successful_outcomes", 0)
            stats["selected"] += 1
            if len(result) >= limit:
                break
        self.last_retrieval = {
            "query_terms": sorted(normalized_query),
            "candidate_count": len(records),
            "selected": selected_diagnostics,
        }
        return result

    @staticmethod
    def context_excerpt(record: MemoryRecord, max_characters: int) -> str:
        """Render bounded evidence without changing the authoritative stored record."""

        content = record.content
        if len(content) <= max_characters:
            return content
        try:
            value = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return content[: max(0, max_characters - 24)] + "...[bounded excerpt]"
        if isinstance(value, dict) and isinstance(value.get("artifact_measurements"), list):
            measurements = value["artifact_measurements"]
            value = dict(value)
            value["artifact_measurement_count"] = len(measurements)
            value["artifact_measurements"] = [
                {
                    "artifact_id": item.get("artifact_id"),
                    "program_id": item.get("program_id"),
                    "performance": item.get("performance"),
                }
                for item in measurements
                if isinstance(item, dict)
            ]
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if len(rendered) <= max_characters:
            return rendered
        return rendered[: max(0, max_characters - 24)] + "...[bounded excerpt]"

    def mark_retrieval_use(self, cited_record_ids: set[str]) -> None:
        selected = {
            str(item["record_id"]) for item in self.last_retrieval.get("selected", [])
        }
        used = sorted(selected & cited_record_ids)
        for record_id in used:
            stats = self.retrieval_stats.setdefault(record_id, {})
            stats.setdefault("selected", 0)
            stats.setdefault("used", 0)
            stats.setdefault("outcome_attempts", 0)
            stats.setdefault("successful_outcomes", 0)
            stats["used"] += 1
        self.last_retrieval["used_as_causal_parent"] = used

    def mark_causal_outcome(
        self, causal_parent_ids: set[str], *, success: bool
    ) -> None:
        """Credit cited retained experience from an executed world consequence."""

        for record_id in sorted(causal_parent_ids):
            stats = self.retrieval_stats.get(record_id)
            if stats is None:
                continue
            stats.setdefault("outcome_attempts", 0)
            stats.setdefault("successful_outcomes", 0)
            stats["outcome_attempts"] += 1
            stats["successful_outcomes"] += int(success)

    def get(self, record_id: str) -> MemoryRecord | None:
        """Resolve a record the agent actually retains, preferring durable notes."""

        for collection in (self.notebook, self.episodic, self.working):
            for record in reversed(collection):
                if record.record_id == record_id:
                    return record
        return None

    def latest(self, kind: str) -> MemoryRecord | None:
        """Return the newest retained record of one semantic kind, if any."""

        candidates = (
            record
            for collection in (self.notebook, self.episodic, self.working)
            for record in reversed(collection)
            if record.kind == kind
        )
        return max(candidates, key=lambda record: record.tick, default=None)


class CulturalArchive:
    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self.by_id: dict[str, MemoryRecord] = {}

    def publish(self, record: MemoryRecord) -> None:
        if record.record_id in self.by_id:
            return
        self.records.append(record)
        self.by_id[record.record_id] = record

    def recent(self, limit: int = 32) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records[-limit:]]

    def retrieve(self, query_text: str, limit: int = 8) -> list[dict[str, Any]]:
        """Deterministically retrieve public evidence using agent-authored focus text."""

        if not self.records or limit <= 0:
            return []
        query = set(AgentMemory.tokenize(query_text))
        if not query:
            return self.recent(limit)
        tokenized = {
            record.record_id: AgentMemory.tokenize(record.content)
            for record in self.records
        }
        document_frequency: dict[str, int] = {}
        for terms in tokenized.values():
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        average_length = sum(map(len, tokenized.values())) / max(1, len(tokenized))
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self.records:
            terms = tokenized[record.record_id]
            counts = {term: terms.count(term) for term in query if term in terms}
            lexical = 0.0
            for term, frequency in counts.items():
                df = document_frequency.get(term, 0)
                inverse = math.log(
                    1.0 + (len(self.records) - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * len(terms) / max(1.0, average_length)
                )
                lexical += inverse * frequency * 2.2 / denominator
            scored.append((record.salience + 0.22 * lexical + 1e-9 * record.tick, record))
        scored.sort(key=lambda item: (-item[0], item[1].record_id))
        return [record.as_dict() for _, record in scored[:limit]]
