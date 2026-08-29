"""Verified, provenance-preserving cultural inheritance for artifact programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .programs import ArtifactProgram, instruction_diff

MAX_RETAINED_SKILL_EVIDENCE = 16
MAX_RETAINED_SKILL_MEASUREMENTS = 8

PROGRAM_FORK_AUTHORSHIP_VERSION = 2


def program_fork_authorship_summary(
    programs: Mapping[str, Mapping[str, Any]],
    lineage_edges: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
) -> dict[str, int | float]:
    """Count identifiable agent-to-agent descent conservatively.

    Starter programs and any other non-agent-authored parents are not eligible
    opportunities for cross-agent transmission. A parent program with several
    registered authors is counted as cross-agent only when the child author is
    not among those known parent authors.
    """

    agents = set(map(str, agent_ids))
    eligible = 0
    cross_agent = 0
    for edge in lineage_edges:
        parent = programs.get(str(edge.get("parent_program_id", "")), {})
        parent_agent_authors = {
            str(author)
            for author in parent.get("authors", [])
            if str(author) in agents
        }
        child_author = str(edge.get("author", ""))
        if not parent_agent_authors or child_author not in agents:
            continue
        eligible += 1
        cross_agent += int(child_author not in parent_agent_authors)
    return {
        "program_fork_authorship_version": PROGRAM_FORK_AUTHORSHIP_VERSION,
        "agent_parent_program_forks": eligible,
        "cross_agent_program_forks": cross_agent,
        "cross_agent_program_fork_fraction": round(
            cross_agent / max(1, eligible), 6
        ),
    }


class ProgramLibrary:
    """Authoritative program registry plus bounded per-agent knowledge indices.

    A library entry grants knowledge of a program, never material access or a physics
    advantage. ``verified`` means the agent obtained a measured artifact observation;
    it is deliberately not an evaluator pass/fail label.
    """

    def __init__(self, agent_ids: list[str]) -> None:
        self.programs: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, dict[str, Any]]] = {
            agent_id: {} for agent_id in agent_ids
        }
        self.lineage_edges: list[dict[str, Any]] = []

    def register(
        self,
        program: ArtifactProgram,
        *,
        tick: int,
        artifact_id: str,
        parent: ArtifactProgram | None = None,
        index_skill: bool = True,
    ) -> dict[str, Any]:
        program_id = program.program_id
        record = self.programs.setdefault(
            program_id,
            {
                "program_id": program_id,
                "name": program.name,
                "instructions": [dict(item) for item in program.instructions],
                "authors": [],
                "first_tick": tick,
                "installations": [],
            },
        )
        if program.author not in record["authors"]:
            record["authors"].append(program.author)
            record["authors"].sort()
        installation = {"tick": tick, "artifact_id": artifact_id, "author": program.author}
        record["installations"].append(installation)
        parent_id = parent.program_id if parent is not None else None
        if parent_id is not None:
            edge = {
                "parent_program_id": parent_id,
                "child_program_id": program_id,
                "tick": tick,
                "artifact_id": artifact_id,
                "author": program.author,
                "instruction_diff": instruction_diff(parent, program),
            }
            self.lineage_edges.append(edge)
        if index_skill:
            self.observe(
                program.author,
                program_id,
                tick=tick,
                source="authored",
                evidence_id=artifact_id,
                verified=False,
            )
        return {**installation, "program_id": program_id, "parent_program_id": parent_id}

    def observe(
        self,
        agent_id: str,
        program_id: str,
        *,
        tick: int,
        source: str,
        evidence_id: str = "",
        verified: bool = False,
        measurements: dict[str, Any] | None = None,
    ) -> None:
        if agent_id not in self.skills or program_id not in self.programs:
            return
        entry = self.skills[agent_id].setdefault(
            program_id,
            {
                "program_id": program_id,
                "first_observed_tick": tick,
                "last_observed_tick": tick,
                "sources": [],
                "evidence_ids": [],
                "evidence_count": 0,
                "verified": False,
                "measurements": [],
                "measurement_count": 0,
                "inherited": False,
            },
        )
        entry["last_observed_tick"] = tick
        if source not in entry["sources"]:
            entry["sources"].append(source)
            entry["sources"].sort()
        if evidence_id and evidence_id not in entry["evidence_ids"]:
            entry["evidence_ids"].append(evidence_id)
            entry["evidence_ids"] = entry["evidence_ids"][-MAX_RETAINED_SKILL_EVIDENCE:]
            entry["evidence_count"] += 1
        entry["verified"] = bool(entry["verified"] or verified)
        if measurements is not None:
            entry["measurements"].append({"tick": tick, **measurements})
            entry["measurements"] = entry["measurements"][
                -MAX_RETAINED_SKILL_MEASUREMENTS:
            ]
            entry["measurement_count"] += 1

    def knows(self, agent_id: str, program_id: str) -> bool:
        return program_id in self.skills.get(agent_id, {})

    def would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """Content rediscovery may repeat a node, but ancestry remains a DAG."""

        adjacency: dict[str, set[str]] = {}
        for edge in self.lineage_edges:
            adjacency.setdefault(str(edge["parent_program_id"]), set()).add(
                str(edge["child_program_id"])
            )
        stack = [child_id]
        visited: set[str] = set()
        while stack:
            node = stack.pop()
            if node == parent_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency.get(node, ()))
        return False

    def resolve(self, program_id: str) -> ArtifactProgram | None:
        record = self.programs.get(program_id)
        if record is None:
            return None
        return ArtifactProgram.from_dict(
            {"name": record["name"], "instructions": record["instructions"]},
            author=str(record["authors"][0]) if record["authors"] else "unknown",
        )

    def transfer(
        self, sender: str, recipient: str, program_id: str, *, tick: int, evidence_id: str
    ) -> bool:
        source = self.skills.get(sender, {}).get(program_id)
        if source is None:
            return False
        self.observe(
            recipient,
            program_id,
            tick=tick,
            source=f"taught_by:{sender}",
            evidence_id=evidence_id,
            verified=bool(source["verified"]),
            measurements=(
                dict(source["measurements"][-1]) if source["measurements"] else None
            ),
        )
        return True

    def inherit(self, agent_id: str, *, limit: int, tick: int) -> list[str]:
        """Keep only the strongest verified cultural records across replacement."""

        existing = self.skills.get(agent_id, {})
        ranked = sorted(
            (entry for entry in existing.values() if entry["verified"]),
            key=lambda entry: (
                -len(entry["measurements"]),
                -int(entry["last_observed_tick"]),
                str(entry["program_id"]),
            ),
        )[:limit]
        inherited: dict[str, dict[str, Any]] = {}
        for original in ranked:
            entry = dict(original)
            entry["sources"] = ["cultural_inheritance"]
            entry["first_observed_tick"] = tick
            entry["last_observed_tick"] = tick
            entry["inherited"] = True
            inherited[str(entry["program_id"])] = entry
        self.skills[agent_id] = inherited
        return sorted(inherited)

    def clear_agent(self, agent_id: str) -> None:
        self.skills[agent_id] = {}

    def view(
        self,
        agent_id: str,
        limit: int = 12,
        *,
        evidence_limit: int = 4,
        measurement_limit: int = 3,
        focus_program_ids: set[str] | None = None,
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        focus = set(focus_program_ids or ())
        entries = sorted(
            self.skills.get(agent_id, {}).values(),
            key=lambda entry: (
                str(entry["program_id"]) not in focus,
                not bool(entry["verified"]),
                -int(entry["last_observed_tick"]),
                str(entry["program_id"]),
            ),
        )[:limit]
        result: list[dict[str, Any]] = []
        for entry in entries:
            program = self.programs[str(entry["program_id"])]
            if not compact:
                result.append(
                    {
                        **entry,
                        "name": program["name"],
                        "instructions": program["instructions"],
                    }
                )
                continue
            result.append(
                {
                    "program_id": entry["program_id"],
                    "name": program["name"],
                    "instructions": program["instructions"],
                    "verified": bool(entry["verified"]),
                    "inherited": bool(entry["inherited"]),
                    "sources": list(entry["sources"]),
                    "first_observed_tick": int(entry["first_observed_tick"]),
                    "last_observed_tick": int(entry["last_observed_tick"]),
                    "evidence_count": int(entry.get("evidence_count", 0)),
                    "recent_evidence_ids": list(entry["evidence_ids"])[-evidence_limit:],
                    "measurement_count": int(entry.get("measurement_count", 0)),
                    "recent_measurements": list(entry["measurements"])[
                        -measurement_limit:
                    ],
                }
            )
        return result

    def authoritative_state(self) -> dict[str, Any]:
        return {
            "programs": self.programs,
            "skills": self.skills,
            "lineage_edges": self.lineage_edges,
        }
