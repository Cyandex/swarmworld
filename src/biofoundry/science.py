"""Grounded collective-science mission state and objective evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .artifacts import SERVICE_NAMES
from .config import ScienceConfig
from .materials import MaterialBatch, MaterialLab, required_inputs
from .types import Resource

if TYPE_CHECKING:
    from .scenarios import ScenarioDefinition


@dataclass(slots=True)
class ExperimentalBatch:
    agent: str
    recipe_id: str
    source_recipe: dict[str, Any]
    batch: MaterialBatch
    tested: bool = False


class ResearchMission:
    """Deterministic mission scorecard hidden behind observable experiment results."""

    def __init__(
        self,
        config: ScienceConfig,
        *,
        scenario: ScenarioDefinition | None = None,
    ):
        self.config = config
        self.scenario = scenario
        if scenario is None:
            self._material_evaluator = MaterialLab()
        else:
            from .scenarios import ScenarioMaterialLab

            self._material_evaluator = ScenarioMaterialLab(scenario)
        self.depot: NDArray[np.float32] = np.zeros(len(Resource), dtype=np.float32)
        self.depot_contributions: list[dict[str, float]] = [
            {} for _ in range(len(Resource))
        ]
        self.depositor_history: set[str] = set()
        self.inspections: list[dict[str, Any]] = []
        self.proposals: list[dict[str, Any]] = []
        self.batches: list[ExperimentalBatch] = []
        self.tests: list[dict[str, Any]] = []
        self.claims: dict[str, str] = {}
        self.publications: dict[str, dict[str, Any]] = {}

    def _normalize_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        if self.scenario is None:
            return json.loads(json.dumps(recipe))
        return self.scenario.normalize_recipe(recipe)

    def _public_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        if self.scenario is None:
            return json.loads(json.dumps(recipe))
        return self.scenario.externalize_recipe(recipe)

    def _required_inputs(self, recipe: dict[str, Any]) -> dict[Resource, float]:
        if self.scenario is None:
            return required_inputs(recipe)
        return self._material_evaluator.required_inputs(recipe)

    def _recipe_id(self, recipe: dict[str, Any]) -> str:
        return self.recipe_id(self._normalize_recipe(recipe))

    def _resource_name(self, resource: Resource | int) -> str:
        value = Resource(int(resource))
        if self.scenario is None:
            return value.name
        return self.scenario.identifier("resources", int(value))

    @staticmethod
    def operative_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
        """Return the scale-invariant recipe fields that determine properties.

        MaterialLab uses relative feedstock composition and ordered unit operations
        to determine material properties. Absolute mass controls only batch size;
        names and design prose are descriptive metadata. Keeping those fields out
        of the identity lets a tested coupon validate a geometrically larger build
        without treating two physically different processes as equivalent.
        """

        required = required_inputs(recipe)
        total_mass = float(sum(required.values()))
        composition = [
            {
                "resource": resource.name,
                "fraction": round(float(required[resource]) / total_mass, 12),
            }
            for resource in sorted(required, key=int)
        ]
        steps = [
            {
                "operation": str(step["operation"]).upper(),
                "intensity": round(float(step["intensity"]), 12),
            }
            for step in recipe.get("steps", [])
        ]
        return {
            "identity_version": 2,
            "composition": composition,
            "steps": steps,
        }

    @classmethod
    def recipe_id(cls, recipe: dict[str, Any]) -> str:
        encoded = json.dumps(
            cls.operative_recipe(recipe),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "recipe_" + hashlib.sha256(encoded).hexdigest()[:12]

    def scaled_recipe(self, recipe: dict[str, Any]) -> dict[str, Any]:
        scale = self.config.test_scale
        return {
            "inputs": [
                {"resource": item["resource"], "mass": float(item["mass"]) * scale}
                for item in recipe["inputs"]
            ],
            "steps": [dict(step) for step in recipe["steps"]],
            "output_form": str(recipe.get("output_form", "experimental_coupon")),
            "design_principles": list(recipe.get("design_principles", ["controlled_test"])),
        }

    @staticmethod
    def material_utility(batch: MaterialBatch) -> float:
        properties = batch.properties
        functional_peak = max(
            properties["permeability"],
            properties["adhesion"],
            properties["healing"],
            properties["responsiveness"],
        )
        functional_second = sorted(
            (
                properties["permeability"],
                properties["adhesion"],
                properties["healing"],
                properties["responsiveness"],
            )
        )[-2]
        value = (
            0.24 * properties["quality"]
            + 0.20 * (1.0 - properties["degradation"])
            + 0.16 * properties["toughness"]
            + 0.10 * properties["stiffness"]
            + 0.20 * functional_peak
            + 0.10 * functional_second
        )
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def behavioral_novelty(artifacts: Any, index: int) -> float:
        fingerprint = artifacts.lifetime_peak_services[index].astype(np.float64)
        baseline_distance = float(
            np.linalg.norm(fingerprint) / np.sqrt(len(fingerprint))
        )
        if index == 0:
            return baseline_distance
        distances = np.linalg.norm(
            artifacts.lifetime_peak_services[:index].astype(np.float64) - fingerprint,
            axis=1,
        ) / np.sqrt(len(fingerprint))
        return float(min(baseline_distance, float(np.min(distances))))

    def portfolio_metrics(self, artifacts: Any) -> dict[str, Any]:
        """Measure collective service coverage without rewarding artifact count."""

        if artifacts.count == 0:
            coverage = np.zeros(len(SERVICE_NAMES), dtype=np.float64)
            return {
                "service_coverage": dict.fromkeys(SERVICE_NAMES, 0.0),
                "portfolio_resilience": 0.0,
                "service_breadth": 0,
                "portfolio_contributors": 0,
                "mean_behavioral_redundancy": None,
                "self_organized_portfolio": False,
            }
        fingerprints = artifacts.lifetime_peak_services[: artifacts.count].astype(
            np.float64
        )
        coverage = np.max(fingerprints, axis=0)
        coverage_mean = float(np.mean(coverage))
        coverage_balance = (
            float(np.min(coverage) / coverage_mean) if coverage_mean > 1e-12 else 0.0
        )
        resilience = coverage_mean * (0.5 + 0.5 * coverage_balance)
        breadth = int(np.count_nonzero(coverage >= self.config.target_artifact_performance))
        winners = np.argmax(fingerprints, axis=0)
        contributing_agents = {
            artifacts.creator[int(winner)]
            for service, winner in enumerate(winners)
            if coverage[service] >= self.config.target_artifact_performance
        }
        normalized = np.linalg.norm(fingerprints, axis=1)
        pairwise: list[float] = []
        for first in range(artifacts.count):
            if normalized[first] <= 1e-12:
                continue
            for second in range(first + 1, artifacts.count):
                if normalized[second] <= 1e-12:
                    continue
                pairwise.append(
                    float(
                        np.dot(fingerprints[first], fingerprints[second])
                        / (normalized[first] * normalized[second])
                    )
                )
        return {
            "service_coverage": {
                name: round(float(coverage[index]), 6)
                for index, name in enumerate(SERVICE_NAMES)
            },
            "portfolio_resilience": round(float(np.clip(resilience, 0.0, 1.0)), 6),
            "service_breadth": breadth,
            "portfolio_contributors": len(contributing_agents),
            "mean_behavioral_redundancy": (
                round(float(np.mean(pairwise)), 6) if pairwise else None
            ),
            "self_organized_portfolio": (
                breadth >= 2 and len(contributing_agents) >= self.config.minimum_contributors
            ),
        }

    def record_inspection(self, record: dict[str, Any]) -> None:
        self.inspections.append(record)

    def register_publication(
        self,
        record_id: str,
        *,
        author: str,
        tick: int,
        grounded_resources: set[Resource],
        causal_parents: list[str],
    ) -> None:
        """Store structured provenance that can be cited without parsing prose."""

        self.publications[record_id] = {
            "record_id": record_id,
            "author": author,
            "tick": tick,
            "grounded_resources": [
                resource.name for resource in sorted(grounded_resources, key=int)
            ],
            "causal_parents": sorted(set(causal_parents)),
        }

    def publication_lineage(self, record_ids: list[str] | set[str]) -> dict[str, Any]:
        """Return the finite citation ancestry behind a set of public records."""

        visited: set[str] = set()
        authors: set[str] = set()
        grounded_authors: set[str] = set()
        resources: set[str] = set()
        max_depth = 0
        stack = [(record_id, 1) for record_id in sorted(set(record_ids))]
        while stack:
            record_id, depth = stack.pop()
            if record_id in visited:
                continue
            publication = self.publications.get(record_id)
            if publication is None:
                continue
            visited.add(record_id)
            authors.add(str(publication["author"]))
            if publication["grounded_resources"]:
                grounded_authors.add(str(publication["author"]))
            resources.update(str(item) for item in publication["grounded_resources"])
            max_depth = max(max_depth, depth)
            stack.extend(
                (str(parent), depth + 1)
                for parent in publication["causal_parents"]
                if parent not in visited
            )
        return {
            "record_ids": sorted(visited),
            "authors": sorted(authors),
            "grounded_authors": sorted(grounded_authors),
            "grounded_resources": sorted(resources),
            "max_depth": max_depth,
        }

    def citation_metrics(self) -> dict[str, int]:
        edges: set[tuple[str, str]] = set()
        cross_agent_edges: set[tuple[str, str]] = set()
        max_depth = 0
        for child_id, publication in self.publications.items():
            max_depth = max(
                max_depth,
                int(self.publication_lineage({child_id})["max_depth"]),
            )
            for parent_id in publication["causal_parents"]:
                parent = self.publications.get(parent_id)
                if parent is None:
                    continue
                edge = (str(parent_id), child_id)
                edges.add(edge)
                if parent["author"] != publication["author"]:
                    cross_agent_edges.add(edge)
        return {
            "publications": len(self.publications),
            "publication_authors": len(
                {str(item["author"]) for item in self.publications.values()}
            ),
            "citation_edges": len(edges),
            "cross_agent_citation_edges": len(cross_agent_edges),
            "max_citation_depth": max_depth,
        }

    def propose(
        self,
        agent: str,
        recipe: dict[str, Any],
        tick: int,
        causal_parents: list[str],
        *,
        combined: bool,
        parent_authors: list[str] | None = None,
        inherited_resources: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_recipe = self._normalize_recipe(recipe)
        evaluator = self._material_evaluator
        inventory = np.zeros(len(Resource), dtype=np.float32)
        for resource, mass in self._required_inputs(normalized_recipe).items():
            inventory[int(resource)] = np.float32(mass)
        evaluated = evaluator.execute(
            normalized_recipe,
            inventory,
            tick=tick,
            contributors=[agent],
            causal_parents=causal_parents,
        )
        inherited = sorted(set(inherited_resources or []))
        inherited_slots = {
            (
                self.scenario.slot("resources", resource)
                if self.scenario is not None
                else resource.upper()
            )
            for resource in inherited
        }
        full_utility = self.material_utility(evaluated)
        ablation_utilities: list[float] = []
        if combined and inherited_slots:
            for removed in sorted(inherited_slots):
                ablated_inputs = [
                    dict(item)
                    for item in normalized_recipe["inputs"]
                    if str(item["resource"]).upper() != removed.upper()
                ]
                if not ablated_inputs:
                    continue
                ablated_recipe = {
                    **normalized_recipe,
                    "inputs": ablated_inputs,
                    "steps": [dict(step) for step in normalized_recipe["steps"]],
                    "design_principles": list(
                        normalized_recipe.get("design_principles", [])
                    ),
                }
                ablated_inventory = np.zeros(len(Resource), dtype=np.float32)
                for resource, mass in self._required_inputs(ablated_recipe).items():
                    ablated_inventory[int(resource)] = np.float32(mass)
                ablated = evaluator.execute(
                    ablated_recipe,
                    ablated_inventory,
                    tick=tick,
                    contributors=[agent],
                    causal_parents=causal_parents,
                )
                ablation_utilities.append(self.material_utility(ablated))
        composition_synergy = (
            round(full_utility - max(ablation_utilities), 6)
            if ablation_utilities
            else None
        )
        proposal = {
            "recipe_id": self._recipe_id(normalized_recipe),
            "agent": agent,
            "tick": tick,
            "recipe": self._public_recipe(normalized_recipe),
            "causal_parents": list(causal_parents),
            "combined": combined,
            "parent_authors": sorted(set(parent_authors or [])),
            "inherited_resources": inherited,
            "grounding_expanded": bool(inherited),
            # Authoritative evaluation is never included in the agent observation.
            "evaluation_utility": round(full_utility, 6),
            # Matched material counterfactual: full recipe minus its strongest
            # leave-one-inherited-material-out ablation under identical processing.
            "composition_synergy": composition_synergy,
        }
        self.proposals.append(proposal)
        return proposal

    def stage_batch(
        self, agent: str, recipe: dict[str, Any], batch: MaterialBatch
    ) -> dict[str, Any]:
        normalized_recipe = self._normalize_recipe(recipe)
        staged = ExperimentalBatch(
            agent=agent,
            recipe_id=self._recipe_id(normalized_recipe),
            source_recipe=self._public_recipe(normalized_recipe),
            batch=batch,
        )
        self.batches.append(staged)
        return {
            "batch_id": batch.batch_id,
            "recipe_id": staged.recipe_id,
            "agent": agent,
            "mass": round(batch.mass, 6),
            "source_recipe": staged.source_recipe,
        }

    def latest_untested_batch(self, agent: str) -> ExperimentalBatch | None:
        for batch in reversed(self.batches):
            if batch.agent == agent and not batch.tested:
                return batch
        return None

    def private_experiment_view(self, agent: str, limit: int = 8) -> dict[str, Any]:
        """Return only experimental state the named scientist has directly measured.

        Pending samples expose their identity and authored recipe, but not latent
        material properties or utility. Those values enter the view only after TEST.
        """

        pending = [
            {
                "batch_id": staged.batch.batch_id,
                "recipe_id": staged.recipe_id,
                "created_tick": staged.batch.tick,
                "mass": round(staged.batch.mass, 6),
                "source_recipe": staged.source_recipe,
                "status": "UNTESTED",
            }
            for staged in reversed(self.batches)
            if staged.agent == agent and not staged.tested
        ][:limit]
        measured = [
            {
                key: value
                for key, value in result.items()
                if key != "passes_material_target"
            }
            for result in reversed(self.tests)
            if result["agent"] == agent
        ][:limit]
        return {
            "pending_microbatches": pending,
            "recent_test_results": measured,
            "test_semantics": "TEST measures the newest pending microbatch first.",
        }

    def test_batch(self, batch: ExperimentalBatch, tick: int) -> dict[str, Any]:
        batch.tested = True
        result: dict[str, Any] = {
            "test_id": f"test_{len(self.tests):08d}",
            "batch_id": batch.batch.batch_id,
            "recipe_id": batch.recipe_id,
            "agent": batch.agent,
            "tick": tick,
            "properties": {
                name: round(value, 6) for name, value in batch.batch.properties.items()
            },
            "material_utility": round(self.material_utility(batch.batch), 6),
            "passes_material_target": (
                self.material_utility(batch.batch) >= self.config.target_material_utility
            ),
        }
        if self.config.experience_reuse:
            # This is the agent-authored input that produced the measured coupon,
            # not evaluator state. Retaining it closes the experience-reuse loop.
            result["source_recipe"] = json.loads(json.dumps(batch.source_recipe))
        self.tests.append(result)
        return result

    def deposit(self, agent: str, resource: Resource, amount: float) -> None:
        self.depot[int(resource)] += np.float32(amount)
        self.depositor_history.add(agent)
        contributions = self.depot_contributions[int(resource)]
        contributions[agent] = contributions.get(agent, 0.0) + amount

    def consume_depot(self, resource: Resource, amount: float) -> list[str]:
        if float(self.depot[int(resource)]) + 1e-7 < amount:
            raise ValueError(f"shared depot lacks {amount:.3f} {resource.name}")
        contributors: list[str] = []
        remaining = amount
        contributions = self.depot_contributions[int(resource)]
        for agent in sorted(contributions):
            available = contributions[agent]
            taken = min(available, remaining)
            if taken > 1e-9:
                contributions[agent] = available - taken
                contributors.append(agent)
                remaining -= taken
            if remaining <= 1e-7:
                break
        self.depot[int(resource)] -= np.float32(amount)
        return contributors

    def available(self, inventory: NDArray[np.float32]) -> NDArray[np.float32]:
        return inventory + self.depot

    def claim(self, agent: str, task: str) -> None:
        self.claims[agent] = task[:160]

    def _milestones(self, artifacts: Any, archive_size: int) -> dict[str, bool]:
        target_indices = list(range(artifacts.count))
        tested_recipes = {test["recipe_id"] for test in self.tests}
        passing_tested_recipes = {
            test["recipe_id"]
            for test in self.tests
            if float(test["material_utility"])
            >= self.config.target_material_utility
        }
        built_from_test = False
        collaborative_build = False
        programmed = False
        performance = False
        causally_composed_artifact = False
        functional_outcome_artifact = False
        emergent_outcome_artifact = False
        invented_identity = False
        behavioral_novelty = False
        for index in target_indices:
            provenance = artifacts.provenance[index]
            recipe = provenance["batch"]["recipe"]
            recipe_id = self._recipe_id(recipe)
            artifact_parents = set(provenance.get("causal_parents", []))
            citation_lineage = self.publication_lineage(artifact_parents)
            is_tested = recipe_id in tested_recipes
            is_causally_composed = (
                bool(artifact_parents)
                and len(set(citation_lineage["grounded_authors"]))
                >= self.config.minimum_contributors
            )
            is_materially_collaborative = (
                len(set(provenance["contributors"]))
                >= self.config.minimum_contributors
            )
            is_collaborative = is_materially_collaborative or is_causally_composed
            spec = provenance.get("artifact_spec", {})
            is_invented = bool(
                spec.get("name")
                and spec.get("claimed_function")
                and spec.get("architecture")
                and spec.get("bio_inspiration")
                and spec.get("predicted_effects")
            )
            is_programmed = any(
                entry.get("program", {}).get("author") != "system"
                for entry in provenance["program_history"]
            )
            has_performance = (
                float(artifacts.lifetime_peak_performance[index])
                >= self.config.target_artifact_performance
            )
            novelty = self.behavioral_novelty(artifacts, index)
            is_novel = novelty >= self.config.target_behavioral_novelty
            is_functional = (
                recipe_id in passing_tested_recipes
                and is_invented
                and is_programmed
                and has_performance
                and is_novel
            )
            built_from_test |= is_tested
            causally_composed_artifact |= is_causally_composed
            collaborative_build |= is_collaborative
            programmed |= is_programmed
            performance |= has_performance
            invented_identity |= is_invented
            behavioral_novelty |= is_novel
            functional_outcome_artifact |= is_functional
            emergent_outcome_artifact |= (
                is_functional and is_causally_composed and is_collaborative
            )
        return {
            "inspect_environment": bool(self.inspections),
            "propose_valid_recipe": bool(self.proposals),
            "fabricate_microbatch": bool(self.batches),
            "test_microbatch": bool(self.tests),
            "publish_evidence": archive_size > 0,
            "pool_resources": (
                len(self.depositor_history) >= self.config.minimum_contributors
            ),
            "combine_independent_evidence": any(
                proposal["combined"]
                and len(set(proposal["causal_parents"])) >= 2
                and len(
                    set(
                        self.publication_lineage(proposal["causal_parents"])[
                            "grounded_authors"
                        ]
                    )
                )
                >= self.config.minimum_contributors
                for proposal in self.proposals
            ),
            "causally_composed_artifact": causally_composed_artifact,
            "build_tested_design": built_from_test,
            "distributed_contribution": collaborative_build,
            "install_artifact_program": programmed,
            "pass_field_performance": performance,
            "invent_artifact_identity": invented_identity,
            "demonstrate_behavioral_novelty": behavioral_novelty,
            "functional_outcome_artifact": functional_outcome_artifact,
            "emergent_outcome_artifact": emergent_outcome_artifact,
        }

    def scorecard(self, artifacts: Any, archive_size: int) -> dict[str, Any]:
        milestones = self._milestones(artifacts, archive_size)
        score = sum(milestones.values()) / len(milestones)
        best_material_utility = max(
            (float(test["material_utility"]) for test in self.tests), default=0.0
        )
        target_performances = [
            float(artifacts.lifetime_peak_performance[index])
            for index in range(artifacts.count)
        ]
        best_artifact_performance = max(target_performances, default=0.0)
        best_current_program_peak_performance = max(
            (float(value) for value in artifacts.peak_performance[: artifacts.count]),
            default=0.0,
        )
        best_final_state_artifact_performance = max(
            (float(value) for value in artifacts.performance[: artifacts.count]),
            default=0.0,
        )
        novelty_values = [
            self.behavioral_novelty(artifacts, index)
            for index in range(artifacts.count)
        ]
        passing_recipe_ids = {
            test["recipe_id"]
            for test in self.tests
            if float(test["material_utility"])
            >= self.config.target_material_utility
        }
        validated_inventions = 0
        for index in range(artifacts.count):
            provenance = artifacts.provenance[index]
            spec = provenance.get("artifact_spec", {})
            recipe_id = self._recipe_id(provenance["batch"]["recipe"])
            if (
                recipe_id in passing_recipe_ids
                and spec.get("name")
                and spec.get("claimed_function")
                and spec.get("architecture")
                and spec.get("bio_inspiration")
                and spec.get("predicted_effects")
                and any(
                    entry.get("program", {}).get("author") != "system"
                    for entry in provenance["program_history"]
                )
                and float(artifacts.lifetime_peak_performance[index])
                >= self.config.target_artifact_performance
                and novelty_values[index] >= self.config.target_behavioral_novelty
            ):
                validated_inventions += 1
        outcome_success = milestones["functional_outcome_artifact"]
        emergent_success = (
            milestones["emergent_outcome_artifact"]
            and milestones["publish_evidence"]
        )
        independent_utilities = [
            float(proposal["evaluation_utility"])
            for proposal in self.proposals
            if not proposal["combined"]
        ]
        combined_utilities = [
            float(proposal["evaluation_utility"])
            for proposal in self.proposals
            if proposal["combined"] and proposal.get("grounding_expanded", False)
        ]
        best_independent = max(independent_utilities, default=0.0)
        best_combined = max(combined_utilities, default=0.0)
        composition_synergies = [
            float(proposal["composition_synergy"])
            for proposal in self.proposals
            if proposal.get("composition_synergy") is not None
        ]
        composition_synergy = max(composition_synergies, default=None)
        citation_metrics = self.citation_metrics()
        portfolio = self.portfolio_metrics(artifacts)
        inventions = []
        for index in range(artifacts.count):
            spec = artifacts.specs[index]
            inventions.append(
                {
                    "artifact_id": f"artifact_{index:08d}",
                    "name": spec.get("name", "Untitled material system"),
                    "architecture": spec.get("architecture", ""),
                    "claimed_function": spec.get("claimed_function", ""),
                    "bio_inspiration": spec.get("bio_inspiration", []),
                    "performance": round(
                        float(artifacts.lifetime_peak_performance[index]), 6
                    ),
                    "current_program_peak_performance": round(
                        float(artifacts.peak_performance[index]), 6
                    ),
                    "final_state_performance": round(
                        float(artifacts.performance[index]), 6
                    ),
                    "behavioral_novelty": round(
                        self.behavioral_novelty(artifacts, index), 6
                    ),
                    "services": {
                        service: round(
                            float(
                                artifacts.lifetime_peak_services[index, service_index]
                            ),
                            6,
                        )
                        for service_index, service in enumerate(SERVICE_NAMES)
                    },
                    "causal_lineage": self.publication_lineage(
                        set(artifacts.provenance[index].get("causal_parents", []))
                    ),
                }
            )
        return {
            "mission": self.config.mission,
            "score": round(score, 6),
            "completed": emergent_success,
            "outcome_success": outcome_success,
            "emergent_success": emergent_success,
            "milestones": milestones,
            "counts": {
                "inspections": len(self.inspections),
                "proposals": len(self.proposals),
                "microbatches": len(self.batches),
                "tests": len(self.tests),
                "claims": len(self.claims),
                "validated_inventions": validated_inventions,
                "combined_designs": sum(
                    1 for proposal in self.proposals if proposal["combined"]
                ),
                "grounding_expanding_combinations": len(combined_utilities),
                "depot_contributors": len(self.depositor_history),
                **citation_metrics,
            },
            "best_material_utility": round(best_material_utility, 6),
            "best_artifact_performance": round(best_artifact_performance, 6),
            "best_lifetime_artifact_performance": round(
                best_artifact_performance, 6
            ),
            "best_current_program_peak_performance": round(
                best_current_program_peak_performance, 6
            ),
            "best_final_state_artifact_performance": round(
                best_final_state_artifact_performance, 6
            ),
            "best_behavioral_novelty": round(max(novelty_values, default=0.0), 6),
            **portfolio,
            "best_independent_recipe_utility": (
                round(best_independent, 6) if independent_utilities else None
            ),
            "best_combined_recipe_utility": (
                round(best_combined, 6) if combined_utilities else None
            ),
            "composition_synergy": composition_synergy,
            # Backward-compatible alias for older renderers and analysis scripts.
            "composition_gain": composition_synergy,
            "inventions": inventions,
            "depot": {
                self._resource_name(index): round(float(amount), 6)
                for index, amount in enumerate(self.depot)
                if index and amount > 1e-7
            },
        }

    def agent_view(
        self, artifacts: Any, archive_size: int, *, evidence_limit: int = 16
    ) -> dict[str, Any]:
        del artifacts, archive_size
        public_evidence = []
        public_profiles: dict[str, dict[str, Any]] = {}
        if self.config.communication:
            for publication in sorted(
                self.publications.values(),
                key=lambda item: (int(item["tick"]), item["record_id"]),
            )[-evidence_limit:]:
                entry = {
                    "record_id": publication["record_id"],
                    "author": publication["author"],
                    "tick": publication["tick"],
                    "grounded_resources": publication["grounded_resources"],
                    "causal_parents": publication["causal_parents"],
                }
                public_evidence.append(entry)
                if not self.config.social_awareness:
                    continue
                author = str(publication["author"])
                profile = public_profiles.setdefault(
                    author,
                    {
                        "task": self.claims.get(author, ""),
                        "publications": 0,
                        "published_grounded_resources": set(),
                    },
                )
                profile["publications"] += 1
                profile["published_grounded_resources"].update(
                    publication["grounded_resources"]
                )
            if self.config.social_awareness:
                for author, task in self.claims.items():
                    public_profiles.setdefault(
                        author,
                        {
                            "task": task,
                            "publications": 0,
                            "published_grounded_resources": set(),
                        },
                    )["task"] = task
                for profile in public_profiles.values():
                    profile["published_grounded_resources"] = sorted(
                        profile["published_grounded_resources"]
                    )
        return {
            "objective": (
                "Collectively invent, validate, build, and program persistent bioinspired "
                "material systems that improve habitat resilience under changing moisture, "
                "solar exposure, contamination, damage, and resource scarcity. No artifact "
                "class or solution is prescribed: explore locally, identify a need, invent "
                "a name and falsifiable function, test recipes, publish evidence, and combine "
                "independent contributions into useful systems."
            ),
            "evaluation_axes": [
                "measured material viability",
                "realized field services under local conditions",
                "behavior distinct from prior artifacts",
                "traceable use of independent empirical evidence",
            ],
            "shared_depot": {
                self._resource_name(index): round(float(amount), 6)
                for index, amount in enumerate(self.depot)
                if self.config.shared_depot and index and amount > 1e-7
            },
            "task_claims": (
                dict(self.claims)
                if self.config.communication and self.config.social_awareness
                else {}
            ),
            "public_evidence_catalog": public_evidence,
            "public_agent_profiles": dict(sorted(public_profiles.items())),
            "available_channels": {
                "action_feedback": self.config.action_feedback,
                "immediate_replanning": self.config.immediate_replanning,
                "communication": self.config.communication,
                "social_awareness": self.config.social_awareness,
                "shared_depot": self.config.shared_depot,
                "public_infrastructure_map": self.config.public_infrastructure_map,
                "global_landmarks": self.config.global_landmarks,
            },
        }

    def authoritative_state(self) -> dict[str, Any]:
        return {
            "depot": self.depot.tolist(),
            "depot_contributions": self.depot_contributions,
            "depositor_history": sorted(self.depositor_history),
            "inspections": self.inspections,
            "proposals": self.proposals,
            "batches": [
                {
                    "agent": batch.agent,
                    "recipe_id": batch.recipe_id,
                    "source_recipe": batch.source_recipe,
                    "batch": batch.batch.as_dict(),
                    "tested": batch.tested,
                }
                for batch in self.batches
            ],
            "tests": self.tests,
            "claims": self.claims,
            "publications": [
                self.publications[record_id] for record_id in sorted(self.publications)
            ],
        }
