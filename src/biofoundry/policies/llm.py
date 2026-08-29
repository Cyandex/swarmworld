"""Sparse macroturn LLM policy with strict structured-action parsing."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections import deque
from typing import Any

from tqdm import tqdm

from ..capabilities import (
    addressing_available,
    available_action_types,
    replies_available,
)
from ..memory import MemoryRecord
from ..providers.base import ChatMessage, LLMProvider
from ..providers.openai_compatible import ProviderError
from ..simulation import BioFoundrySimulation
from ..structured_output import (
    MAX_PLAN_ACTIONS,
    coerce_transport_numbers,
    validate_action_plan,
)
from ..types import ActionType, AgentAction

SYSTEM_PROMPT = """You are one homogeneous research agent in BioFoundry World.
You have only local observations, bounded memory, and your own scientific knowledge.
Collectively invent interacting bioinspired material systems for habitat resilience. No artifact
catalog, biological analogy, or final design is supplied: identify needs and originate
candidate mechanisms yourself. Do not invent observations or claim unmeasured effects.

Return exactly one JSON object containing a private `research_state` update and a
`plan` list of 1 to
{max_plan_actions} actions that will execute over later microticks. A later macroturn
can continue the work, so include only actions that remain valid without new evidence.
The research state is your own notebook: preserve or revise your self-chosen goal,
hypothesis, progress, next observable checkpoint, and collaboration need. Its text is
not rewarded and does not change physics. Preserve up to sixteen stable evidence or
program IDs in `evidence_ids`; these pointers let general retrieval learn which prior
experience actually supports later actions.
In recipes, list each material input once with its total mass and give one to four
distinct design principles as short phrases; never repeat steps or text merely to fill
an array. Every action must contain every legal key:
verb, direction, resource, artifact, target_x, target_y, target_artifact_id,
target_agent_id, reply_to, amount, message, insight, recipe, program, artifact_spec,
causal_parents. Use
direction "STAY", resource/artifact "NONE", amount 0, target coordinates -1, empty
target IDs, reply_to, and message strings, null recipe/program/insight/artifact_spec,
and an empty causal_parents list when a field is irrelevant. Enum names are supplied.
Programs use the bounded straight-line Artifact Behavior DSL; never return Python,
shell code, imports, network calls, or loops.

The response schema is representational vocabulary, not evidence that a material,
station, or affordance exists in this world. Base executable designs on your own
empirical_knowledge. Unlisted materials remain uncertain rather than proven absent;
form and revise your own beliefs from observations and action outcomes.

You are not assigned a role. If a public task board is visible, infer an unmet need,
claim a useful task, and revise it when other agents' work or feedback warrants.
Prefer actions that advance the complete scientific cycle over repetitive harvesting.
"""

SCENARIO_SYSTEM_PROMPT = """You are one research agent in the declarative scenario
{scenario_name}. You have local observations, bounded memory, and the scenario catalog
embedded in each observation. Ground claims in observations, measured tests, or cited
records. The scenario defines the resources, operations, facilities, environmental
fields, services, and mission; do not import vocabulary from another world.

Return exactly one JSON object containing a private `research_state` and a `plan` of
1 to {max_plan_actions} actions. Every action uses the complete neutral-field shape
required by the supplied schema. Programs are bounded straight-line DSL programs;
never return Python, shell code, imports, network calls, or loops.

{scenario_instructions}
"""

SCENARIO_ACTION_MANUAL = """Scenario action mechanics:
- MOVE changes one cardinal tile; explore when a landmark is not locally known.
- INSPECT records local terrain, fields, matter, facilities, and nearby artifacts.
- HARVEST collects local matter and always uses resource=\"NONE\".
- DEPOSIT and OPERATE require a declared workspace or facility.
- A recipe uses only scenario resource and operation IDs from the active schema.
- OPERATE creates a measured microbatch; TEST reveals its deterministic properties.
- Publications and causal parents must refer to real recorded evidence IDs.
- BUILD consumes a full recipe and authors an original system specification.
- Claims, names, prompts, and generated imagery do not change physics.
"""

ACTION_MANUAL = """Grounded action mechanics:
- MOVE changes one cardinal tile. Resources, stations, agents, and artifacts are only
  reported when locally discovered. If a needed landmark is absent, explore rather
  than assuming its coordinates; INSPECT, memory, and published records preserve clues.
- `local_affordances` reports current physical possibilities authoritatively. It is not
  a recommendation and does not choose a task, recipe, hypothesis, or destination.
- `public_infrastructure` is the society's factual base map of fixed laboratories.
  Biological resources and environmental evidence are deliberately absent from it and
  still require local exploration.
- INSPECT creates a private, citable record of site conditions, nutrients, resource
  presence, and services of adjacent artifacts; it does not reveal latent processed-
  material properties. Only TEST of a microbatch made by OPERATE reveals material
  properties and utility.
- HARVEST samples at most 0.30 mass of whatever collectable material is physically
  present on the current tile. Set resource="NONE"; HARVEST never requests named matter.
- DEPOSIT transfers `amount` of `resource` to the shared depot and requires a foundry
  or station tile. BUILD and OPERATE may draw from both personal inventory and depot.
- PROPOSE_RECIPE validates and records a complete recipe but consumes no material.
- OPERATE requires a complete recipe and a foundry/station tile; it fabricates a
  reduced-scale microbatch and consumes feedstocks. TEST measures your latest untested
  microbatch and reveals objective material properties and utility.
- PUBLISH requires a structured insight and creates a public record ID. Other agents
  can cite that ID in `causal_parents`. A publication can ground a material for later
  combination only when its text explicitly names matter its author had personally
  observed or possessed; unsupported material names confer no capability.
- DEPOSIT_INSIGHT leaves a spatial record that nearby agents can read without adding it
  to the global archive; it remains available when public communication is disabled.
- COMBINE_DESIGN requires a complete recipe and at least two distinct published record
  IDs in `causal_parents`. Unlike PROPOSE_RECIPE, it may use recipe inputs grounded by
  those cited publications, allowing independently observed materials to be composed.
- BUILD consumes the full recipe. In science mode it also requires `artifact_spec`:
  {name, claimed_function, architecture, bio_inspiration[1..8],
  predicted_effects[1..8], geometry}. Geometry has integer layers 1..16 and continuous
  surface_area 0.25..4, channel_density, anisotropy, branching, connectivity, curvature,
  and modularity, each 0..1. Invent the architecture and values from your own reasoning.
  There is no artifact catalog: artifact="MATERIAL_SYSTEM" is an unconstrained system.
  Names and prose claims are recorded but do not change physics.
- WRITE_PROGRAM creates an original bounded DSL program. FORK_PROGRAM requires a known
  parent program_id and changed instructions, producing a descent-with-modification edge.
- For WRITE_PROGRAM/FORK_PROGRAM, REPAIR, or DISMANTLE, copy the stable `id` from
  `nearby_artifacts` into target_artifact_id. Coordinates alone are ambiguous when
  several artifacts share a tile.
- CLAIM_TASK writes `message` to the public task board. The archive view is a bounded
  recent window; TEACH can privately transfer selected records from your own retained
  memory, including observations and test results, or a message into nearby agents'
  durable notebooks. Set target_agent_id to address COMMUNICATE, TEACH, or TRADE and
  reply_to to the message record being answered. When `pending_requests` is present,
  any successful non-WAIT action may set reply_to to its message_id, creating an
  explicit request-to-action causal edge without requiring a prose response. TRADE
  transfers `amount` to an adjacent addressed agent.
- METABOLIZE exists only in the optional metabolism treatment. It consumes `amount` of
  a named biological resource from personal inventory and restores bounded energy.
- REPAIR/DISMANTLE require a nearby artifact. A failed action is reported next turn and
  triggers immediate replanning. Material utility and field-service equations are
  hidden: learn by comparing controlled recipes, tests, and observed artifact services.
  Program sensors include local nutrients. Available actuators are collect_water,
  grow, heal, set_open,
  reduce_contamination, and emit_signal. Their realized effects are gated by measured
  material properties, geometry, local conditions, health, and resource limits.
- `recent_behavior_summary` is a factual bounded-window count of your own attempts, not
  an evaluator judgment. Use your research state to decide whether repetition remains
  informative or whether your self-chosen checkpoint warrants a new approach.
"""


def capability_action_manual(allowed: tuple[ActionType, ...]) -> str:
    """Remove mechanics for verbs unavailable in an experimental treatment."""

    if set(allowed) == set(ActionType):
        return ACTION_MANUAL
    names = {action.name for action in allowed}
    known = {action.name for action in ActionType}
    segments = ACTION_MANUAL.split("\n- ")
    retained = [segments[0]]
    for segment in segments[1:]:
        referenced = {
            token for token in re.findall(r"\b[A-Z][A-Z_]+\b", segment) if token in known
        }
        if not referenced or referenced <= names:
            retained.append(segment)
    return retained[0] + "".join(f"\n- {segment}" for segment in retained[1:])


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError("model response contains no JSON object") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _shrink_observation(packet: dict[str, Any]) -> bool:
    """Remove the lowest-priority final item while preserving causal core state."""

    list_paths: tuple[tuple[tuple[str, ...], int], ...] = (
        (("verified_skill_library",), 1),
        (("nearby_artifacts",), 2),
        (("public_archive",), 1),
        (("nearby_insights",), 1),
        (("private_experiments", "recent_test_results"), 1),
        (("research_mission", "public_evidence_catalog"), 4),
        (("recent_action_results",), 4),
        (("nearby_artifact_catalog",), 8),
    )
    for path, minimum in list_paths:
        value: Any = packet
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, list) and len(value) > minimum:
            value.pop()
            return True
    mission = packet.get("research_mission")
    if isinstance(mission, dict):
        for name in ("public_agent_profiles", "task_claims"):
            mapping = mission.get(name)
            if isinstance(mapping, dict) and len(mapping) > 4:
                mapping.pop(sorted(mapping)[-1])
                return True
    for key in (
        "nearby_insights",
        "public_archive",
        "nearby_artifact_catalog",
        "verified_skill_library",
    ):
        value = packet.get(key)
        if value:
            packet[key] = []
            return True
    return False


class LLMPolicy:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.action_queues: dict[str, deque[AgentAction]] = {}
        self.last_traces: list[dict[str, Any]] = []
        self.calls_made = 0
        self.provider_attempts = 0
        self.provider_outage = False

    async def refresh_plans(
        self,
        simulation: BioFoundrySimulation,
        show_progress: bool = True,
    ) -> bool:
        """Refresh scheduled plans, returning false on a total provider outage."""

        scheduled_all = simulation.scheduled_macro_agents()
        call_budget = getattr(getattr(self.provider, "config", None), "call_budget", None)
        remaining = (
            len(scheduled_all)
            if call_budget is None
            else max(0, int(call_budget) - self.calls_made)
        )
        scheduled = scheduled_all[:remaining]
        freeze_on_outage = bool(
            getattr(
                getattr(self.provider, "config", None),
                "freeze_on_provider_outage",
                False,
            )
        )
        memory_snapshots = (
            {
                agent_id: copy.deepcopy(
                    simulation.memories[
                        simulation.population.id_to_index[agent_id]
                    ]
                )
                for agent_id in scheduled
            }
            if freeze_on_outage
            else {}
        )
        jobs = [self._decide(simulation, agent_id) for agent_id in scheduled]
        decided: dict[str, list[AgentAction]] = {}
        traces: dict[str, dict[str, Any]] = {}
        iterator = asyncio.as_completed(jobs)
        if show_progress and jobs:
            iterator = tqdm(
                iterator,
                total=len(jobs),
                desc=f"LLM macroturn {simulation.tick}",
                leave=False,
            )
        for future in iterator:
            agent_id, plan, trace = await future
            decided[agent_id] = plan
            traces[agent_id] = trace
        self.provider_attempts += len(scheduled)
        self.last_traces = [traces[agent_id] for agent_id in sorted(traces)]
        transient_provider_failure = bool(scheduled) and any(
            trace.get("error_kind") == "provider_transient"
            for trace in traces.values()
        )
        self.provider_outage = bool(transient_provider_failure and freeze_on_outage)
        if self.provider_outage:
            for agent_id, trace in traces.items():
                index = simulation.population.id_to_index[agent_id]
                simulation.memories[index] = memory_snapshots[agent_id]
                trace["queued_actions_replaced"] = 0
                trace["queue_preserved_during_outage"] = len(
                    self.action_queues.get(agent_id, ())
                )
                trace["planning_committed"] = False
            return False
        self.calls_made += len(scheduled)
        for agent_id, plan in decided.items():
            trace = traces[agent_id]
            trace["queued_actions_replaced"] = len(
                self.action_queues.get(agent_id, ())
            )
            trace["planning_committed"] = True
            self.action_queues[agent_id] = deque(plan)
        # Once the budget is exhausted, acknowledge pending replans so they do not
        # remain spuriously scheduled on every later microtick. Existing queues still
        # execute to completion.
        simulation.acknowledge_macroturn(scheduled_all)
        return True

    def next_actions(self, simulation: BioFoundrySimulation) -> dict[str, AgentAction]:
        """Consume one queued motor action only when the world will advance."""

        actions: dict[str, AgentAction] = {}
        for agent_id in simulation.agent_ids:
            queue = self.action_queues.setdefault(agent_id, deque())
            actions[agent_id] = queue.popleft() if queue else AgentAction()
        return actions

    async def actions(
        self,
        simulation: BioFoundrySimulation,
        show_progress: bool = True,
    ) -> dict[str, AgentAction]:
        await self.refresh_plans(simulation, show_progress=show_progress)
        return self.next_actions(simulation)

    async def _decide(
        self, simulation: BioFoundrySimulation, agent_id: str
    ) -> tuple[str, list[AgentAction], dict[str, Any]]:
        index = simulation.population.id_to_index[agent_id]
        max_plan_actions = int(
            getattr(
                getattr(self.provider, "config", None),
                "max_plan_actions",
                MAX_PLAN_ACTIONS,
            )
        )
        provider_config = getattr(self.provider, "config", None)
        allowed_actions = available_action_types(simulation.config)
        allow_addressing = addressing_available(simulation.config)
        allow_replies = replies_available(simulation.config)
        action_manual = capability_action_manual(allowed_actions)
        if simulation.scenario is not None:
            action_manual = SCENARIO_ACTION_MANUAL
        experience_attention = bool(
            getattr(provider_config, "experience_attention", False)
        )
        prior_state_record = simulation.memories[index].latest("research_state")
        prior_state = (
            prior_state_record.content if prior_state_record is not None else "none yet"
        )
        observation = simulation.semantic_observation(
            index,
            compact=experience_attention,
            focus_text=prior_state,
            artifact_detail_limit=int(
                getattr(provider_config, "nearby_artifact_detail_limit", 6)
            ),
            archive_limit=int(
                getattr(provider_config, "public_archive_retrieval_limit", 4)
            ),
        )
        observation_packet = json.loads(observation)
        query_text = " ".join(
            (
                prior_state,
                json.dumps(observation_packet.get("pending_requests", []), sort_keys=True),
                json.dumps(observation_packet.get("local", {}), sort_keys=True),
            )
        )
        memories = simulation.memories[index].retrieve(
            set(simulation.memories[index].tokenize(query_text)),
            limit=(
                int(getattr(provider_config, "memory_retrieval_limit", 4))
                if experience_attention
                else 6
            ),
            adaptive=experience_attention,
            feedback_weight=float(
                getattr(provider_config, "retrieval_feedback_weight", 0.45)
            ),
            exploration_weight=float(
                getattr(provider_config, "retrieval_exploration_weight", 0.12)
            ),
        )
        memory_record_characters = int(
            getattr(provider_config, "memory_record_characters", 2_000)
        )
        memory_entries = [
            (
                f"- [{record.record_id}; kind={record.kind}; tick={record.tick}] "
                f"{simulation.memories[index].context_excerpt(record, memory_record_characters)}"
            )
            for record in memories
        ]
        grounded = sorted(simulation.grounded_resources(index), key=int)
        grounded_ids = (
            ", ".join(simulation.resource_name(resource) for resource in grounded)
            if grounded
            else "none yet"
        )
        tail = (
            f"{action_manual}\n"
            f"Empirically grounded material IDs for this agent: {grounded_ids}. "
            "PROPOSE_RECIPE inputs must use only these material names. COMBINE_DESIGN "
            "may additionally use materials explicitly grounded in its cited publications. "
            "Named resources are used only for matter already held or available in the "
            "shared depot during DEPOSIT or TRADE; HARVEST always uses resource=\"NONE\".\n"
            f"Executable verb names in this treatment: "
            f"{', '.join(action.name for action in allowed_actions)}. "
            "The response schema contains exactly these verbs. Direction names: "
            "STAY, NORTH, EAST, SOUTH, WEST. "
            "Resource names are those listed as empirically grounded; use NONE when "
            "irrelevant. Artifact names: NONE, MATERIAL_SYSTEM; architecture is authored "
            "in artifact_spec. Return exactly "
            "{\"research_state\":{\"goal\":\"...\",\"hypothesis\":\"...\","
            "\"progress\":\"...\",\"next_checkpoint\":\"...\","
            "\"collaboration_need\":\"...\",\"evidence_ids\":[]},"
            f'\"plan\":[...]}} with 1-{max_plan_actions} actions.'
        )
        system_prompt = (
            SCENARIO_SYSTEM_PROMPT.format(
                scenario_name=simulation.scenario.name,
                max_plan_actions=max_plan_actions,
                scenario_instructions=simulation.scenario.agent_prompt,
            )
            if simulation.scenario is not None
            else SYSTEM_PROMPT.format(max_plan_actions=max_plan_actions)
        )

        def render_prompt() -> str:
            rendered_observation = json.dumps(
                observation_packet, sort_keys=True, separators=(",", ":")
            )
            memory_text = "\n".join(memory_entries) or "- none"
            return (
                f"Agent: {agent_id}\nObservation: {rendered_observation}\n"
                f"Relevant memory:\n{memory_text}\n"
                f"Prior self-authored research state: {prior_state}\n{tail}"
            )

        prompt = render_prompt()
        context_budget = int(
            getattr(provider_config, "context_budget_characters", 64_000)
        )
        original_prompt_characters = len(system_prompt) + len(prompt)
        if experience_attention:
            while len(system_prompt) + len(prompt) > context_budget:
                if memory_entries:
                    memory_entries.pop()
                elif not _shrink_observation(observation_packet):
                    break
                prompt = render_prompt()
        context_diagnostics = {
            "experience_attention": experience_attention,
            "budget_characters": context_budget,
            "original_prompt_characters": original_prompt_characters,
            "final_prompt_characters": len(system_prompt) + len(prompt),
            "memories_included": len(memory_entries),
            "artifact_details_included": len(
                observation_packet.get("nearby_artifacts", [])
            ),
            "skill_records_included": len(
                observation_packet.get("verified_skill_library", [])
            ),
            "archive_records_included": len(
                observation_packet.get("public_archive", [])
            ),
        }
        response = ""
        error = ""
        error_kind = ""
        usage: dict[str, Any] = {}
        provider_metadata: dict[str, Any] = {}
        try:
            messages = [
                ChatMessage(
                    "system",
                    system_prompt,
                ),
                ChatMessage("user", prompt),
            ]
            generate_record = getattr(self.provider, "generate_record", None)
            if callable(generate_record):
                generation = await generate_record(messages)
                response = generation.text
                usage = dict(generation.usage)
                provider_metadata = dict(getattr(generation, "metadata", {}))
            else:
                response = await self.provider.generate(messages)
            decoded = validate_action_plan(
                coerce_transport_numbers(_extract_json(response)),
                allowed_action_types=allowed_actions,
                allow_addressing=allow_addressing,
                allow_replies=allow_replies,
                resource_names=(
                    simulation.scenario.resource_names
                    if simulation.scenario is not None
                    else None
                ),
                operation_names=(
                    simulation.scenario.operation_names
                    if simulation.scenario is not None
                    else None
                ),
            )
            research_state = dict(decoded["research_state"])
            research_state.setdefault("evidence_ids", [])
            raw_plan = decoded["plan"]
            if simulation.scenario is not None:
                raw_plan = [
                    simulation.scenario.normalize_action_mapping(item)
                    for item in raw_plan
                ]
            plan = [AgentAction.from_value(item) for item in raw_plan]
            retained_evidence = {
                str(record_id)
                for record_id in research_state.get("evidence_ids", [])
                if simulation.memories[index].get(str(record_id)) is not None
                or str(record_id) in simulation.archive.by_id
                or simulation.program_library.knows(agent_id, str(record_id))
            }
            research_state["evidence_ids"] = sorted(retained_evidence)
            cited = retained_evidence | {
                parent for action in plan for parent in action.causal_parents
            }
            simulation.memories[index].mark_retrieval_use(cited)
            simulation.memories[index].remember(
                MemoryRecord(
                    record_id=f"research_state_{agent_id}_{simulation.tick:08d}",
                    tick=simulation.tick,
                    kind="research_state",
                    content=json.dumps(research_state, sort_keys=True),
                    salience=0.95,
                    causal_parents=tuple(sorted(retained_evidence)),
                ),
                notebook=True,
            )
        except ProviderError as exc:
            error = str(exc)[:500]
            error_kind = (
                "provider_transient" if exc.retryable else "provider_terminal"
            )
            plan = [AgentAction()]
            research_state = None
        except Exception as exc:
            error = str(exc)[:500]
            error_kind = "invalid_output"
            plan = [AgentAction()]
            research_state = None
        trace = {
            "tick": simulation.tick,
            "agent": agent_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response": response,
            "error": error,
            "error_kind": error_kind,
            "usage": usage,
            "provider": provider_metadata,
            "capabilities": {
                "actions": [action.name for action in allowed_actions],
                "addressing": allow_addressing,
                "replies": allow_replies,
            },
            "prompt_characters": len(system_prompt) + len(prompt),
            "response_characters": len(response),
            "actions": [action.as_dict() for action in plan],
            "research_state": research_state,
            "context": context_diagnostics,
            "retrieval": (
                simulation.memories[index].last_retrieval
                if simulation.config.science.retrieval_diagnostics
                else {}
            ),
            "_trace_templates": {
                "system_prompt": system_prompt,
                "action_manual": action_manual,
            },
        }
        return agent_id, plan, trace
