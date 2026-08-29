import json
from dataclasses import replace
from pathlib import Path

from biofoundry.cli import RESEARCH_CONDITIONS, RESEARCH_LLM_CONDITIONS
from biofoundry.config import GameConfig, load_config
from biofoundry.memory import AgentMemory, MemoryRecord
from biofoundry.program_library import (
    MAX_RETAINED_SKILL_EVIDENCE,
    MAX_RETAINED_SKILL_MEASUREMENTS,
    ProgramLibrary,
)
from biofoundry.programs import ArtifactProgram
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction

ROOT = Path(__file__).resolve().parents[1]


def test_v5_learning_mechanisms_remain_explicit_ablation_switches() -> None:
    baseline = GameConfig()
    assert baseline.llm.experience_attention is False
    assert baseline.llm.freeze_on_provider_outage is False
    assert baseline.science.experience_reuse is False
    assert baseline.science.request_tracking is False
    assert baseline.science.message_interrupts is True
    assert baseline.science.selective_replanning is False
    assert baseline.science.decision_schedule == "event-driven"

    treatment = load_config(ROOT / "configs" / "openai-gpt-5.6-luna.yaml")
    assert treatment.llm.experience_attention is True
    assert treatment.llm.freeze_on_provider_outage is True
    assert treatment.science.experience_reuse is True
    assert treatment.science.request_tracking is True
    assert treatment.science.message_interrupts is False
    assert treatment.science.selective_replanning is True
    assert treatment.trace.compression == "gzip"

    assert RESEARCH_LLM_CONDITIONS["legacy-transcript"]["experience_attention"] is False
    assert RESEARCH_LLM_CONDITIONS["bounded-context-only"]["retrieval_feedback_weight"] == 0.0
    assert RESEARCH_CONDITIONS["no-experience-reuse"] == {"experience_reuse": False}


def test_retrieval_credit_changes_future_selection_without_domain_rules() -> None:
    memory = AgentMemory(16)
    for record_id in ("a", "b"):
        memory.remember(
            MemoryRecord(record_id, 0, "observation", "interface response", 0.5),
            notebook=True,
        )

    first = memory.retrieve({"interface"}, limit=1, adaptive=True)
    assert first[0].record_id == "a"
    memory.mark_retrieval_use({"a"})
    memory.mark_causal_outcome({"a"}, success=True)
    second = memory.retrieve({"interface"}, limit=1, adaptive=True)

    assert second[0].record_id == "a"
    diagnostic = memory.last_retrieval["selected"][0]
    assert diagnostic["feedback_rate"] == 1.0
    assert diagnostic["citation_rate"] == 1.0
    assert diagnostic["causal_success_rate"] == 1.0
    assert diagnostic["adaptive_bonus"] > 0.0


def test_executed_action_outcome_updates_cited_experience_credit() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    evidence = MemoryRecord(
        "observation_credit", 0, "observation", "measured local interface", 0.8
    )
    simulation.memories[0].remember(evidence, notebook=True)
    simulation.memories[0].retrieve({"interface"}, limit=1, adaptive=True)
    simulation.memories[0].mark_retrieval_use({evidence.record_id})

    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.INSPECT,
                causal_parents=[evidence.record_id],
            )
        }
    )

    stats = simulation.memories[0].retrieval_stats[evidence.record_id]
    assert stats["outcome_attempts"] == 1
    assert stats["successful_outcomes"] == 1


def test_verified_skill_experience_has_bounded_context_but_exact_program_identity() -> None:
    agent = "agent_000000"
    library = ProgramLibrary([agent])
    program = ArtifactProgram.from_dict(
        {"name": "self-authored", "instructions": [{"op": "collect_water", "value": 0.01}]},
        author=agent,
    )
    library.register(program, tick=0, artifact_id="artifact_00000000")
    for tick in range(40):
        library.observe(
            agent,
            program.program_id,
            tick=tick,
            source="measured_artifact",
            evidence_id=f"evidence_{tick}",
            verified=True,
            measurements={"performance": tick / 100},
        )

    skill = library.skills[agent][program.program_id]
    assert len(skill["evidence_ids"]) == MAX_RETAINED_SKILL_EVIDENCE
    assert len(skill["measurements"]) == MAX_RETAINED_SKILL_MEASUREMENTS
    compact = library.view(agent, compact=True)[0]
    assert compact["program_id"] == program.program_id
    assert compact["evidence_count"] == 41
    assert compact["measurement_count"] == 40
    assert len(compact["recent_evidence_ids"]) == 4
    assert len(compact["recent_measurements"]) == 3


def test_arbitrary_successful_action_can_fulfill_an_agent_authored_request() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    simulation = BioFoundrySimulation(config)
    simulation.population.x[:2] = 10
    simulation.population.y[:2] = 10
    sender, recipient = simulation.agent_ids[:2]

    delivered = simulation.step(
        {
            sender: AgentAction(
                verb=ActionType.COMMUNICATE,
                target_agent_id=recipient,
                message="Please inspect this location and report what is measured.",
            )
        }
    )
    message_id = next(
        str(event.payload["record_id"])
        for event in delivered.events
        if event.kind == "message_delivered"
    )
    inbox = json.loads(simulation.semantic_observation(1))["pending_requests"]
    assert inbox[0]["message_id"] == message_id

    result = simulation.step(
        {
            recipient: AgentAction(
                verb=ActionType.INSPECT,
                reply_to=message_id,
                causal_parents=[message_id],
            )
        }
    )

    fulfillment = next(
        event for event in result.events if event.kind == "request_fulfilled"
    )
    assert fulfillment.payload["verb"] == "INSPECT"
    observation_id = next(
        str(event.payload["record_id"])
        for event in result.events
        if event.kind == "sample_inspected"
    )
    assert observation_id in fulfillment.payload["result_ids"]
    assert simulation.memories[0].get(observation_id) is not None
    assert simulation.memories[0].latest("request_fulfillment") is not None
    assert simulation.message_records[message_id]["fulfillments"][0]["agent"] == recipient
    assert json.loads(simulation.semantic_observation(1))["pending_requests"] == []


def test_message_delivery_is_observable_without_forcing_plan_replacement() -> None:
    base = load_config(ROOT / "configs" / "demo.yaml")
    config = replace(
        base,
        science=replace(base.science, message_interrupts=False),
    )
    simulation = BioFoundrySimulation(config)
    simulation.population.x[:2] = 10
    simulation.population.y[:2] = 10
    simulation._observe_visible_cells(0)
    simulation._observe_visible_cells(1)
    simulation.needs_replan[:] = False

    simulation.step(
        {
            simulation.agent_ids[0]: AgentAction(
                verb=ActionType.COMMUNICATE,
                target_agent_id=simulation.agent_ids[1],
                message="Evidence is available when your current experiment completes.",
            )
        }
    )

    assert not simulation.needs_replan[1]
    assert simulation.memories[1].latest("message") is not None
