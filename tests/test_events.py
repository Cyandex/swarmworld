from pathlib import Path

import pytest

from biofoundry import ENGINE_REVISION
from biofoundry.config import load_config
from biofoundry.counterfactuals import (
    apply_recorded_model_trace,
    load_action_trace,
    replay_intervention,
)
from biofoundry.events import EventRecorder, digest_snapshot, read_records
from biofoundry.simulation import BioFoundrySimulation

ROOT = Path(__file__).resolve().parents[1]


def test_event_log_snapshot_digest_roundtrip(tmp_path: Path) -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    output = tmp_path / "episode.jsonl"
    with EventRecorder(output, {"test": True}) as recorder:
        digest = recorder.write_snapshot(simulation.snapshot(display_limit=None))
        result = simulation.step({})
        recorder.write_events(result.events)
        final = recorder.write_snapshot(simulation.snapshot(display_limit=None))
    records = list(read_records(output))
    assert records[1]["digest"] == digest
    assert records[-1]["digest"] == final


def test_action_trace_reproduces_authoritative_state(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    original = BioFoundrySimulation(config)
    output = tmp_path / "trace.jsonl"
    actions = {original.agent_ids[0]: {"verb": 1, "direction": 2}}
    with EventRecorder(output, {"config": config.as_dict()}) as recorder:
        recorder.write_snapshot(
            original.snapshot(display_limit=None),
            state_digest=original.state_digest(),
        )
        recorder.write_record({"type": "actions", "tick": 0, "actions": actions})
        original.step(actions)
        recorder.write_snapshot(
            original.snapshot(display_limit=None),
            state_digest=original.state_digest(),
        )

    replayed = BioFoundrySimulation(config)
    records = list(read_records(output))
    assert records[1]["digest"] == digest_snapshot(records[1]["snapshot"])
    replayed.step(records[2]["actions"])
    assert replayed.state_digest() == records[3]["state_digest"]


def test_counterfactual_trace_requires_matching_engine_revision(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    compatible = tmp_path / "compatible.jsonl"
    with EventRecorder(
        compatible,
        {"config": config.as_dict(), "engine_revision": ENGINE_REVISION},
    ) as recorder:
        recorder.write_record({"type": "actions", "tick": 0, "actions": {}})
    loaded_config, actions = load_action_trace(compatible)
    assert loaded_config == config.as_dict()
    assert len(actions) == 1

    legacy = tmp_path / "legacy.jsonl"
    with EventRecorder(legacy, {"config": config.as_dict()}) as recorder:
        recorder.write_record({"type": "actions", "tick": 0, "actions": {}})
    with pytest.raises(ValueError, match="engine-revision marker"):
        load_action_trace(legacy)


def test_llm_trace_replays_policy_side_macroturn_acknowledgements() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    original = BioFoundrySimulation(config)
    action_records = []
    for _ in range(4):
        macroturn_agents = original.scheduled_macro_agents()
        original.acknowledge_macroturn(macroturn_agents)
        record = {
            "type": "actions",
            "tick": original.tick,
            "macroturn_agents": macroturn_agents,
            "actions": {},
        }
        action_records.append(record)
        original.step({})

    replayed = replay_intervention(
        config.as_dict(), action_records, show_progress=False
    )

    assert replayed.state_digest() == original.state_digest()


def test_llm_trace_replays_private_research_state_memory() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    original = BioFoundrySimulation(config)
    agent_id = original.agent_ids[0]
    trace = {
        "type": "model_trace",
        "tick": 0,
        "agent": agent_id,
        "research_state": {
            "goal": "measure fungal matter",
            "hypothesis": "processing changes permeability",
            "progress": "baseline pending",
            "next_checkpoint": "fabricate a microbatch",
            "collaboration_need": "none",
            "evidence_ids": ["observation_a"],
        },
    }
    apply_recorded_model_trace(original, trace)
    macroturn_agents = original.scheduled_macro_agents()
    original.acknowledge_macroturn(macroturn_agents)
    original.step({})

    replayed = replay_intervention(
        config.as_dict(),
        [
            {
                "type": "actions",
                "tick": 0,
                "macroturn_agents": macroturn_agents,
                "model_traces": [trace],
                "actions": {},
            }
        ],
        show_progress=False,
    )
    assert replayed.state_digest() == original.state_digest()


def test_uncommitted_outage_trace_does_not_mutate_replayed_memory() -> None:
    simulation = BioFoundrySimulation(load_config(ROOT / "configs" / "demo.yaml"))
    apply_recorded_model_trace(
        simulation,
        {
            "tick": 0,
            "agent": simulation.agent_ids[0],
            "planning_committed": False,
            "research_state": {
                "goal": "this proposal was rolled back",
                "evidence_ids": [],
            },
        },
    )
    assert simulation.memories[0].latest("research_state") is None


def test_legacy_llm_trace_recomputes_deterministic_macroturn_schedule() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    original = BioFoundrySimulation(config)
    action_records = []
    for _ in range(4):
        original.acknowledge_macroturn(original.scheduled_macro_agents())
        action_records.append(
            {
                "type": "actions",
                "tick": original.tick,
                "recorded_policy": "llm",
                "actions": {},
            }
        )
        original.step({})

    replayed = replay_intervention(
        config.as_dict(), action_records, show_progress=False
    )

    assert replayed.state_digest() == original.state_digest()
