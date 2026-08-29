import json

from biofoundry.trace_analysis import analyze_trace


def test_trace_report_separates_plans_execution_and_rejections(tmp_path) -> None:
    records = [
        {"type": "header", "metadata": {"config": {"simulation": {"seed": 7}}}},
        {
            "type": "model_trace",
            "tick": 0,
            "agent": "agent_000000",
            "actions": [{"verb": 2}, {"verb": 1}],
            "queued_actions_replaced": 3,
            "error": "",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "retrieval": {
                "selected": [{"record_id": "memory_a", "score": 1.0}],
                "used_as_causal_parent": ["memory_a"],
            },
            "research_state": {
                "goal": "test a wet interface",
                "hypothesis": "porosity improves capture",
                "progress": "none",
                "next_checkpoint": "fabricate one batch",
                "collaboration_need": "find a second material",
            },
        },
        {
            "type": "model_trace",
            "tick": 1,
            "agent": "agent_000000",
            "actions": [],
            "queued_actions_replaced": 0,
            "error": "",
            "usage": {},
            "research_state": {
                "goal": "test the wet material interface",
                "hypothesis": "porosity improves capture",
                "progress": "sample located",
                "next_checkpoint": "fabricate one batch",
                "collaboration_need": "find a second material",
            },
        },
        {
            "type": "actions",
            "tick": 0,
            "actions": {"agent_000000": {"verb": 2}},
        },
        {
            "type": "event",
            "tick": 0,
            "kind": "action_rejected",
            "payload": {"agent": "agent_000000", "reason": "blocked"},
        },
        {
            "type": "event",
            "tick": 0,
            "kind": "action_result",
            "payload": {
                "agent": "agent_000000",
                "action": "INSPECT",
                "success": False,
                "causal_parents": ["memory_a"],
            },
        },
        {
            "type": "event",
            "tick": 2,
            "kind": "microbatch_fabricated",
            "payload": {
                "agent": "agent_000000",
                "batch_id": "batch_00000000",
                "recipe_id": "recipe_a",
                "mass": 0.1,
            },
        },
        {
            "type": "provider_outage",
            "tick": 8,
            "world_advanced": False,
            "errors": [{"agent": "agent_000000", "error": "temporary DNS"}],
        },
        {
            "type": "model_trace",
            "tick": 8,
            "agent": "agent_000000",
            "planning_committed": False,
            "actions": [{"verb": 8}],
            "queued_actions_replaced": 0,
            "error": "model request failed: temporary DNS",
            "retrieval": {
                "selected": [{"record_id": "rolled_back_memory"}],
                "used_as_causal_parent": ["rolled_back_memory"],
            },
            "research_state": {"goal": "rolled back"},
        },
        {
            "type": "event",
            "tick": 8,
            "kind": "request_fulfilled",
            "payload": {
                "message_id": "message_a",
                "agent": "agent_000000",
                "verb": "INSPECT",
            },
        },
        {
            "type": "event",
            "tick": 3,
            "kind": "material_tested",
            "payload": {
                "agent": "agent_000000",
                "batch_id": "batch_00000000",
                "test_id": "test_00000000",
                "material_utility": 0.42,
            },
        },
        {
            "type": "event",
            "tick": 4,
            "kind": "task_claimed",
            "payload": {"agent": "agent_000001", "task": "test wet interfaces"},
        },
        {
            "type": "event",
            "tick": 5,
            "kind": "insight_deposited",
            "payload": {
                "record_id": "insight_parent",
                "author": "agent_000000",
                "causal_parents": [],
                "published": True,
            },
        },
        {
            "type": "event",
            "tick": 6,
            "kind": "insight_deposited",
            "payload": {
                "record_id": "insight_child",
                "author": "agent_000001",
                "causal_parents": ["insight_parent"],
                "published": True,
            },
        },
        {
            "type": "event",
            "tick": 7,
            "kind": "artifact_built",
            "payload": {
                "artifact_id": "artifact_00000000",
                "artifact_name": "Cited material",
                "agent": "agent_000001",
                "batch": {"causal_parents": ["insight_child"]},
            },
        },
        {
            "type": "snapshot",
            "snapshot": {
                "artifacts": {
                    "ids": ["artifact_00000000"],
                    "peak_performance": [0.25],
                },
                "research": {"outcome_success": False},
            },
        },
    ]
    source = tmp_path / "trace.jsonl"
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    report = analyze_trace(source)

    assert report["actions"]["planned"] == {"INSPECT": 1, "MOVE": 1}
    assert report["actions"]["executed"] == {"INSPECT": 1}
    assert report["model"]["queued_actions_replaced"]["total"] == 3
    assert report["model"]["provider_outage_attempts"] == 1
    assert report["model"]["rolled_back_plans"] == 1
    assert report["model"]["research_state"]["updates"] == 2
    assert report["model"]["research_state"]["agents"] == 1
    assert (
        report["model"]["research_state"]["per_agent"]["agent_000000"]["final_goal"]
        == "test the wet material interface"
    )
    assert report["model"]["research_state"]["mean_consecutive_goal_similarity"] > 0
    assert report["retrieval"]["selected_records"] == 1
    assert report["retrieval"]["selected_records_later_cited"] == 1
    assert report["retrieval"]["causal_outcome_attempts"] == 1
    assert report["retrieval"]["causal_success_rate"] == 0.0
    assert report["execution"]["rejections"] == 1
    assert report["execution"]["rejection_rate"] == 1.0
    assert report["experimental_cycle"]["test_completion_fraction"] == 1.0
    assert report["experimental_cycle"]["fabrication_to_test_latency_ticks"]["mean"] == 1.0
    assert report["experimental_cycle"]["untested_batch_ids"] == []
    assert report["experimental_cycle"]["tests"][0]["material_utility"] == 0.42
    assert report["coordination"]["distinct_task_claimants"] == 1
    assert report["information_flow"]["request_fulfillments"] == 1
    assert report["artifacts"]["cross_agent_epistemic_lineage_count"] == 1
    assert report["artifacts"]["best_cross_agent_lineage_peak_performance"] == 0.25
    assert report["artifacts"]["lineages"][0]["external_ancestry_authors"] == [
        "agent_000000"
    ]
    assert report["artifacts"]["lineages"][0]["created_tick"] == 7
    assert report["final_research"]["outcome_success"] is False
