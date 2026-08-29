import re
from typing import Any

import pytest

from biofoundry.programs import ACTUATOR_OPS, SENSORS
from biofoundry.structured_output import (
    ACTION_PLAN_SCHEMA,
    ACTION_PLAN_TEXT_FORMAT,
    MAX_PLAN_ACTIONS,
    bounded_action_plan_text_format,
    coerce_transport_numbers,
    grammar_safe_action_plan_text_format,
    validate_action_plan,
)
from biofoundry.types import ActionType, ArtifactType, Direction, ProcessOperation, Resource

RESEARCH_STATE = {
    "goal": "characterize local matter",
    "hypothesis": "processing changes moisture response",
    "progress": "no test yet",
    "next_checkpoint": "obtain one measured batch",
    "collaboration_need": "",
    "evidence_ids": [],
}


def _plan(*actions: dict[str, Any]) -> dict[str, Any]:
    return {"research_state": dict(RESEARCH_STATE), "plan": list(actions)}


def _assert_all_objects_are_closed(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(node.get("properties", {}))
        for value in node.values():
            _assert_all_objects_are_closed(value)


def _assert_free_strings_are_bounded(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "string" and "enum" not in node and "pattern" not in node:
            assert isinstance(node.get("maxLength"), int)
            assert node["maxLength"] > 0
        for value in node.values():
            _assert_free_strings_are_bounded(value)
    elif isinstance(node, list):
        for value in node:
            _assert_free_strings_are_bounded(value)
    elif isinstance(node, list):
        for value in node:
            _assert_all_objects_are_closed(value)


def test_structured_output_uses_strict_named_json_schema() -> None:
    assert ACTION_PLAN_TEXT_FORMAT["type"] == "json_schema"
    assert ACTION_PLAN_TEXT_FORMAT["strict"] is True
    assert ACTION_PLAN_TEXT_FORMAT["name"] == "biofoundry_agent_plan"
    assert ACTION_PLAN_TEXT_FORMAT["schema"] is ACTION_PLAN_SCHEMA
    assert ACTION_PLAN_SCHEMA["type"] == "object"
    assert ACTION_PLAN_SCHEMA["properties"]["plan"]["maxItems"] == MAX_PLAN_ACTIONS == 16
    assert "anyOf" not in ACTION_PLAN_SCHEMA
    _assert_all_objects_are_closed(ACTION_PLAN_SCHEMA)
    _assert_free_strings_are_bounded(ACTION_PLAN_SCHEMA)


def test_action_schema_covers_every_simulator_enum() -> None:
    definitions = ACTION_PLAN_SCHEMA["$defs"]
    branches = definitions["action"]["anyOf"]
    assert {
        verb
        for branch in branches
        for verb in branch["properties"]["verb"]["enum"]
    } == {value.name for value in ActionType}
    assert {
        resource
        for branch in branches
        for resource in branch["properties"]["resource"]["enum"]
    } == {value.name for value in Resource}
    assert {
        direction
        for branch in branches
        for direction in branch["properties"]["direction"]["enum"]
    } == {value.name for value in Direction}
    for branch in branches:
        action = branch["properties"]
        assert set(action["direction"]["enum"]) <= {value.name for value in Direction}
        assert action["artifact"]["enum"] == [value.name for value in ArtifactType]
    assert definitions["recipe_step"]["properties"]["operation"]["enum"] == [
        value.name for value in ProcessOperation
    ]


def test_program_schema_matches_vm_sensors_registers_and_actuators() -> None:
    definitions = ACTION_PLAN_SCHEMA["$defs"]
    assert definitions["sense_instruction"]["properties"]["sensor"]["enum"] == sorted(
        SENSORS
    )
    assert definitions["actuator_instruction"]["properties"]["op"]["enum"] == sorted(
        ACTUATOR_OPS
    )
    assert definitions["register"]["enum"] == [f"r{index}" for index in range(16)]
    assert definitions["program"]["properties"]["instructions"]["maxItems"] == 64
    principles = definitions["recipe"]["properties"]["design_principles"]
    assert principles["maxItems"] == 4
    assert principles["items"]["maxLength"] == 96


def test_grammar_safe_transport_round_trips_numbers_before_canonical_validation() -> None:
    transport = grammar_safe_action_plan_text_format()
    action = transport["schema"]["$defs"]["action"]["anyOf"][0]["properties"]
    assert action["amount"]["type"] == "string"
    assert action["target_x"]["type"] == "string"
    canonical_action = ACTION_PLAN_SCHEMA["$defs"]["action"]["anyOf"][0]["properties"]
    assert canonical_action["amount"]["type"] == "number"
    intensity = transport["schema"]["$defs"]["recipe_step"]["properties"][
        "intensity"
    ]
    assert re.fullmatch(intensity["pattern"], "0.75")
    assert re.fullmatch(intensity["pattern"], "1.0")
    assert re.fullmatch(intensity["pattern"], "19") is None

    bounded = bounded_action_plan_text_format(6, grammar_safe_numbers=True)
    assert bounded["schema"]["properties"]["plan"]["maxItems"] == 6
    assert bounded["schema"]["$defs"]["action"]["anyOf"][0]["properties"]["amount"][
        "type"
    ] == "string"

    payload = {
        "research_state": dict(RESEARCH_STATE),
        "plan": [
            {
                "verb": "WAIT",
                "direction": "STAY",
                "resource": "NONE",
                "artifact": "NONE",
                "target_x": "-1",
                "target_y": "-1",
                "target_artifact_id": "",
                "amount": "0",
                "message": "",
                "recipe": None,
                "program": None,
                "artifact_spec": None,
                "insight": None,
                "causal_parents": [],
            }
        ]
    }
    canonical = coerce_transport_numbers(payload)
    assert canonical["plan"][0]["target_x"] == -1
    assert canonical["plan"][0]["amount"] == 0.0
    validate_action_plan(canonical)


def test_local_validation_blocks_old_malformed_insight_and_recipe_shapes() -> None:
    base_action = {
        "verb": "DEPOSIT_INSIGHT",
        "direction": "STAY",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "amount": 0.0,
        "message": "",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": {
            "content": "layered cuticle redirects cracks",
            "kind": "observation",
            "salience": 0.8,
        },
        "causal_parents": [],
    }
    assert validate_action_plan(_plan(base_action))["plan"][0] == base_action

    malformed_insight = {**base_action, "insight": "layered cuticle redirects cracks"}
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan(malformed_insight))

    malformed_recipe = {
        **base_action,
        "verb": "PROPOSE_RECIPE",
        "insight": None,
        "recipe": {
            "inputs": ["CHITIN"],
            "steps": [{"operation": "WASH", "intensity": 0.5}],
            "output_form": "film",
            "design_principles": ["layering"],
        },
    }
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan(malformed_recipe))


def test_action_payload_is_discriminated_by_verb() -> None:
    wait = {
        "verb": "WAIT",
        "direction": "STAY",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "amount": 0.0,
        "message": "",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": None,
        "causal_parents": [],
    }
    validate_action_plan(_plan(wait))
    incoherent = {
        **wait,
        "recipe": {
            "inputs": [{"resource": "KELP", "mass": 1.0}],
            "steps": [{"operation": "WASH", "intensity": 0.5}],
            "output_form": "film",
            "design_principles": ["clean interface"],
        },
    }
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan(incoherent))


def test_noncommunication_actions_may_close_requests_but_wait_cannot() -> None:
    inspect = {
        "verb": "INSPECT",
        "direction": "STAY",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "target_agent_id": "",
        "reply_to": "message_000000000000000000000001",
        "amount": 0.0,
        "message": "",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": None,
        "causal_parents": [],
    }
    validate_action_plan(_plan(inspect))
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan({**inspect, "verb": "WAIT"}))


def test_harvest_cannot_request_named_matter() -> None:
    harvest = {
        "verb": "HARVEST",
        "direction": "STAY",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "amount": 0.0,
        "message": "sample local matter",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": None,
        "causal_parents": [],
    }
    validate_action_plan(_plan(harvest))
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan({**harvest, "resource": "MINERAL"}))


def test_move_cannot_select_stay_direction() -> None:
    move = {
        "verb": "MOVE",
        "direction": "NORTH",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "amount": 0.0,
        "message": "explore north",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": None,
        "causal_parents": [],
    }
    validate_action_plan(_plan(move))
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(_plan({**move, "direction": "STAY"}))


def test_research_state_is_closed_private_model_authored_structure() -> None:
    schema = ACTION_PLAN_SCHEMA["$defs"]["research_state"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(RESEARCH_STATE)
    assert "score" not in schema["properties"]
    assert "assigned_role" not in schema["properties"]


def test_condition_specific_schema_excludes_disabled_verbs_and_social_links() -> None:
    allowed = (ActionType.WAIT, ActionType.MOVE, ActionType.INSPECT)
    contract = bounded_action_plan_text_format(
        4,
        allowed_action_types=allowed,
        allow_addressing=False,
        allow_replies=False,
    )
    branches = contract["schema"]["$defs"]["action"]["anyOf"]
    verbs = {
        verb
        for branch in branches
        for verb in branch["properties"]["verb"]["enum"]
    }
    assert verbs == {action.name for action in allowed}
    assert all(
        branch["properties"]["target_agent_id"]["enum"] == [""]
        and branch["properties"]["reply_to"]["enum"] == [""]
        for branch in branches
    )

    wait = {
        "verb": "WAIT",
        "direction": "STAY",
        "resource": "NONE",
        "artifact": "NONE",
        "target_x": -1,
        "target_y": -1,
        "target_artifact_id": "",
        "target_agent_id": "",
        "reply_to": "",
        "amount": 0.0,
        "message": "",
        "recipe": None,
        "program": None,
        "artifact_spec": None,
        "insight": None,
        "causal_parents": [],
    }
    validate_action_plan(_plan(wait), allowed_action_types=allowed)
    with pytest.raises(ValueError, match="action plan violates schema"):
        validate_action_plan(
            _plan({**wait, "verb": "COMMUNICATE"}),
            allowed_action_types=allowed,
            allow_addressing=False,
            allow_replies=False,
        )
