"""Strict JSON Schema for model-authored BioFoundry action plans.

The schema mirrors the simulator's executable boundary.  Every object is closed and
every property is required, as required by OpenAI Structured Outputs.  Fields that
are irrelevant to a particular action are represented explicitly with neutral values
or ``null`` rather than being omitted.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any

from jsonschema import Draft202012Validator

from .programs import ACTUATOR_OPS, MAX_INSTRUCTIONS, MAX_REGISTERS, SENSORS
from .types import ActionType, ArtifactType, Direction, ProcessOperation, Resource

MAX_PLAN_ACTIONS = 16


def _closed_object(
    properties: dict[str, Any],
    *,
    description: str | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }
    if description is not None:
        schema["description"] = description
    return schema


def _nullable_ref(name: str) -> dict[str, Any]:
    return {"anyOf": [{"$ref": f"#/$defs/{name}"}, {"type": "null"}]}


_registers = [f"r{index}" for index in range(MAX_REGISTERS)]
_operands = sorted(SENSORS) + _registers

ACTION_PLAN_SCHEMA: dict[str, Any] = _closed_object(
    {
        "research_state": {"$ref": "#/$defs/research_state"},
        "plan": {
            "type": "array",
            "description": (
                "One to sixteen atomic actions, executed in order on later microticks. "
                "Later macroturns can continue the work without limiting lifetime actions."
            ),
            "items": {"$ref": "#/$defs/action"},
            "minItems": 1,
            "maxItems": MAX_PLAN_ACTIONS,
        }
    },
    description="A bounded BioFoundry action plan.",
)
ACTION_PLAN_SCHEMA["$defs"] = {
    "research_state": _closed_object(
        {
            "goal": {
                "type": "string",
                "maxLength": 320,
                "description": "The agent's current self-chosen research objective.",
            },
            "hypothesis": {
                "type": "string",
                "maxLength": 480,
                "description": (
                    "The agent's current falsifiable expectation, or an empty string."
                ),
            },
            "progress": {
                "type": "string",
                "maxLength": 480,
                "description": "What the agent believes it has established so far.",
            },
            "next_checkpoint": {
                "type": "string",
                "maxLength": 320,
                "description": (
                    "An observable condition at which the agent will reassess its plan."
                ),
            },
            "collaboration_need": {
                "type": "string",
                "maxLength": 320,
                "description": (
                    "Evidence, material, or help sought from peers, or an empty string."
                ),
            },
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "maxLength": 128},
                "maxItems": 16,
                "description": (
                    "Stable retained evidence/program IDs supporting the current state."
                ),
            },
        },
        description=(
            "A private, model-authored research notebook update. The simulator does "
            "not assign or score its content."
        ),
    ),
    "recipe_input": _closed_object(
        {
            "resource": {
                "type": "string",
                "enum": [resource.name for resource in Resource if resource != Resource.NONE],
                "description": (
                    "A representational material name, not a claim that it exists. "
                    "Executable recipes are accepted only when the acting agent has "
                    "empirical grounding for this material."
                ),
            },
            "mass": {
                "type": "number",
                "description": "Mass units consumed from the agent inventory.",
                "exclusiveMinimum": 0.0,
                "maximum": 20.0,
            },
        }
    ),
    "recipe_step": _closed_object(
        {
            "operation": {
                "type": "string",
                "enum": [operation.name for operation in ProcessOperation],
            },
            "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        }
    ),
    "recipe": _closed_object(
        {
            "inputs": {
                "type": "array",
                "description": (
                    "Distinct material inputs. List each resource at most once and "
                    "combine its requested mass into that entry."
                ),
                "items": {"$ref": "#/$defs/recipe_input"},
                "minItems": 1,
                "maxItems": 8,
            },
            "steps": {
                "type": "array",
                "description": (
                    "A purposeful processing sequence; do not repeat steps merely to "
                    "fill the array."
                ),
                "items": {"$ref": "#/$defs/recipe_step"},
                "minItems": 1,
                "maxItems": 12,
            },
            "output_form": {"type": "string", "maxLength": 160},
            "design_principles": {
                "type": "array",
                "description": (
                    "One to four distinct scientific design principles, each written "
                    "as a short phrase. Never repeat text to fill the array."
                ),
                "items": {"type": "string", "maxLength": 96},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        description="A material recipe accepted by the deterministic material lab.",
    ),
    "insight": _closed_object(
        {
            "content": {
                "type": "string",
                "maxLength": 1000,
                "description": "A concise, evidence-grounded scientific observation or claim.",
            },
            "kind": {"type": "string", "maxLength": 80},
            "salience": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        description="A spatially persistent scientific insight.",
    ),
    "artifact_geometry": _closed_object(
        {
            "layers": {"type": "integer", "minimum": 1, "maximum": 16},
            "surface_area": {"type": "number", "minimum": 0.25, "maximum": 4.0},
            "channel_density": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "anisotropy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "branching": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "connectivity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "curvature": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "modularity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        description="Generic geometry parameters; physical feasibility is bounded by mass.",
    ),
    "artifact_spec": _closed_object(
        {
            "name": {
                "type": "string",
                "maxLength": 160,
                "description": "Agent-invented name; it has no privileged physical meaning.",
            },
            "claimed_function": {
                "type": "string",
                "maxLength": 600,
                "description": "Falsifiable intended contribution to habitat resilience.",
            },
            "architecture": {
                "type": "string",
                "maxLength": 1200,
                "description": (
                    "Agent-authored physical organization; unrestricted prose with no "
                    "privileged simulator semantics."
                ),
            },
            "bio_inspiration": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "minItems": 1,
                "maxItems": 8,
            },
            "predicted_effects": {
                "type": "array",
                "items": {"type": "string", "maxLength": 300},
                "minItems": 1,
                "maxItems": 8,
            },
            "geometry": {"$ref": "#/$defs/artifact_geometry"},
        },
        description=(
            "An agent-authored identity and hypothesis for a generic material system. "
            "The simulator scores measured behavior, never the prose claim."
        ),
    ),
    "operand": {
        "anyOf": [
            {"type": "number"},
            {"type": "string", "enum": _operands},
        ]
    },
    "register": {"type": "string", "enum": _registers},
    "sense_instruction": _closed_object(
        {
            "op": {"type": "string", "enum": ["sense"]},
            "dest": {"$ref": "#/$defs/register"},
            "sensor": {"type": "string", "enum": sorted(SENSORS)},
        }
    ),
    "const_instruction": _closed_object(
        {
            "op": {"type": "string", "enum": ["const"]},
            "dest": {"$ref": "#/$defs/register"},
            "value": {"type": "number"},
        }
    ),
    "copy_instruction": _closed_object(
        {
            "op": {"type": "string", "enum": ["copy"]},
            "dest": {"$ref": "#/$defs/register"},
            "a": {"$ref": "#/$defs/operand"},
        }
    ),
    "binary_instruction": _closed_object(
        {
            "op": {
                "type": "string",
                "enum": ["add", "sub", "mul", "min", "max", "lt", "gt"],
            },
            "dest": {"$ref": "#/$defs/register"},
            "a": {"$ref": "#/$defs/operand"},
            "b": {"$ref": "#/$defs/operand"},
        }
    ),
    "actuator_instruction": _closed_object(
        {
            "op": {"type": "string", "enum": sorted(ACTUATOR_OPS)},
            "value": {"$ref": "#/$defs/operand"},
        }
    ),
    "instruction": {
        "anyOf": [
            {"$ref": "#/$defs/sense_instruction"},
            {"$ref": "#/$defs/const_instruction"},
            {"$ref": "#/$defs/copy_instruction"},
            {"$ref": "#/$defs/binary_instruction"},
            {"$ref": "#/$defs/actuator_instruction"},
        ]
    },
    "program": _closed_object(
        {
            "name": {"type": "string", "maxLength": 160},
            "parent_program": {
                "anyOf": [
                    {"type": "string", "maxLength": 160},
                    {"type": "null"},
                ],
                "description": (
                    "For FORK_PROGRAM, the exact program_id of a locally observed, "
                    "authored, inherited, or taught parent; otherwise null."
                ),
            },
            "instructions": {
                "type": "array",
                "items": {"$ref": "#/$defs/instruction"},
                "minItems": 1,
                "maxItems": MAX_INSTRUCTIONS,
            },
        },
        description="A bounded straight-line artifact behavior program.",
    ),
    "action": _closed_object(
        {
            "verb": {
                "type": "string",
                "enum": [action.name for action in ActionType],
            },
            "direction": {
                "type": "string",
                "enum": [direction.name for direction in Direction],
            },
            "resource": {
                "type": "string",
                "enum": [resource.name for resource in Resource],
                "description": (
                    "Use NONE for HARVEST, which samples local matter. A named material "
                    "is used only for grounded inventory transfer actions."
                ),
            },
            "artifact": {
                "type": "string",
                "enum": [artifact.name for artifact in ArtifactType],
            },
            "target_x": {"type": "integer"},
            "target_y": {"type": "integer"},
            "target_artifact_id": {
                "type": "string",
                "maxLength": 64,
                "description": (
                    "Stable ID of a locally observed artifact for programming, repair, "
                    "or dismantling; use an empty string when irrelevant."
                ),
            },
            "target_agent_id": {
                "type": "string",
                "maxLength": 64,
                "description": (
                    "A currently visible agent ID for addressed COMMUNICATE, TEACH, "
                    "or TRADE; empty string broadcasts or uses the local default."
                ),
            },
            "reply_to": {
                "type": "string",
                "maxLength": 128,
                "description": (
                    "Message record ID being answered or fulfilled; empty when this "
                    "action is not a reply."
                ),
            },
            "amount": {
                "type": "number",
                "description": "Requested resource mass; use zero when irrelevant.",
                "minimum": 0.0,
                "maximum": 20.0,
            },
            "message": {"type": "string", "maxLength": 500},
            "recipe": _nullable_ref("recipe"),
            "program": _nullable_ref("program"),
            "insight": _nullable_ref("insight"),
            "artifact_spec": _nullable_ref("artifact_spec"),
            "causal_parents": {
                "type": "array",
                "items": {"type": "string", "maxLength": 128},
                "maxItems": 32,
            },
        },
        description=(
            "One atomic action. Use direction STAY, resource/artifact NONE, amount 0, "
            "target coordinates -1, empty target_artifact_id and message strings, null "
            "recipe/program/insight/artifact_spec, and an empty causal_parents array "
            "whenever a field is irrelevant to the selected verb."
        ),
    ),
}


# Couple each action verb to the scientific payload it can actually execute.  A
# single action object with four independent nullable payloads is technically
# closed, but it admits semantically incoherent combinations such as INSPECT plus
# a recipe and makes constrained decoding unnecessarily ambiguous.  Grouping verbs
# by payload shape keeps every key explicit while turning the schema into an exact
# description of the simulator boundary.
_NEUTRAL_PAYLOAD_VERBS = (
    ActionType.INSPECT,
    ActionType.DEPOSIT,
    ActionType.TEST,
    ActionType.REPAIR,
    ActionType.DISMANTLE,
    ActionType.CLAIM_TASK,
)
_RECIPE_PAYLOAD_VERBS = (
    ActionType.OPERATE,
    ActionType.PROPOSE_RECIPE,
    ActionType.COMBINE_DESIGN,
)
_INSIGHT_PAYLOAD_VERBS = (ActionType.PUBLISH, ActionType.DEPOSIT_INSIGHT)
ACTION_PLAN_SCHEMA["$defs"]["original_program"] = copy.deepcopy(
    ACTION_PLAN_SCHEMA["$defs"]["program"]
)
ACTION_PLAN_SCHEMA["$defs"]["original_program"]["properties"]["parent_program"] = {
    "type": "null",
    "description": "Original programs cannot claim a parent.",
}
ACTION_PLAN_SCHEMA["$defs"]["fork_program"] = copy.deepcopy(
    ACTION_PLAN_SCHEMA["$defs"]["program"]
)
ACTION_PLAN_SCHEMA["$defs"]["fork_program"]["properties"]["parent_program"] = {
    "type": "string",
    "pattern": r"^program_[0-9a-f]{24}$",
    "description": "Exact immutable ID of a program known to this agent.",
}


def _action_variant(
    verbs: tuple[ActionType, ...],
    *,
    recipe: str = "null",
    program: str = "null",
    insight: str = "null",
    artifact_spec: str = "null",
    property_overrides: dict[str, Any] | None = None,
    allow_addressing: bool = False,
    allow_reply: bool = False,
) -> dict[str, Any]:
    """Create a closed action branch with verb-dependent payload structure."""

    generic = copy.deepcopy(ACTION_PLAN_SCHEMA["$defs"]["action"])
    properties = generic["properties"]
    properties["verb"] = {
        "type": "string",
        "enum": [verb.name for verb in verbs],
    }
    for field, mode in (
        ("recipe", recipe),
        ("program", program),
        ("insight", insight),
        ("artifact_spec", artifact_spec),
    ):
        properties[field] = (
            {"type": "null"}
            if mode == "null"
            else {"$ref": f"#/$defs/{mode}"}
            if mode != "optional"
            else _nullable_ref(field)
        )
    if not allow_addressing:
        properties["target_agent_id"] = {"type": "string", "enum": [""]}
    if not allow_reply:
        properties["reply_to"] = {"type": "string", "enum": [""]}
    if property_overrides:
        properties.update(copy.deepcopy(property_overrides))
    return generic


ACTION_PLAN_SCHEMA["$defs"]["action"] = {
    "description": (
        "One atomic action, discriminated by verb so irrelevant invention payloads "
        "must be null and required scientific payloads must be complete."
    ),
    "anyOf": [
        _action_variant((ActionType.WAIT,)),
        _action_variant(_NEUTRAL_PAYLOAD_VERBS, allow_reply=True),
        _action_variant(
            (ActionType.COMMUNICATE, ActionType.TEACH, ActionType.TRADE),
            allow_addressing=True,
            allow_reply=True,
        ),
        _action_variant(
            (ActionType.MOVE,),
            allow_reply=True,
            property_overrides={
                "direction": {
                    "type": "string",
                    "enum": [
                        direction.name
                        for direction in Direction
                        if direction != Direction.STAY
                    ],
                    "description": "MOVE requires one cardinal direction.",
                }
            },
        ),
        _action_variant(
            (ActionType.HARVEST,),
            allow_reply=True,
            property_overrides={
                "resource": {
                    "type": "string",
                    "enum": [Resource.NONE.name],
                    "description": "HARVEST samples present local matter; it never names it.",
                }
            },
        ),
        _action_variant(
            (ActionType.METABOLIZE,),
            allow_reply=True,
            property_overrides={
                "resource": {
                    "type": "string",
                    "enum": [
                        Resource.KELP.name,
                        Resource.FUNGUS.name,
                        Resource.CHITIN.name,
                        Resource.CELLULOSE.name,
                        Resource.CATALYST.name,
                    ],
                    "description": "Personally held metabolizable organic feedstock.",
                },
                "amount": {
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "maximum": 20.0,
                },
            },
        ),
        _action_variant(_RECIPE_PAYLOAD_VERBS, recipe="recipe", allow_reply=True),
        _action_variant(_INSIGHT_PAYLOAD_VERBS, insight="insight", allow_reply=True),
        _action_variant(
            (ActionType.WRITE_PROGRAM,), program="original_program", allow_reply=True
        ),
        _action_variant(
            (ActionType.FORK_PROGRAM,), program="fork_program", allow_reply=True
        ),
        _action_variant(
            (ActionType.BUILD,),
            recipe="recipe",
            program="optional",
            artifact_spec="artifact_spec",
            allow_reply=True,
            property_overrides={"program": _nullable_ref("original_program")},
        ),
    ],
}

ACTION_PLAN_TEXT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "biofoundry_agent_plan",
    "strict": True,
    "schema": ACTION_PLAN_SCHEMA,
}

_INTEGER_STRING = re.compile(r"^-?[0-9]{1,8}$")
_NUMBER_STRING = re.compile(r"^-?[0-9]{1,8}(?:\.[0-9]{1,6})?$")


def grammar_safe_action_plan_text_format() -> dict[str, Any]:
    """Return the same closed schema with finite numeric-string productions.

    mistral.rs 0.8.x accepts the Responses API JSON Schema shape but its grammar can
    emit decimal digits forever at constrained JSON-number positions. This
    transport-only representation prevents that decoder pathology. The response is
    converted back to canonical JSON numbers and validated against
    ``ACTION_PLAN_SCHEMA`` before any action reaches the simulator.
    """

    result = copy.deepcopy(ACTION_PLAN_TEXT_FORMAT)

    def rewrite(node: Any) -> None:
        if isinstance(node, dict):
            scalar_type = node.get("type")
            if isinstance(scalar_type, str) and scalar_type in {"integer", "number"}:
                description = str(node.get("description", "")).strip()
                minimum = node.get("minimum")
                maximum = node.get("maximum")
                if scalar_type == "number" and minimum == 0.0 and maximum == 1.0:
                    pattern = r"^(?:0(?:\.[0-9]{1,6})?|1(?:\.0{1,6})?)$"
                else:
                    pattern = (
                        r"^-?[0-9]{1,8}$"
                        if scalar_type == "integer"
                        else r"^-?[0-9]{1,8}(\.[0-9]{1,6})?$"
                    )
                bounds = []
                if minimum is not None:
                    bounds.append(f"minimum {minimum}")
                if maximum is not None:
                    bounds.append(f"maximum {maximum}")
                if bounds:
                    suffix = "Numeric string with " + " and ".join(bounds) + "."
                    description = f"{description} {suffix}".strip()
                node.clear()
                node.update(
                    {
                        "type": "string",
                        "pattern": pattern,
                    }
                )
                if description:
                    node["description"] = description
                return
            for child in node.values():
                rewrite(child)
        elif isinstance(node, list):
            for child in node:
                rewrite(child)

    rewrite(result["schema"])
    return result


def bounded_action_plan_text_format(
    max_plan_actions: int,
    *,
    grammar_safe_numbers: bool = False,
    allowed_action_types: Iterable[ActionType | str] | None = None,
    allow_addressing: bool = True,
    allow_replies: bool = True,
    resource_names: Iterable[str] | None = None,
    operation_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return the exact provider contract for one experimental capability set."""

    if not 1 <= max_plan_actions <= MAX_PLAN_ACTIONS:
        raise ValueError(f"max_plan_actions must be in [1, {MAX_PLAN_ACTIONS}]")
    result = (
        grammar_safe_action_plan_text_format()
        if grammar_safe_numbers
        else copy.deepcopy(ACTION_PLAN_TEXT_FORMAT)
    )
    result["schema"]["properties"]["plan"]["maxItems"] = max_plan_actions
    if resource_names is not None:
        resources = list(dict.fromkeys(str(item) for item in resource_names))
        if "NONE" not in resources:
            raise ValueError("resource_names must include NONE")
        nonzero = [item for item in resources if item != "NONE"]
        result["schema"]["$defs"]["recipe_input"]["properties"]["resource"][
            "enum"
        ] = nonzero
        for branch in result["schema"]["$defs"]["action"]["anyOf"]:
            branch["properties"]["resource"]["enum"] = resources
    if operation_names is not None:
        operations = list(dict.fromkeys(str(item) for item in operation_names))
        if not operations:
            raise ValueError("operation_names cannot be empty")
        result["schema"]["$defs"]["recipe_step"]["properties"]["operation"][
            "enum"
        ] = operations
    if allowed_action_types is not None:
        allowed = {
            item.name if isinstance(item, ActionType) else str(item)
            for item in allowed_action_types
        }
        branches = result["schema"]["$defs"]["action"]["anyOf"]
        retained: list[dict[str, Any]] = []
        for branch in branches:
            verb_schema = branch["properties"]["verb"]
            verbs = [verb for verb in verb_schema["enum"] if verb in allowed]
            if not verbs:
                continue
            verb_schema["enum"] = verbs
            if not allow_addressing:
                branch["properties"]["target_agent_id"] = {
                    "type": "string",
                    "enum": [""],
                }
            if not allow_replies:
                branch["properties"]["reply_to"] = {
                    "type": "string",
                    "enum": [""],
                }
            retained.append(branch)
        if not retained:
            raise ValueError("allowed_action_types must retain at least one action")
        result["schema"]["$defs"]["action"]["anyOf"] = retained
    return result


def coerce_transport_numbers(value: Any) -> Any:
    """Convert grammar-safe numeric strings to canonical plan numbers in-place."""

    if not isinstance(value, dict) or not isinstance(value.get("plan"), list):
        return value

    def number(mapping: dict[str, Any], key: str, *, integer: bool = False) -> None:
        raw = mapping.get(key)
        pattern = _INTEGER_STRING if integer else _NUMBER_STRING
        if not isinstance(raw, str) or pattern.fullmatch(raw) is None:
            return
        mapping[key] = int(raw) if integer else float(raw)

    for raw_action in value["plan"]:
        if not isinstance(raw_action, dict):
            continue
        number(raw_action, "target_x", integer=True)
        number(raw_action, "target_y", integer=True)
        number(raw_action, "amount")
        recipe = raw_action.get("recipe")
        if isinstance(recipe, dict):
            for item in recipe.get("inputs", []):
                if isinstance(item, dict):
                    number(item, "mass")
            for step in recipe.get("steps", []):
                if isinstance(step, dict):
                    number(step, "intensity")
        insight = raw_action.get("insight")
        if isinstance(insight, dict):
            number(insight, "salience")
        spec = raw_action.get("artifact_spec")
        if isinstance(spec, dict) and isinstance(spec.get("geometry"), dict):
            geometry = spec["geometry"]
            number(geometry, "layers", integer=True)
            for key in (
                "surface_area",
                "channel_density",
                "anisotropy",
                "branching",
                "connectivity",
                "curvature",
                "modularity",
            ):
                number(geometry, key)
        program = raw_action.get("program")
        if isinstance(program, dict):
            for instruction in program.get("instructions", []):
                if not isinstance(instruction, dict):
                    continue
                for key in ("value", "a", "b"):
                    number(instruction, key)
    return value

_ACTION_PLAN_VALIDATOR = Draft202012Validator(ACTION_PLAN_SCHEMA)


def validate_action_plan(
    value: Any,
    *,
    allowed_action_types: Iterable[ActionType | str] | None = None,
    allow_addressing: bool = True,
    allow_replies: bool = True,
    resource_names: Iterable[str] | None = None,
    operation_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a decoded response locally and return its typed root mapping."""
    # Engine revision 4 added explicit addressing fields. Accept old recorded/local
    # payloads by validating a neutral-field copy while preserving their exact value;
    # the advertised Structured Outputs schema remains fully closed and requires them.
    candidate = copy.deepcopy(value)
    if isinstance(candidate, dict) and isinstance(candidate.get("plan"), list):
        research_state = candidate.get("research_state")
        if isinstance(research_state, dict):
            research_state.setdefault("evidence_ids", [])
        for action in candidate["plan"]:
            if isinstance(action, dict):
                action.setdefault("target_agent_id", "")
                action.setdefault("reply_to", "")
    validator = _ACTION_PLAN_VALIDATOR
    if (
        allowed_action_types is not None
        or not allow_addressing
        or not allow_replies
        or resource_names is not None
        or operation_names is not None
    ):
        contract = bounded_action_plan_text_format(
            MAX_PLAN_ACTIONS,
            allowed_action_types=allowed_action_types or tuple(ActionType),
            allow_addressing=allow_addressing,
            allow_replies=allow_replies,
            resource_names=resource_names,
            operation_names=operation_names,
        )
        validator = Draft202012Validator(contract["schema"])
    errors = sorted(
        validator.iter_errors(candidate),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ValueError(f"action plan violates schema at {location}: {error.message}")
    return value
