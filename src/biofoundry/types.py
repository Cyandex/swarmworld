"""Stable numeric types shared by the simulator, API, logs, and renderer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

PROTOCOL_VERSION = 1


class Terrain(IntEnum):
    DEEP_WATER = 0
    TIDAL = 1
    MEADOW = 2
    FUNGAL_GROVE = 3
    CHITIN_GARDEN = 4
    CELLULOSE_FIELD = 5
    MINERAL_SPRING = 6
    FOUNDRY = 7
    TEST_FIELD = 8


class Resource(IntEnum):
    NONE = 0
    KELP = 1
    SHELL = 2
    FUNGUS = 3
    CHITIN = 4
    CELLULOSE = 5
    MINERAL = 6
    WATER = 7
    CATALYST = 8


class Station(IntEnum):
    NONE = 0
    FERMENTER = 1
    WASHER = 2
    PRESS = 3
    ALIGNER = 4
    TESTER = 5
    ARCHIVE = 6


class ActionType(IntEnum):
    WAIT = 0
    MOVE = 1
    INSPECT = 2
    HARVEST = 3
    DEPOSIT = 4
    OPERATE = 5
    TEST = 6
    PROPOSE_RECIPE = 7
    BUILD = 8
    REPAIR = 9
    DISMANTLE = 10
    COMMUNICATE = 11
    PUBLISH = 12
    DEPOSIT_INSIGHT = 13
    WRITE_PROGRAM = 14
    FORK_PROGRAM = 15
    CLAIM_TASK = 16
    TEACH = 17
    TRADE = 18
    COMBINE_DESIGN = 19
    METABOLIZE = 20


class Direction(IntEnum):
    STAY = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class ArtifactType(IntEnum):
    NONE = 0
    MATERIAL_SYSTEM = 1


class ProcessOperation(IntEnum):
    WASH = 0
    GRIND = 1
    FERMENT = 2
    ALKALINE_TREAT = 3
    MINERALIZE = 4
    ALIGN = 5
    WEAVE = 6
    PRESS = 7
    DRY = 8
    COAT = 9


DIRECTION_DELTAS: dict[Direction, tuple[int, int]] = {
    Direction.STAY: (0, 0),
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}


@dataclass(slots=True)
class AgentAction:
    """Validated semantic action before it enters the simulator."""

    verb: ActionType = ActionType.WAIT
    direction: Direction = Direction.STAY
    resource: Resource = Resource.NONE
    artifact: ArtifactType = ArtifactType.NONE
    target_x: int = -1
    target_y: int = -1
    target_artifact_id: str = ""
    target_agent_id: str = ""
    reply_to: str = ""
    amount: float = 0.0
    message: str = ""
    recipe: dict[str, Any] | None = None
    program: dict[str, Any] | None = None
    insight: dict[str, Any] | None = None
    artifact_spec: dict[str, Any] | None = None
    causal_parents: list[str] = field(default_factory=list)

    @staticmethod
    def _enum_member(enum_type: type[IntEnum], value: Any, default: IntEnum) -> IntEnum:
        """Accept semantic model names while retaining numeric wire compatibility."""
        if isinstance(value, str):
            try:
                return enum_type[value]
            except KeyError:
                value = int(value)
        return enum_type(int(value if value is not None else default))

    @classmethod
    def from_value(cls, value: AgentAction | Mapping[str, Any] | None) -> AgentAction:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(
            verb=cls._enum_member(
                ActionType, value.get("verb", ActionType.WAIT), ActionType.WAIT
            ),
            direction=cls._enum_member(
                Direction, value.get("direction", Direction.STAY), Direction.STAY
            ),
            resource=cls._enum_member(
                Resource, value.get("resource", Resource.NONE), Resource.NONE
            ),
            artifact=cls._enum_member(
                ArtifactType, value.get("artifact", ArtifactType.NONE), ArtifactType.NONE
            ),
            target_x=int(value.get("target_x", value.get("target", [-1, -1])[0])),
            target_y=int(value.get("target_y", value.get("target", [-1, -1])[1])),
            target_artifact_id=str(value.get("target_artifact_id", ""))[:64],
            target_agent_id=str(value.get("target_agent_id", ""))[:64],
            reply_to=str(value.get("reply_to", ""))[:128],
            amount=float(value.get("amount", 0.0)),
            message=str(value.get("message", ""))[:512],
            recipe=value.get("recipe"),
            program=value.get("program"),
            insight=value.get("insight"),
            artifact_spec=value.get("artifact_spec"),
            causal_parents=list(value.get("causal_parents", []))[:32],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verb": int(self.verb),
            "direction": int(self.direction),
            "resource": int(self.resource),
            "artifact": int(self.artifact),
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_artifact_id": self.target_artifact_id,
            "target_agent_id": self.target_agent_id,
            "reply_to": self.reply_to,
            "amount": self.amount,
            "message": self.message,
            "recipe": self.recipe,
            "program": self.program,
            "insight": self.insight,
            "artifact_spec": self.artifact_spec,
            "causal_parents": self.causal_parents,
        }


@dataclass(slots=True)
class WorldEvent:
    tick: int
    kind: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"tick": self.tick, "kind": self.kind, "payload": self.payload}
