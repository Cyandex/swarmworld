"""A bounded deterministic DSL for agent-authored artifact behavior."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import ArtifactType

MAX_INSTRUCTIONS = 64
MAX_REGISTERS = 16
MAX_ACTUATION_PER_TICK = 0.05

SENSORS = {
    "moisture",
    "nutrients",
    "temperature",
    "solar",
    "contamination",
    "health",
    "maturity",
    "storage",
    "reserve",
    "open_fraction",
    "permeability",
    "toughness",
    "healing",
    "responsiveness",
}

ARITHMETIC_OPS = {"const", "copy", "add", "sub", "mul", "min", "max", "lt", "gt"}
ACTUATOR_OPS = {
    "collect_water",
    "grow",
    "heal",
    "set_open",
    "reduce_contamination",
    "emit_signal",
}


class ProgramValidationError(ValueError):
    pass


def _register_index(value: Any) -> int:
    if isinstance(value, int):
        index = value
    elif isinstance(value, str) and value.startswith("r") and value[1:].isdigit():
        index = int(value[1:])
    else:
        raise ProgramValidationError(f"expected register r0..r{MAX_REGISTERS - 1}, got {value!r}")
    if not 0 <= index < MAX_REGISTERS:
        raise ProgramValidationError(f"register index outside range: {index}")
    return index


def _operand(value: Any, registers: np.ndarray, sensors: Mapping[str, float]) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value in sensors:
        return float(sensors[value])
    return float(registers[_register_index(value)])


@dataclass(frozen=True, slots=True)
class ArtifactProgram:
    name: str
    instructions: tuple[dict[str, Any], ...]
    author: str = "system"
    parent_program: str | None = None

    @property
    def program_id(self) -> str:
        """Content identity: names, authors, and claimed ancestry cannot spoof it."""

        payload = json.dumps(
            [dict(item) for item in self.instructions],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"program_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], author: str = "system") -> ArtifactProgram:
        instructions = tuple(dict(item) for item in data.get("instructions", []))
        program = cls(
            name=str(data.get("name", "untitled_program"))[:80],
            instructions=instructions,
            author=author,
            parent_program=(
                str(data["parent_program"])[:120] if data.get("parent_program") else None
            ),
        )
        program.validate()
        return program

    def validate(self) -> None:
        if not self.name:
            raise ProgramValidationError("program name cannot be empty")
        if not 1 <= len(self.instructions) <= MAX_INSTRUCTIONS:
            raise ProgramValidationError(
                f"program must contain 1 to {MAX_INSTRUCTIONS} instructions"
            )
        for pc, instruction in enumerate(self.instructions):
            op = str(instruction.get("op", "")).lower()
            if op == "sense":
                _register_index(instruction.get("dest"))
                if instruction.get("sensor") not in SENSORS:
                    raise ProgramValidationError(f"instruction {pc}: unknown sensor")
            elif op in ARITHMETIC_OPS:
                _register_index(instruction.get("dest"))
                if op == "const":
                    if not isinstance(instruction.get("value"), (int, float)):
                        raise ProgramValidationError(f"instruction {pc}: const needs a number")
                elif op == "copy":
                    self._validate_operand(instruction.get("a"), pc)
                else:
                    self._validate_operand(instruction.get("a"), pc)
                    self._validate_operand(instruction.get("b"), pc)
            elif op in ACTUATOR_OPS:
                self._validate_operand(instruction.get("value", 0.0), pc)
            else:
                raise ProgramValidationError(f"instruction {pc}: unsupported op {op!r}")

    @staticmethod
    def _validate_operand(value: Any, pc: int) -> None:
        if isinstance(value, (int, float)):
            return
        if isinstance(value, str) and value in SENSORS:
            return
        try:
            _register_index(value)
        except ProgramValidationError as exc:
            raise ProgramValidationError(f"instruction {pc}: invalid operand {value!r}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "name": self.name,
            "author": self.author,
            "parent_program": self.parent_program,
            "instructions": [dict(item) for item in self.instructions],
        }


def instruction_diff(
    parent: ArtifactProgram, child: ArtifactProgram
) -> list[dict[str, Any]]:
    """Return a deterministic, executable-instruction lineage diff."""

    changes: list[dict[str, Any]] = []
    length = max(len(parent.instructions), len(child.instructions))
    for index in range(length):
        before = dict(parent.instructions[index]) if index < len(parent.instructions) else None
        after = dict(child.instructions[index]) if index < len(child.instructions) else None
        if before != after:
            changes.append({"index": index, "before": before, "after": after})
    return changes


class ProgramVM:
    """Straight-line interpreter: no jumps, loops, allocation, or external access."""

    def execute(
        self,
        program: ArtifactProgram,
        sensors: Mapping[str, float],
    ) -> dict[str, float]:
        registers = np.zeros(MAX_REGISTERS, dtype=np.float32)
        outputs: dict[str, float] = {}
        for instruction in program.instructions:
            op = str(instruction["op"]).lower()
            if op == "sense":
                registers[_register_index(instruction["dest"])] = np.float32(
                    sensors[str(instruction["sensor"])]
                )
                continue
            if op in ARITHMETIC_OPS:
                dest = _register_index(instruction["dest"])
                if op == "const":
                    result = float(instruction["value"])
                elif op == "copy":
                    result = _operand(instruction["a"], registers, sensors)
                else:
                    a = _operand(instruction["a"], registers, sensors)
                    b = _operand(instruction["b"], registers, sensors)
                    result = {
                        "add": a + b,
                        "sub": a - b,
                        "mul": a * b,
                        "min": min(a, b),
                        "max": max(a, b),
                        "lt": float(a < b),
                        "gt": float(a > b),
                    }[op]
                registers[dest] = np.float32(np.clip(result, -4.0, 4.0))
                continue
            value = _operand(instruction.get("value", 0.0), registers, sensors)
            if op == "set_open":
                outputs[op] = float(np.clip(value, 0.0, 1.0))
            else:
                outputs[op] = outputs.get(op, 0.0) + float(
                    np.clip(value, 0.0, MAX_ACTUATION_PER_TICK)
                )
        return outputs


def default_program(kind: ArtifactType) -> ArtifactProgram:
    # Physical form never supplies a function. Agents must author behavior explicitly.
    return ArtifactProgram.from_dict(
        {
            "name": f"passive_{kind.name.lower()}",
            "instructions": [{"op": "const", "dest": "r0", "value": 0.0}],
        }
    )
