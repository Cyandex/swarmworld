import pytest

from biofoundry.programs import ArtifactProgram, ProgramValidationError, ProgramVM


def test_program_runs_straight_line_sensor_logic() -> None:
    program = ArtifactProgram.from_dict(
        {
            "name": "collector",
            "instructions": [
                {"op": "mul", "dest": "r0", "a": "moisture", "b": 0.02},
                {"op": "collect_water", "value": "r0"},
            ],
        },
        author="agent_000001",
    )
    output = ProgramVM().execute(program, {"moisture": 0.8})
    assert output["collect_water"] == pytest.approx(0.016)


def test_program_rejects_arbitrary_code_and_loops() -> None:
    with pytest.raises(ProgramValidationError):
        ArtifactProgram.from_dict(
            {
                "name": "unsafe",
                "instructions": [{"op": "python", "value": "import os"}],
            }
        )
    with pytest.raises(ProgramValidationError):
        ArtifactProgram.from_dict(
            {
                "name": "loop",
                "instructions": [{"op": "jump", "value": -1}],
            }
        )
