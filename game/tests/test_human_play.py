"""Deterministic tests for the additive human-play layer.

Run with: python -m pytest game/tests
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

import yaml

from biofoundry.config import config_from_dict
from biofoundry.events import read_records
from biofoundry.playback import PlaybackTrace, TracePlayback
from biofoundry.simulation import BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction, Direction
from game.policy import HumanAwareLLMPolicy, ScheduleFilterView
from game.session import GameSession
from game.settings import GameSettings, load_game_profile, settings_from_mapping

HUMAN = "agent_000000"


def engine_config(**overrides):
    data = {
        "population": {"agents": 4, "macro_interval": 4},
        "simulation": {"seed": 11, "max_ticks": 64, "snapshot_interval": 4},
        "server": {"ticks_per_second": 1000.0},
    }
    data.update(overrides)
    return config_from_dict(data)


def settings(**overrides) -> GameSettings:
    values = {
        "human_agent": HUMAN,
        "decision_interval": 1,
        "input_timeout_seconds": 0.0,
        "require_response": False,
        "queue_limit": 4,
    }
    values.update(overrides)
    return settings_from_mapping(values)


def join(session: GameSession, connection: str = "conn-1") -> dict:
    reply = session.handle_game_command({"command": "join"}, connection)
    assert reply["accepted"], reply
    return reply


def submit(
    session: GameSession,
    action: dict,
    request_id: str = "req",
    connection: str = "conn-1",
) -> dict:
    return session.handle_game_command(
        {"command": "human_action", "request_id": request_id, "action": action},
        connection,
    )


def tick(session: GameSession) -> bool:
    return asyncio.run(session._tick_once())


def walkable_direction(simulation: BioFoundrySimulation, index: int) -> Direction:
    x = int(simulation.population.x[index])
    y = int(simulation.population.y[index])
    for direction, (dx, dy) in (
        (Direction.NORTH, (0, -1)),
        (Direction.EAST, (1, 0)),
        (Direction.SOUTH, (0, 1)),
        (Direction.WEST, (-1, 0)),
    ):
        nx, ny = x + dx, y + dy
        if 0 <= nx < simulation.world.width and 0 <= ny < simulation.world.height:
            if simulation.world.walkable[ny, nx]:
                return direction
    raise AssertionError("spawn cell has no walkable neighbor")


def test_profile_loader(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "game": {"human_agent": "agent_000001", "input_timeout_seconds": 2.5},
                "population": {"agents": 5},
            }
        )
    )
    config, game_settings = load_game_profile(profile)
    assert config.population.agents == 5
    assert game_settings.human_agent == "agent_000001"
    assert game_settings.input_timeout_seconds == 2.5

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"game": {"not_a_key": 1}}))
    try:
        load_game_profile(bad)
    except ValueError as exc:
        assert "not_a_key" in str(exc)
    else:
        raise AssertionError("unknown game key was accepted")


def test_schedule_filter_view():
    simulation = BioFoundrySimulation(engine_config())
    scheduled = simulation.scheduled_macro_agents()
    assert HUMAN in scheduled
    view = ScheduleFilterView(simulation, {HUMAN})
    filtered = view.scheduled_macro_agents()
    assert HUMAN not in filtered
    assert set(filtered) == set(scheduled) - {HUMAN}
    assert view.tick == simulation.tick
    assert view.population is simulation.population


def test_human_aware_policy_skips_excluded_queue():
    simulation = BioFoundrySimulation(engine_config())
    policy = HumanAwareLLMPolicy(provider=None, excluded_agents={HUMAN})
    policy.action_queues[HUMAN] = deque([AgentAction(verb=ActionType.INSPECT)])
    actions = policy.next_actions(simulation)
    assert HUMAN not in actions
    assert len(policy.action_queues[HUMAN]) == 1  # frozen, not consumed
    assert all(action.verb == ActionType.WAIT for action in actions.values())


def test_llm_session_excludes_joined_human():
    config = engine_config(llm={"enabled": True})
    session = GameSession(config, settings())
    assert isinstance(session.policy, HumanAwareLLMPolicy)
    assert HUMAN in session.simulation.scheduled_macro_agents()
    join(session)
    assert HUMAN not in session.policy.visible_scheduled(session.simulation)
    assert HUMAN in session.policy.excluded_agents
    release = session.handle_game_command({"command": "release"}, "conn-1")
    assert release["accepted"]
    assert HUMAN not in session.policy.excluded_agents


def test_end_to_end_human_session_records_and_replays(tmp_path):
    record = tmp_path / "session.jsonl"
    session = GameSession(engine_config(), settings(), record_path=record)
    join(session)
    index = session.simulation.population.id_to_index[HUMAN]
    start = (
        int(session.simulation.population.x[index]),
        int(session.simulation.population.y[index]),
    )
    direction = walkable_direction(session.simulation, index)

    assert submit(session, {"verb": "INSPECT"}, "r-inspect")["accepted"]
    assert tick(session)  # tick 0: human INSPECT

    # Durability: the actions record is flushed before any snapshot interval.
    flushed = [json.loads(line) for line in record.read_text().splitlines()]
    assert any(item.get("type") == "actions" for item in flushed)

    assert tick(session)  # tick 1: no human input -> WAIT (absent from record)
    assert submit(session, {"verb": "MOVE", "direction": direction.name}, "r-move")[
        "accepted"
    ]
    while session.simulation.tick < 8:
        assert tick(session)
    moved = (
        int(session.simulation.population.x[index]),
        int(session.simulation.population.y[index]),
    )
    assert moved != start
    session.close()

    records = list(read_records(record))
    action_records = {
        item["tick"]: item for item in records if item.get("type") == "actions"
    }
    assert action_records[0]["actions"][HUMAN]["verb"] == int(ActionType.INSPECT)
    assert HUMAN not in action_records[1]["actions"]
    assert action_records[2]["actions"][HUMAN]["verb"] == int(ActionType.MOVE)
    assert action_records[0]["macroturn_agents"] == []
    kinds = {item.get("kind") for item in records if item.get("type") == "event"}
    assert "sample_inspected" in kinds

    sidecar_lines = [
        json.loads(line)
        for line in (tmp_path / "session.jsonl.human.jsonl").read_text().splitlines()
    ]
    sidecar_types = [line["type"] for line in sidecar_lines]
    assert sidecar_types[0] == "session"
    assert "join" in sidecar_types
    human_actions = [line for line in sidecar_lines if line["type"] == "human_action"]
    assert [line["tick"] for line in human_actions] == [0, 2]
    assert human_actions[0]["request_id"] == "r-inspect"

    # The canonical trace replays deterministically with the standard tools.
    trace = PlaybackTrace.load(record)
    replay_sim = BioFoundrySimulation(trace.config)
    verified = 0
    for action_record in trace.action_records:
        TracePlayback._apply_record(replay_sim, action_record, policy=trace.policy)
        expected = trace.state_digests.get(replay_sim.tick)
        if expected is not None:
            assert replay_sim.state_digest() == expected
            verified += 1
    assert verified >= 2


def test_timeout_window_waits_then_continues():
    session = GameSession(engine_config(), settings(input_timeout_seconds=0.3))
    join(session)
    started = time.perf_counter()
    assert tick(session)
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.25
    assert session.simulation.tick == 1


def test_require_response_blocks_until_input():
    session = GameSession(engine_config(), settings(require_response=True))
    join(session)

    async def scenario() -> None:
        task = asyncio.create_task(session._tick_once())
        await asyncio.sleep(0.35)
        assert not task.done(), "world advanced without a human response"
        reply = submit(session, {"verb": "WAIT"}, "r-wait")
        assert reply["accepted"]
        assert await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())
    assert session.simulation.tick == 1


def test_command_guards():
    session = GameSession(engine_config(), settings())

    # human_action before join is refused.
    reply = submit(session, {"verb": "WAIT"})
    assert not reply["accepted"]

    join(session)

    # Malformed payloads are rejected, never raised.
    reply = session.handle_game_command(
        {"command": "human_action", "request_id": "r", "action": [1, 2]}, "conn-1"
    )
    assert not reply["accepted"] and "mapping" in reply["detail"]
    reply = session.handle_game_command(
        {"command": "manual_action", "agent": "agent_000001", "action": [1, 2]},
        "conn-1",
    )
    assert reply["type"] == "control" and "mapping" in reply["detail"]

    # The possessed slot cannot be puppeted through manual_action.
    reply = session.handle_game_command(
        {"command": "manual_action", "agent": HUMAN, "action": {"verb": 2}}, "conn-2"
    )
    assert "possessed" in reply["detail"]

    # A second connection cannot steal the slot, and only the owner releases it.
    reply = session.handle_game_command({"command": "join"}, "conn-2")
    assert not reply["accepted"]
    reply = session.handle_game_command({"command": "release"}, "conn-2")
    assert not reply["accepted"]
    assert session.release_connection("conn-1")
    assert not session.human.joined

    # Queue bound is enforced.
    join(session)
    for count in range(4):
        assert submit(session, {"verb": "WAIT"}, f"r{count}")["accepted"]
    assert not submit(session, {"verb": "WAIT"}, "overflow")["accepted"]


def test_paused_human_action_grants_single_step():
    session = GameSession(engine_config(), settings())
    join(session)
    session.handle_game_command({"command": "set_paused", "paused": True}, "conn-1")
    assert session.paused
    reply = submit(session, {"verb": "INSPECT"}, "r-paused")
    assert reply["accepted"]
    assert session.step_budget == 1
