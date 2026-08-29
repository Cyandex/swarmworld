from pathlib import Path

from fastapi.testclient import TestClient

from biofoundry import ENGINE_REVISION
from biofoundry.config import load_config
from biofoundry.events import read_records
from biofoundry.server import LiveGame, create_app
from biofoundry.types import ActionType

ROOT = Path(__file__).resolve().parents[1]


def test_live_server_health_snapshot_and_websocket(monkeypatch) -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    monkeypatch.setattr("biofoundry.server.HEARTBEAT_INTERVAL_SECONDS", 0.01)
    app = create_app(config)
    app.state.live.paused = True
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["agents"] == 12
        snapshot = client.get("/snapshot").json()
        assert snapshot["protocol"] == 1
        history = client.get("/history").json()
        assert history["version"] == 3
        assert history["points"][0]["tick"] == 0
        with client.websocket_connect("/ws") as websocket:
            for _ in range(12):
                packet = websocket.receive_json()
                if packet["type"] == "snapshot":
                    break
            assert packet["type"] == "snapshot"
            assert packet["snapshot"]["agents"]["count"] == 12
            assert packet["dynamics_history"]["points"][0]["tick"] == 0
            replay_ticks = []
            for _ in range(12):
                packet = websocket.receive_json()
                if packet["type"] == "replay_frame":
                    replay_ticks.append(packet["frame"]["snapshot"]["tick"])
                if packet["type"] == "replay_ready":
                    break
            assert packet["type"] == "replay_ready"
            assert replay_ticks == [0]
            for _ in range(12):
                packet = websocket.receive_json()
                if packet["type"] == "heartbeat":
                    break
            assert packet["type"] == "heartbeat"
            assert packet["tick"] == 0
            websocket.send_json({"command": "set_paused", "paused": True})
            for _ in range(12):
                packet = websocket.receive_json()
                if packet["type"] == "control":
                    break
            assert packet["type"] == "control"
            assert packet["control"]["paused"] is True
            websocket.send_json({"command": "step"})
            for _ in range(100):
                packet = websocket.receive_json()
                if packet["type"] == "frame":
                    break
            assert packet["type"] == "frame"
            assert packet["dynamics_point"]["tick"] == packet["snapshot"]["tick"]
            assert len(packet["dynamics_point"]["action_distribution"]) == len(ActionType)


def test_live_game_accepts_pause_speed_step_and_manual_action() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    live = LiveGame(config)
    assert live.handle_command({"command": "toggle_pause"})["control"]["paused"] is True
    state = live.handle_command({"command": "set_speed", "speed_multiplier": 2.0})
    assert state["control"]["speed_multiplier"] == 2.0
    live.handle_command({"command": "step"})
    assert live.step_budget == 1
    agent_id = live.simulation.agent_ids[0]
    result = live.handle_command(
        {
            "command": "manual_action",
            "agent": agent_id,
            "action": {"verb": 1, "direction": 2},
        }
    )
    assert result["detail"] == ""
    assert live.manual_actions[agent_id].direction == 2
    assert live.step_budget == 2


def test_live_game_retains_sampled_world_replay_for_late_viewers() -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    live = LiveGame(config)

    for tick in range(1, config.simulation.snapshot_interval * 2 + 1):
        live.simulation.tick = tick
        live._queue_replay_events(
            [{"tick": tick, "kind": "test_event", "payload": {"tick": tick}}]
        )
        live._record_replay_frame()

    ticks = [frame["snapshot"]["tick"] for frame in live.replay_frames]
    assert ticks == [
        0,
        config.simulation.snapshot_interval,
        config.simulation.snapshot_interval * 2,
    ]
    assert live.replay_manifest()["complete_history"] is True
    assert live.replay_frames[-1]["events"][-1]["tick"] == ticks[-1]


def test_live_recording_marks_engine_and_policy(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "demo.yaml")
    path = tmp_path / "live.jsonl"
    live = LiveGame(config, record_path=path)
    live.close()
    header = next(read_records(path))
    assert header["metadata"]["engine_revision"] == ENGINE_REVISION
    assert header["metadata"]["policy"] == "research-oracle"
