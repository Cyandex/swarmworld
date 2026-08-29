from pathlib import Path

from fastapi.testclient import TestClient

from biofoundry import ENGINE_REVISION
from biofoundry.config import load_config
from biofoundry.events import EventRecorder
from biofoundry.playback import PlaybackTrace, TracePlayback, create_playback_app
from biofoundry.simulation import BioFoundrySimulation

ROOT = Path(__file__).resolve().parents[1]


def _write_trace(path: Path, *, ticks: int = 3) -> BioFoundrySimulation:
    config = load_config(ROOT / "configs" / "demo.yaml")
    simulation = BioFoundrySimulation(config)
    with EventRecorder(
        path,
        {
            "config": config.as_dict(),
            "engine_revision": ENGINE_REVISION,
            "policy": "scripted",
        },
        compression="gzip",
    ) as recorder:
        recorder.write_snapshot(
            simulation.snapshot(display_limit=None),
            state_digest=simulation.state_digest(),
        )
        for tick in range(ticks):
            recorder.write_record(
                {
                    "type": "model_trace",
                    "tick": tick,
                    "agent": simulation.agent_ids[0],
                    "planning_committed": False,
                    "error": "provider unavailable" if tick == 1 else "",
                    "actions": [{"verb": 0}],
                }
            )
            recorder.write_record(
                {
                    "type": "actions",
                    "tick": tick,
                    "macroturn_agents": [],
                    "actions": {},
                }
            )
            result = simulation.step({})
            recorder.write_events(result.events)
            recorder.write_snapshot(
                simulation.snapshot(display_limit=None),
                state_digest=simulation.state_digest(),
            )
    return simulation


def test_trace_playback_seeks_exactly_without_model_calls(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl.gz"
    expected = _write_trace(path)

    trace = PlaybackTrace.load(path)
    playback = TracePlayback(trace, ticks_per_second=12)

    assert trace.max_tick == 3
    assert trace.model_request_count == 3
    assert trace.model_error_count == 1
    assert playback.simulation.tick == 0
    assert playback.integrity_snapshots_verified == 4
    assert playback.dynamics_history["points"][0]["tick"] == 0
    assert playback.keyframes[-1]["snapshot"]["tick"] == 3

    frame = playback.step()
    assert frame is not None
    assert frame["snapshot"]["tick"] == 1
    assert frame["events"][-1]["kind"] == "agent_deliberated"

    reset_frame = playback.seek(0)
    assert reset_frame["playback_seek"] is True
    assert reset_frame["snapshot"]["tick"] == 0

    final_frame = playback.seek(3)
    assert final_frame["snapshot"]["tick"] == 3
    assert playback.simulation.state_digest() == expected.state_digest()
    assert playback.control_state()["playback_complete"] is True
    assert playback.control_state()["model_errors"] == 1


def test_playback_server_exposes_history_and_seekable_websocket(tmp_path: Path) -> None:
    path = tmp_path / "episode.jsonl.gz"
    _write_trace(path)
    app = create_playback_app(path, ticks_per_second=20)

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["mode"] == "offline-playback"
        assert health["llm_enabled"] is False
        assert health["replay"]["exact_tick_seek"] is True
        assert client.get("/history").json()["points"][-1]["tick"] == 3

        with client.websocket_connect("/ws") as websocket:
            packet = websocket.receive_json()
            assert packet["type"] == "snapshot"
            assert packet["control"]["playback_mode"] is True

            replay_ticks = []
            while True:
                packet = websocket.receive_json()
                if packet["type"] == "replay_frame":
                    replay_ticks.append(packet["frame"]["snapshot"]["tick"])
                if packet["type"] == "replay_ready":
                    break
            assert replay_ticks == [0, 1, 2, 3]

            websocket.send_json({"command": "seek", "tick": 2})
            for _ in range(20):
                packet = websocket.receive_json()
                if packet.get("type") == "frame":
                    break
            assert packet["snapshot"]["tick"] == 2
            assert packet["playback_seek"] is True
