"""Offline, deterministic playback of a completed BioFoundry trace."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tqdm import tqdm

from . import ENGINE_REVISION, __version__
from .config import GameConfig, config_from_dict
from .counterfactuals import (
    acknowledge_recorded_macroturns,
    apply_recorded_model_trace,
)
from .dynamics import SocietyDynamicsTracker
from .events import read_records
from .scenarios import load_scenario
from .server import HEARTBEAT_INTERVAL_SECONDS, FrameHub
from .simulation import BioFoundrySimulation

MAX_KEYFRAME_EVENTS = 512
MAX_RECENT_EVENTS = 2_000


def _compact_model_trace(record: dict[str, Any]) -> dict[str, Any]:
    """Retain only fields that affect replay or the presentation event stream."""

    actions = record.get("actions")
    first_action = actions[0] if isinstance(actions, list) and actions else {}
    return {
        "tick": int(record.get("tick", -1)),
        "agent": str(record.get("agent", "")),
        "planning_committed": record.get("planning_committed", True),
        "research_state": record.get("research_state"),
        "error": str(record.get("error", ""))[:280],
        "first_verb": int(first_action.get("verb", 0)) if isinstance(first_action, dict) else 0,
    }


def _priority_events(events: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep causal and human-readable events when a keyframe interval is very busy."""

    if len(events) <= limit:
        return events
    preferred = {
        "message_delivered",
        "artifact_built",
        "artifact_program_installed",
        "artifact_repaired",
        "artifact_dismantled",
        "insight_deposited",
        "sample_tested",
        "microbatch_fabricated",
        "research_milestone",
        "action_rejected",
        "model_error",
    }
    salient = [event for event in events if str(event.get("kind")) in preferred]
    remainder = [event for event in events if str(event.get("kind")) not in preferred]
    if len(salient) >= limit:
        return salient[-limit:]
    return salient + remainder[-(limit - len(salient)) :]


@dataclass(slots=True)
class PlaybackTrace:
    path: Path
    config: GameConfig
    policy: str
    model: str
    action_records: list[dict[str, Any]]
    recorded_snapshots: dict[int, dict[str, Any]]
    state_digests: dict[int, str]
    model_request_count: int
    model_error_count: int
    max_tick: int

    @classmethod
    def load(cls, path: str | Path) -> PlaybackTrace:
        source = Path(path)
        header: dict[str, Any] | None = None
        action_records: list[dict[str, Any]] = []
        model_traces: dict[int, list[dict[str, Any]]] = {}
        snapshots: dict[int, dict[str, Any]] = {}
        state_digests: dict[int, str] = {}
        model_request_count = 0
        model_error_count = 0

        for record in tqdm(read_records(source), desc="Reading playback trace", unit="record"):
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "model_trace":
                model_request_count += 1
                model_error_count += int(bool(record.get("error")))
                tick = int(record.get("tick", -1))
                model_traces.setdefault(tick, []).append(_compact_model_trace(record))
            elif kind == "actions":
                tick = int(record.get("tick", -1))
                action = {
                    "tick": tick,
                    "macroturn_agents": list(record.get("macroturn_agents", [])),
                    "actions": dict(record.get("actions", {})),
                    "model_traces": model_traces.pop(tick, []),
                }
                action_records.append(action)
            elif kind == "snapshot" and isinstance(record.get("snapshot"), dict):
                snapshot = dict(record["snapshot"])
                tick = int(snapshot.get("tick", -1))
                snapshots[tick] = snapshot
                if record.get("state_digest"):
                    state_digests[tick] = str(record["state_digest"])

        if header is None:
            raise ValueError("playback trace has no header")
        metadata = dict(header.get("metadata", {}))
        revision = metadata.get("engine_revision")
        if revision != ENGINE_REVISION:
            raise ValueError(
                "exact playback requires the generating engine revision: "
                f"trace={revision}, current={ENGINE_REVISION}. Check out the recorded Git commit."
            )
        config_data = metadata.get("config")
        if not isinstance(config_data, dict):
            raise ValueError("playback trace has no complete configuration")
        config = config_from_dict(config_data)
        config.validate()
        scenario = load_scenario(config.world.scenario_package)
        recorded_scenario = metadata.get("scenario", {})
        if scenario is not None and isinstance(recorded_scenario, dict):
            recorded_hash = recorded_scenario.get("package_hash")
            if recorded_hash and recorded_hash != scenario.package_hash:
                raise ValueError(
                    "scenario package differs from the recorded trace: "
                    f"trace={recorded_hash}, current={scenario.package_hash}"
                )
        action_records.sort(key=lambda item: int(item["tick"]))
        max_tick = max(
            max(snapshots, default=0),
            int(action_records[-1]["tick"]) + 1 if action_records else 0,
        )
        return cls(
            path=source.resolve(),
            config=config,
            policy=str(metadata.get("policy", "recorded")),
            model=config.llm.model,
            action_records=action_records,
            recorded_snapshots=snapshots,
            state_digests=state_digests,
            model_request_count=model_request_count,
            model_error_count=model_error_count,
            max_tick=max_tick,
        )


class TracePlayback:
    """Reconstruct a trace without model calls and expose seekable presentation state."""

    def __init__(self, trace: PlaybackTrace, *, ticks_per_second: float = 8.0):
        if ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be positive")
        self.trace = trace
        self.base_ticks_per_second = float(ticks_per_second)
        self.speed_multiplier = 1.0
        self.paused = True
        self.complete = False
        self.simulation = BioFoundrySimulation(trace.config)
        self.action_index = 0
        self.recent_events: list[dict[str, Any]] = []
        self.dynamics_history: dict[str, Any] = {}
        self.keyframes: list[dict[str, Any]] = []
        self.integrity_snapshots_verified = 0
        self._prepare_observer_history()
        self.reset()

    @staticmethod
    def _trace_event(record: dict[str, Any]) -> dict[str, Any]:
        error = str(record.get("error", ""))
        return {
            "tick": int(record.get("tick", 0)),
            "kind": "model_error" if error else "agent_deliberated",
            "payload": {
                "agent": str(record.get("agent", "")),
                "verb": int(record.get("first_verb", 0)),
                "error": error,
            },
        }

    @staticmethod
    def _apply_record(
        simulation: BioFoundrySimulation,
        record: dict[str, Any],
        *,
        policy: str,
    ) -> list[dict[str, Any]]:
        expected_tick = int(record.get("tick", -1))
        if expected_tick != simulation.tick:
            raise ValueError(
                f"action trace tick {expected_tick} does not match replay tick {simulation.tick}"
            )
        for model_trace in record.get("model_traces", []):
            apply_recorded_model_trace(simulation, model_trace)
        acknowledge_recorded_macroturns(
            simulation,
            record,
            policy=policy,
        )
        result = simulation.step(dict(record.get("actions", {})))
        events = [event.as_dict() for event in result.events]
        events.extend(TracePlayback._trace_event(item) for item in record.get("model_traces", []))
        return events

    def _verify_digest(self, simulation: BioFoundrySimulation) -> None:
        expected = self.trace.state_digests.get(simulation.tick)
        if expected is None:
            return
        observed = simulation.state_digest()
        if observed != expected:
            raise ValueError(
                "deterministic playback diverged at tick "
                f"{simulation.tick}: expected {expected}, got {observed}"
            )
        self.integrity_snapshots_verified += 1

    def _prepare_observer_history(self) -> None:
        simulation = BioFoundrySimulation(self.trace.config)
        tracker = SocietyDynamicsTracker()
        interval = max(1, self.trace.config.simulation.snapshot_interval // 4)
        tracker.record(simulation)
        pending_events: list[dict[str, Any]] = []
        keyframes: list[dict[str, Any]] = []

        initial = self.trace.recorded_snapshots.get(0)
        if initial is None:
            initial = simulation.snapshot(display_limit=2048)
        keyframes.append({"snapshot": initial, "events": []})
        self._verify_digest(simulation)

        for record in tqdm(
            self.trace.action_records,
            desc="Reconstructing playback index",
            unit="tick",
            leave=False,
        ):
            pending_events.extend(
                self._apply_record(simulation, record, policy=self.trace.policy)
            )
            if simulation.tick % interval == 0:
                tracker.record(simulation)
            if simulation.tick in self.trace.recorded_snapshots:
                self._verify_digest(simulation)
                keyframes.append(
                    {
                        "snapshot": self.trace.recorded_snapshots[simulation.tick],
                        "events": _priority_events(pending_events, MAX_KEYFRAME_EVENTS),
                    }
                )
                pending_events = []
        if not tracker.points or tracker.points[-1]["tick"] != simulation.tick:
            tracker.record(simulation)
        if not keyframes or int(keyframes[-1]["snapshot"]["tick"]) != simulation.tick:
            keyframes.append(
                {
                    "snapshot": simulation.snapshot(display_limit=2048),
                    "events": _priority_events(pending_events, MAX_KEYFRAME_EVENTS),
                }
            )
        self.dynamics_history = tracker.packet(sample_interval=interval)
        self.keyframes = keyframes

    def reset(self) -> dict[str, Any]:
        self.simulation = BioFoundrySimulation(self.trace.config)
        self.action_index = 0
        self.recent_events = []
        self.complete = self.trace.max_tick == 0
        return self.frame(events=[], seek=True)

    def frame(
        self,
        *,
        events: list[dict[str, Any]] | None = None,
        seek: bool = False,
    ) -> dict[str, Any]:
        return {
            "type": "frame",
            "snapshot": self.simulation.snapshot(display_limit=2048),
            "events": list(events or []),
            "playback_seek": bool(seek),
            "replay_checkpoint": bool(
                self.simulation.tick % self.trace.config.simulation.snapshot_interval == 0
            ),
        }

    def step(self) -> dict[str, Any] | None:
        if self.action_index >= len(self.trace.action_records):
            self.complete = True
            return None
        record = self.trace.action_records[self.action_index]
        events = self._apply_record(
            self.simulation,
            record,
            policy=self.trace.policy,
        )
        self.action_index += 1
        self.recent_events.extend(events)
        self.recent_events = self.recent_events[-MAX_RECENT_EVENTS:]
        self.complete = self.action_index >= len(self.trace.action_records)
        return self.frame(events=events)

    def seek(self, tick: int) -> dict[str, Any]:
        target = max(0, min(self.trace.max_tick, int(tick)))
        if target < self.simulation.tick:
            self.reset()
        while self.simulation.tick < target:
            if self.step() is None:
                break
        return self.frame(
            events=_priority_events(self.recent_events, MAX_KEYFRAME_EVENTS),
            seek=True,
        )

    def dynamics_point(self) -> dict[str, Any] | None:
        tick = int(self.simulation.tick)
        points = self.dynamics_history.get("points", [])
        matching = [point for point in points if int(point.get("tick", -1)) == tick]
        return dict(matching[-1]) if matching else None

    def manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": "offline-deterministic",
            "trace": str(self.trace.path),
            "sample_interval": self.trace.config.simulation.snapshot_interval,
            "frame_count": len(self.keyframes),
            "first_tick": 0,
            "last_tick": self.trace.max_tick,
            "complete_history": True,
            "exact_tick_seek": True,
            "integrity_snapshots_verified": self.integrity_snapshots_verified,
        }

    def control_state(self) -> dict[str, Any]:
        return {
            "paused": self.paused,
            "speed_multiplier": self.speed_multiplier,
            "step_budget": 0,
            "model_requests": self.trace.model_request_count,
            "model_errors": self.trace.model_error_count,
            "provider_attempts": 0,
            "provider_outage": False,
            "llm_enabled": False,
            "model": self.trace.model,
            "playback_mode": True,
            "playback_complete": self.complete,
            "max_tick": self.trace.max_tick,
            "trace_name": self.trace.path.name,
        }


class PlaybackServer:
    def __init__(self, trace: PlaybackTrace, *, ticks_per_second: float = 8.0):
        self.playback = TracePlayback(trace, ticks_per_second=ticks_per_second)
        self.hub = FrameHub()
        self.running = True
        self._lock = asyncio.Lock()

    async def _broadcast_frame(self, frame: dict[str, Any]) -> None:
        point = self.playback.dynamics_point()
        await self.hub.broadcast(
            {
                **frame,
                "control": self.playback.control_state(),
                **({"dynamics_point": point} if point is not None else {}),
            }
        )

    async def run(self) -> None:
        while self.running:
            if self.playback.paused:
                await asyncio.sleep(0.03)
                continue
            started = asyncio.get_running_loop().time()
            async with self._lock:
                frame = self.playback.step()
            if frame is None:
                self.playback.paused = True
                await self.hub.broadcast(
                    {
                        "type": "complete",
                        "snapshot": self.playback.simulation.snapshot(display_limit=2048),
                        "events": [],
                        "control": self.playback.control_state(),
                        "playback_seek": False,
                    }
                )
                continue
            await self._broadcast_frame(frame)
            if self.playback.complete:
                self.playback.paused = True
                await self.hub.broadcast(
                    {
                        "type": "complete",
                        "snapshot": self.playback.simulation.snapshot(display_limit=2048),
                        "events": frame.get("events", []),
                        "control": self.playback.control_state(),
                        "playback_seek": False,
                    }
                )
                continue
            period = 1.0 / (
                self.playback.base_ticks_per_second * self.playback.speed_multiplier
            )
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.0, period - elapsed))

    async def heartbeat(self) -> None:
        while self.running:
            await self.hub.broadcast(
                {
                    "type": "heartbeat",
                    "tick": self.playback.simulation.tick,
                    "control": self.playback.control_state(),
                }
            )
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    async def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = str(payload.get("command", ""))
        detail = ""
        frame: dict[str, Any] | None = None
        async with self._lock:
            if command == "toggle_pause":
                if self.playback.complete:
                    frame = self.playback.reset()
                self.playback.paused = not self.playback.paused
            elif command == "set_paused":
                self.playback.paused = bool(payload.get("paused", True))
            elif command == "set_speed":
                self.playback.speed_multiplier = max(
                    0.25, min(64.0, float(payload.get("speed_multiplier", 1.0)))
                )
            elif command == "step":
                self.playback.paused = True
                frame = self.playback.step()
            elif command == "seek":
                self.playback.paused = True
                frame = self.playback.seek(int(payload.get("tick", 0)))
            elif command == "play_from_start":
                frame = self.playback.reset()
                self.playback.paused = False
            else:
                detail = f"command {command!r} is unavailable in offline playback"
        if frame is not None:
            await self._broadcast_frame(frame)
        return {
            "type": "control",
            "control": self.playback.control_state(),
            "detail": detail,
        }

    def close(self) -> None:
        self.running = False


def create_playback_app(
    path: str | Path,
    *,
    ticks_per_second: float = 8.0,
) -> FastAPI:
    trace = PlaybackTrace.load(path)
    server = PlaybackServer(trace, ticks_per_second=ticks_per_second)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        playback_task = asyncio.create_task(server.run())
        heartbeat_task = asyncio.create_task(server.heartbeat())
        try:
            yield
        finally:
            server.close()
            playback_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await playback_task
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    app = FastAPI(title="BioFoundry Offline Playback", version=__version__, lifespan=lifespan)
    app.state.playback = server

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "offline-playback",
            "tick": server.playback.simulation.tick,
            "agents": server.playback.simulation.population.size,
            "artifacts": server.playback.simulation.artifacts.count,
            "viewers": len(server.hub.clients),
            "engine_revision": ENGINE_REVISION,
            "replay": server.playback.manifest(),
            **server.playback.control_state(),
        }

    @app.get("/snapshot")
    async def snapshot() -> dict[str, Any]:
        return server.playback.simulation.snapshot(display_limit=2048)

    @app.get("/history")
    async def history() -> dict[str, Any]:
        return server.playback.dynamics_history

    @app.get("/replay/manifest")
    async def replay_manifest() -> dict[str, Any]:
        return server.playback.manifest()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await server.hub.add(websocket)
        await server.hub.send(
            websocket,
            {
                "type": "snapshot",
                "snapshot": server.playback.simulation.snapshot(display_limit=2048),
                "events": [],
                "control": server.playback.control_state(),
                "dynamics_history": server.playback.dynamics_history,
                "replay_manifest": server.playback.manifest(),
                "playback_seek": True,
            },
        )
        for frame in server.playback.keyframes:
            sent = await server.hub.send(
                websocket,
                {
                    "type": "replay_frame",
                    "frame": frame,
                    "replay_manifest": server.playback.manifest(),
                },
            )
            if not sent:
                return
        await server.hub.send(
            websocket,
            {"type": "replay_ready", "replay_manifest": server.playback.manifest()},
        )
        try:
            while True:
                payload: dict[str, Any]
                try:
                    payload = await websocket.receive_json()
                except ValueError:
                    payload = {}
                await server.hub.broadcast(await server.handle_command(payload))
        except WebSocketDisconnect:
            server.hub.remove(websocket)
        except Exception:
            server.hub.remove(websocket)

    return app
