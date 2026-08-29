"""FastAPI/WebSocket bridge from the authoritative simulator to Godot."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import ENGINE_REVISION, __version__
from .capabilities import addressing_available, available_action_types, replies_available
from .config import GameConfig
from .dynamics import SocietyDynamicsTracker
from .events import EventRecorder, canonical_json
from .policies.llm import LLMPolicy
from .policies.scripted import (
    OracleResearchSocietyPolicy,
    ScenarioIndustryPolicy,
    ScriptedBioFoundryPolicy,
)
from .providers.openai_compatible import OpenAICompatibleProvider
from .scenarios import scenario_metadata
from .simulation import BioFoundrySimulation
from .types import AgentAction

HEARTBEAT_INTERVAL_SECONDS = 2.0
REPLAY_PROTOCOL_VERSION = 1
MAX_REPLAY_FRAMES = 256
MAX_REPLAY_EVENTS_PER_FRAME = 512
MAX_PENDING_REPLAY_EVENTS = 2048


class FrameHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self._send_lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)

    def remove(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def send(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        message = canonical_json(payload)
        async with self._send_lock:
            try:
                await websocket.send_text(message)
            except Exception:
                self.remove(websocket)
                return False
        return True

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        stale: list[WebSocket] = []
        message = canonical_json(payload)
        async with self._send_lock:
            for client in tuple(self.clients):
                try:
                    await client.send_text(message)
                except Exception:
                    stale.append(client)
        for client in stale:
            self.remove(client)


class LiveGame:
    def __init__(self, config: GameConfig, record_path: str | Path | None = None):
        self.config = config
        self.simulation = BioFoundrySimulation(config)
        scenario = self.simulation.scenario
        if config.llm.enabled:
            provider = (
                OpenAICompatibleProvider(
                    config.llm,
                    allowed_action_types=available_action_types(config),
                    allow_addressing=addressing_available(config),
                    allow_replies=replies_available(config),
                    resource_names=scenario.resource_names if scenario is not None else None,
                    operation_names=scenario.operation_names if scenario is not None else None,
                )
                if scenario is not None
                else OpenAICompatibleProvider(config.llm)
            )
            self.policy = LLMPolicy(provider)
        else:
            self.policy = (
                ScenarioIndustryPolicy()
                if scenario is not None
                else (
                    OracleResearchSocietyPolicy()
                    if config.science.enabled
                    else ScriptedBioFoundryPolicy()
                )
            )
        self.hub = FrameHub()
        self.running = True
        self.paused = False
        self.step_budget = 0
        self.speed_multiplier = 1.0
        self.manual_actions: dict[str, AgentAction] = {}
        self.model_requests = 0
        self.model_errors = 0
        self.dynamics = SocietyDynamicsTracker()
        self.replay_interval = max(1, int(config.simulation.snapshot_interval))
        self.replay_frames: list[dict[str, Any]] = []
        self._pending_replay_events: list[dict[str, Any]] = []
        policy_name = (
            "llm"
            if isinstance(self.policy, LLMPolicy)
            else (
                "research-oracle"
                if isinstance(self.policy, OracleResearchSocietyPolicy)
                else "scripted"
            )
        )
        record_metadata = {
            "config": config.as_dict(),
            "engine_revision": ENGINE_REVISION,
            "mode": "live_server",
            "policy": policy_name,
            "version": __version__,
        }
        if scenario is not None:
            record_metadata["scenario"] = scenario_metadata(scenario)
        self.recorder = (
            EventRecorder(
                record_path,
                record_metadata,
                compression=config.trace.compression,
                deduplicate_prompts=config.trace.deduplicate_prompts,
            )
            if record_path is not None
            else None
        )
        initial_snapshot = self.simulation.snapshot(display_limit=None)
        self.dynamics.record(self.simulation, initial_snapshot)
        self._record_replay_frame(
            self.simulation.snapshot(display_limit=2048), force=True
        )
        if self.recorder is not None:
            self.recorder.write_snapshot(
                initial_snapshot,
                state_digest=self.simulation.state_digest(),
            )

    def replay_manifest(self) -> dict[str, Any]:
        ticks = [int(frame["snapshot"]["tick"]) for frame in self.replay_frames]
        return {
            "version": REPLAY_PROTOCOL_VERSION,
            "sample_interval": self.replay_interval,
            "frame_count": len(self.replay_frames),
            "first_tick": ticks[0] if ticks else self.simulation.tick,
            "last_tick": ticks[-1] if ticks else self.simulation.tick,
            "complete_history": bool(not ticks or ticks[0] == 0),
        }

    def _queue_replay_events(self, events: list[dict[str, Any]]) -> None:
        self._pending_replay_events.extend(events)
        if len(self._pending_replay_events) > MAX_PENDING_REPLAY_EVENTS:
            self._pending_replay_events = self._pending_replay_events[
                -MAX_PENDING_REPLAY_EVENTS:
            ]

    def _compact_replay_history(self) -> None:
        while len(self.replay_frames) >= MAX_REPLAY_FRAMES:
            self.replay_interval *= 2
            self.replay_frames = [
                frame
                for frame in self.replay_frames
                if int(frame["snapshot"]["tick"]) == 0
                or int(frame["snapshot"]["tick"]) % self.replay_interval == 0
            ]

    def _record_replay_frame(
        self,
        snapshot: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> bool:
        tick = int(self.simulation.tick)
        if not force and tick % self.replay_interval:
            return False
        self._compact_replay_history()
        frame = {
            "snapshot": snapshot
            if snapshot is not None
            else self.simulation.snapshot(display_limit=2048),
            "events": self._pending_replay_events[-MAX_REPLAY_EVENTS_PER_FRAME:],
        }
        self._pending_replay_events = []
        if (
            self.replay_frames
            and int(self.replay_frames[-1]["snapshot"]["tick"]) == tick
        ):
            prior_events = list(self.replay_frames[-1].get("events", []))
            frame["events"] = (prior_events + frame["events"])[
                -MAX_REPLAY_EVENTS_PER_FRAME:
            ]
            self.replay_frames[-1] = frame
        else:
            self.replay_frames.append(frame)
        return True

    def control_state(self) -> dict[str, Any]:
        return {
            "paused": self.paused,
            "speed_multiplier": self.speed_multiplier,
            "step_budget": self.step_budget,
            "model_requests": self.model_requests,
            "model_errors": self.model_errors,
            "provider_outage": bool(
                isinstance(self.policy, LLMPolicy) and self.policy.provider_outage
            ),
            "provider_attempts": (
                self.policy.provider_attempts
                if isinstance(self.policy, LLMPolicy)
                else 0
            ),
            "llm_enabled": isinstance(self.policy, LLMPolicy),
            "model": self.config.llm.model if isinstance(self.policy, LLMPolicy) else "scripted",
        }

    def handle_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = str(payload.get("command", ""))
        detail = ""
        if command == "toggle_pause":
            self.paused = not self.paused
        elif command == "set_paused":
            self.paused = bool(payload.get("paused", True))
        elif command == "step":
            self.paused = True
            self.step_budget += 1
        elif command == "set_speed":
            self.speed_multiplier = max(
                0.25, min(8.0, float(payload.get("speed_multiplier", 1.0)))
            )
        elif command == "manual_action":
            agent_id = str(payload.get("agent", ""))
            if agent_id not in self.simulation.population.id_to_index:
                detail = f"unknown agent {agent_id}"
            else:
                try:
                    raw_action = payload.get("action", {})
                    if self.simulation.scenario is not None and isinstance(
                        raw_action, Mapping
                    ):
                        raw_action = self.simulation.scenario.normalize_action_mapping(
                            dict(raw_action)
                        )
                    self.manual_actions[agent_id] = AgentAction.from_value(
                        raw_action
                    )
                    if self.paused:
                        self.step_budget += 1
                except (TypeError, ValueError, KeyError, IndexError) as exc:
                    detail = str(exc)[:240]
        else:
            detail = f"unknown command {command}"
        return {
            "type": "control",
            "control": self.control_state(),
            "detail": detail,
        }

    async def run(self) -> None:
        while self.running and self.simulation.tick < self.config.simulation.max_ticks:
            if self.paused and self.step_budget <= 0:
                await asyncio.sleep(0.05)
                continue
            period = 1.0 / (
                self.config.server.ticks_per_second * self.speed_multiplier
            )
            started = time.perf_counter()
            macroturn_agents: list[str] = []
            if isinstance(self.policy, LLMPolicy):
                # The policy mutates authoritative scheduling state before the world
                # step. Persist the exact scheduled set so live recordings replay it.
                macroturn_agents = self.simulation.scheduled_macro_agents()
                committed = await self.policy.refresh_plans(self.simulation)
                if not committed:
                    errors: list[dict[str, Any]] = []
                    for trace in self.policy.last_traces:
                        self.model_requests += 1
                        self.model_errors += 1
                        errors.append(
                            {
                                "agent": trace.get("agent", ""),
                                "error": str(trace.get("error", ""))[:280],
                            }
                        )
                        if self.recorder is not None:
                            self.recorder.write_record({"type": "model_trace", **trace})
                    outage = {
                        "type": "provider_outage",
                        "tick": self.simulation.tick,
                        "macroturn_agents": macroturn_agents,
                        "errors": errors,
                        "world_advanced": False,
                    }
                    if self.recorder is not None:
                        self.recorder.write_record(outage)
                    await self.hub.broadcast(
                        {**outage, "control": self.control_state()}
                    )
                    await asyncio.sleep(self.config.llm.provider_retry_seconds)
                    continue
                actions = self.policy.next_actions(self.simulation)
            else:
                actions = self.policy.actions(self.simulation)
            if self.manual_actions:
                actions.update(self.manual_actions)
                self.manual_actions.clear()
            if self.paused and self.step_budget > 0:
                self.step_budget -= 1
            display_events: list[dict[str, Any]] = []
            if self.recorder is not None:
                if isinstance(self.policy, LLMPolicy):
                    for trace in self.policy.last_traces:
                        self.recorder.write_record({"type": "model_trace", **trace})
                self.recorder.write_record(
                    {
                        "type": "actions",
                        "tick": self.simulation.tick,
                        "macroturn_agents": macroturn_agents,
                        "actions": {
                            agent_id: action.as_dict()
                            for agent_id, action in actions.items()
                        },
                    }
                )
            result = self.simulation.step(actions)
            display_events.extend(event.as_dict() for event in result.events)
            if isinstance(self.policy, LLMPolicy):
                for trace in self.policy.last_traces:
                    self.model_requests += 1
                    first_action = (trace.get("actions") or [{}])[0]
                    error = str(trace.get("error", ""))
                    if error:
                        self.model_errors += 1
                    display_events.append(
                        {
                            "tick": trace.get("tick", self.simulation.tick - 1),
                            "kind": "model_error" if error else "agent_deliberated",
                            "payload": {
                                "agent": trace.get("agent", ""),
                                "verb": first_action.get("verb", 0),
                                "error": error[:280],
                            },
                        }
                    )
            if self.recorder is not None:
                self.recorder.write_events(result.events)
                if self.simulation.tick % self.config.simulation.snapshot_interval == 0:
                    self.recorder.write_snapshot(
                        self.simulation.snapshot(display_limit=None),
                        state_digest=self.simulation.state_digest(),
                    )
            self._queue_replay_events(display_events)
            replay_checkpoint = self._record_replay_frame()
            if self.simulation.tick % self.config.server.render_every == 0:
                presentation_snapshot = self.simulation.snapshot(display_limit=2048)
                dynamics_point = self.dynamics.record(
                    self.simulation, presentation_snapshot
                )
                await self.hub.broadcast(
                    {
                        "type": "frame",
                        "snapshot": presentation_snapshot,
                        "events": display_events[-256:],
                        "control": self.control_state(),
                        "dynamics_point": dynamics_point,
                        "replay_checkpoint": replay_checkpoint,
                    }
                )
            elapsed = time.perf_counter() - started
            await asyncio.sleep(max(0.0, period - elapsed))
        final_snapshot = self.simulation.snapshot(display_limit=2048)
        self._record_replay_frame(final_snapshot, force=True)
        final_dynamics = (
            self.dynamics.points[-1]
            if self.dynamics.points
            and self.dynamics.points[-1]["tick"] == self.simulation.tick
            else self.dynamics.record(self.simulation, final_snapshot)
        )
        await self.hub.broadcast(
            {
                "type": "complete",
                "snapshot": final_snapshot,
                "control": self.control_state(),
                "dynamics_point": final_dynamics,
                "replay_checkpoint": True,
            }
        )

    async def heartbeat(self) -> None:
        """Keep viewers informed even while an LLM macroturn blocks world ticks."""
        while self.running:
            await self.hub.broadcast(
                {
                    "type": "heartbeat",
                    "tick": self.simulation.tick,
                    "control": self.control_state(),
                }
            )
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    def close(self) -> None:
        self.running = False
        if self.recorder is not None:
            self.recorder.write_snapshot(
                self.simulation.snapshot(display_limit=None),
                state_digest=self.simulation.state_digest(),
            )
            self.recorder.close()


def create_app(config: GameConfig, record_path: str | Path | None = None) -> FastAPI:
    live = LiveGame(config, record_path=record_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        game_task = asyncio.create_task(live.run())
        heartbeat_task = asyncio.create_task(live.heartbeat())
        try:
            yield
        finally:
            live.close()
            game_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await game_task
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    app = FastAPI(title="BioFoundry World", version="0.1.0", lifespan=lifespan)
    app.state.live = live

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tick": live.simulation.tick,
            "agents": live.simulation.population.size,
            "artifacts": live.simulation.artifacts.count,
            "viewers": len(live.hub.clients),
            "replay": live.replay_manifest(),
            **live.control_state(),
        }

    @app.get("/snapshot")
    async def snapshot() -> dict[str, Any]:
        return live.simulation.snapshot(display_limit=2048)

    @app.get("/history")
    async def history() -> dict[str, Any]:
        return live.dynamics.packet(sample_interval=live.config.server.render_every)

    @app.get("/replay/manifest")
    async def replay_manifest() -> dict[str, Any]:
        return live.replay_manifest()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await live.hub.add(websocket)
        await live.hub.send(
            websocket,
            {
                "type": "snapshot",
                "snapshot": live.simulation.snapshot(display_limit=2048),
                "control": live.control_state(),
                "dynamics_history": live.dynamics.packet(
                    sample_interval=live.config.server.render_every
                ),
                "replay_manifest": live.replay_manifest(),
            },
        )
        for frame in tuple(live.replay_frames):
            sent = await live.hub.send(
                websocket,
                {
                    "type": "replay_frame",
                    "frame": frame,
                    "replay_manifest": live.replay_manifest(),
                },
            )
            if not sent:
                return
        await live.hub.send(
            websocket,
            {
                "type": "replay_ready",
                "replay_manifest": live.replay_manifest(),
            },
        )
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    await live.hub.broadcast(live.handle_command(payload))
        except WebSocketDisconnect:
            live.hub.remove(websocket)
        except Exception:
            live.hub.remove(websocket)

    return app
