"""GameSession: a LiveGame subclass with a possessable human slot and
decision-window pacing. Everything here is additive — the engine, the stock
server, and the trace format are used exactly as shipped.

Pacing (docs/HUMAN_PLAY.md section 2): each tick, if a player is joined and has
no queued input, the loop optionally waits for input — up to
``input_timeout_seconds`` per window, or indefinitely when ``require_response``
is set (turn-based play). Wall-clock waiting never enters the engine, so the
recorded trace replays deterministically with the standard tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from biofoundry import ENGINE_REVISION
from biofoundry import __version__ as engine_version
from biofoundry.config import GameConfig
from biofoundry.policies.llm import LLMPolicy
from biofoundry.server import LiveGame
from biofoundry.types import AgentAction

from .human import HumanSidecar, HumanSlot
from .policy import HumanAwareLLMPolicy
from .settings import GameSettings


class GameSession(LiveGame):
    def __init__(
        self,
        config: GameConfig,
        settings: GameSettings,
        record_path: str | Path | None = None,
    ):
        super().__init__(config, record_path=record_path)
        settings.validate()
        self.settings = settings
        if settings.human_agent not in self.simulation.population.id_to_index:
            raise ValueError(
                f"game.human_agent {settings.human_agent!r} is not in the population "
                f"(agents: {self.simulation.population.size})"
            )
        self.human = HumanSlot(settings.human_agent, settings.queue_limit)
        if isinstance(self.policy, LLMPolicy) and not isinstance(
            self.policy, HumanAwareLLMPolicy
        ):
            self.policy = HumanAwareLLMPolicy.adopt(self.policy)
        self.sidecar = (
            HumanSidecar(
                record_path,
                {
                    "settings": settings.as_dict(),
                    "trace": str(record_path),
                    "engine_revision": ENGINE_REVISION,
                    "engine_version": engine_version,
                    "policy": "llm" if isinstance(self.policy, LLMPolicy) else "scripted",
                },
            )
            if record_path is not None
            else None
        )

    # ------------------------------------------------------------------ state

    def game_state_payload(self) -> dict[str, Any]:
        return {
            "human_agent": self.human.agent_id,
            "joined": self.human.joined,
            "pending_inputs": self.human.pending,
            "settings": self.settings.as_dict(),
            "tick": self.simulation.tick,
        }

    def human_observation(self) -> dict[str, Any] | None:
        """The possessed agent's full information-gated observation packet."""

        if not self.human.joined:
            return None
        index = self.simulation.population.id_to_index[self.human.agent_id]
        try:
            return json.loads(self.simulation.semantic_observation(index, compact=True))
        except Exception:
            return None

    def _sidecar_write(self, record: dict[str, Any]) -> None:
        if self.sidecar is not None:
            self.sidecar.write(record)

    # --------------------------------------------------------------- commands

    def handle_game_command(
        self, payload: dict[str, Any], connection_id: str
    ) -> dict[str, Any]:
        """Handle one client command; returns the reply for the sending client.

        Replies of type ``control`` mirror the stock server and should be
        broadcast by the caller; other reply types are addressed to the sender.
        A truthy ``game_state_changed`` asks the caller to broadcast fresh
        game state to every client.
        """

        command = str(payload.get("command", ""))
        if command == "join":
            accepted, detail = self.human.join(connection_id)
            if accepted:
                if isinstance(self.policy, HumanAwareLLMPolicy):
                    self.policy.excluded_agents.add(self.human.agent_id)
                self._sidecar_write(
                    {"type": "join", "tick": self.simulation.tick, "connection": connection_id}
                )
            return {
                "type": "join_ack",
                "accepted": accepted,
                "agent": self.human.agent_id if accepted else "",
                "detail": detail,
                "game": self.game_state_payload(),
                "game_state_changed": accepted,
            }
        if command == "release":
            released = self.human.release(connection_id)
            if released:
                if isinstance(self.policy, HumanAwareLLMPolicy):
                    self.policy.excluded_agents.discard(self.human.agent_id)
                self._sidecar_write(
                    {"type": "release", "tick": self.simulation.tick, "connection": connection_id}
                )
            return {
                "type": "release_ack",
                "accepted": released,
                "detail": "" if released else "not the possessing connection",
                "game": self.game_state_payload(),
                "game_state_changed": released,
            }
        if command == "human_action":
            request_id = str(payload.get("request_id", ""))[:64]
            raw_action = payload.get("action")
            if not isinstance(raw_action, Mapping):
                return {
                    "type": "action_ack",
                    "request_id": request_id,
                    "accepted": False,
                    "detail": "action must be a mapping of AgentAction fields",
                }
            try:
                normalized = dict(raw_action)
                if self.simulation.scenario is not None:
                    normalized = self.simulation.scenario.normalize_action_mapping(normalized)
                action = AgentAction.from_value(normalized)
            except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
                return {
                    "type": "action_ack",
                    "request_id": request_id,
                    "accepted": False,
                    "detail": str(exc)[:240],
                }
            accepted, detail = self.human.submit(connection_id, action, request_id)
            if accepted and self.paused:
                # Mirror the stock manual-action behavior: acting while paused
                # advances exactly one tick.
                self.step_budget += 1
            return {
                "type": "action_ack",
                "request_id": request_id,
                "accepted": accepted,
                "detail": detail,
                "pending_inputs": self.human.pending,
            }
        if command == "manual_action":
            agent_id = str(payload.get("agent", ""))
            if self.human.joined and agent_id == self.human.agent_id:
                return {
                    "type": "control",
                    "control": self.control_state(),
                    "detail": "agent is possessed by the human player",
                }
            if not isinstance(payload.get("action", {}), Mapping):
                # The stock handler raises AttributeError on non-mapping input;
                # reject here so the connection survives malformed payloads.
                return {
                    "type": "control",
                    "control": self.control_state(),
                    "detail": "action must be a mapping of AgentAction fields",
                }
            return super().handle_command(payload)
        reply = super().handle_command(payload)
        if command in {"toggle_pause", "set_paused", "step", "set_speed"}:
            self._sidecar_write(
                {"type": "control", "command": command, "tick": self.simulation.tick}
            )
        return reply

    def release_connection(self, connection_id: str) -> bool:
        """Force-release the slot when its possessing client disconnects."""

        released = self.human.release(connection_id)
        if released:
            if isinstance(self.policy, HumanAwareLLMPolicy):
                self.policy.excluded_agents.discard(self.human.agent_id)
            self._sidecar_write(
                {
                    "type": "release",
                    "tick": self.simulation.tick,
                    "connection": connection_id,
                    "reason": "disconnect",
                }
            )
        return released

    # ------------------------------------------------------------ tick pacing

    def _window_due(self) -> bool:
        settings = self.settings
        if not self.human.joined or not self.human.is_empty:
            return False
        if not (settings.require_response or settings.input_timeout_seconds > 0):
            return False
        return self.simulation.tick % settings.decision_interval == 0

    async def _await_human(self) -> None:
        settings = self.settings
        deadline = (
            None
            if settings.require_response
            else time.perf_counter() + settings.input_timeout_seconds
        )
        observation = self.human_observation() or {}
        await self.hub.broadcast(
            {
                "type": "decision_prompt",
                "agent": self.human.agent_id,
                "tick": self.simulation.tick,
                "deadline_ms": (
                    None
                    if deadline is None
                    else int(settings.input_timeout_seconds * 1000)
                ),
                "require_response": settings.require_response,
                "affordances": observation.get("local_affordances"),
            }
        )
        while self.running and self.human.joined and self.human.is_empty and not self.paused:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.human.input_event.wait(), timeout=0.1)

    async def _tick_once(self) -> bool:
        """Advance at most one world tick; returns whether the world advanced."""

        period = 1.0 / (self.config.server.ticks_per_second * self.speed_multiplier)
        started = time.perf_counter()
        if self._window_due():
            await self._await_human()
        macroturn_agents: list[str] = []
        if isinstance(self.policy, LLMPolicy):
            macroturn_agents = (
                self.policy.visible_scheduled(self.simulation)
                if isinstance(self.policy, HumanAwareLLMPolicy)
                else self.simulation.scheduled_macro_agents()
            )
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
                await self.hub.broadcast({**outage, "control": self.control_state()})
                await asyncio.sleep(self.config.llm.provider_retry_seconds)
                return False
            actions = self.policy.next_actions(self.simulation)
        else:
            actions = self.policy.actions(self.simulation)
            if self.human.joined:
                # A possessed slot is never driven by the scripted policy; an
                # absent entry is a WAIT to the engine and to replay alike.
                actions.pop(self.human.agent_id, None)
        if self.manual_actions:
            actions.update(self.manual_actions)
            self.manual_actions.clear()
        human_applied: dict[str, Any] | None = None
        popped = self.human.pop() if self.human.joined else None
        if popped is not None:
            action, request_id = popped
            actions[self.human.agent_id] = action
            human_applied = {
                "tick": self.simulation.tick,
                "agent": self.human.agent_id,
                "request_id": request_id,
                "action": action.as_dict(),
            }
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
            # Durability for interactive sessions: the stock recorder flushes
            # only on snapshots. Flushing its public handle keeps the tail of a
            # killed session replayable without modifying the recorder.
            with contextlib.suppress(Exception):
                self.recorder.handle.flush()
        if human_applied is not None:
            self._sidecar_write({"type": "human_action", **human_applied})
        result = self.simulation.step(actions)
        display_events.extend(event.as_dict() for event in result.events)
        if human_applied is not None:
            display_events.append(
                {
                    "tick": human_applied["tick"],
                    "kind": "human_intervention",
                    "payload": human_applied,
                }
            )
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
            dynamics_point = self.dynamics.record(self.simulation, presentation_snapshot)
            await self.hub.broadcast(
                {
                    "type": "frame",
                    "snapshot": presentation_snapshot,
                    "events": display_events[-256:],
                    "control": self.control_state(),
                    "dynamics_point": dynamics_point,
                    "replay_checkpoint": replay_checkpoint,
                    "game": self.game_state_payload(),
                    "human_observation": self.human_observation(),
                }
            )
        elapsed = time.perf_counter() - started
        await asyncio.sleep(max(0.0, period - elapsed))
        return True

    async def run(self) -> None:
        while self.running and self.simulation.tick < self.config.simulation.max_ticks:
            if self.paused and self.step_budget <= 0:
                await asyncio.sleep(0.05)
                continue
            await self._tick_once()
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
                "game": self.game_state_payload(),
            }
        )

    def close(self) -> None:
        super().close()
        self.human.release(None)
        if self.sidecar is not None:
            self.sidecar.write({"type": "session_end", "tick": self.simulation.tick})
            self.sidecar.close()
