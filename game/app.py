"""FastAPI application for the human-playable game server.

Serves the same protocol-v1 WebSocket surface as the stock server (the Three.js
Observatory can attach as a spectator), plus the no-build canvas client at
``/play/`` and the game commands ``join`` / ``release`` / ``human_action``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from biofoundry.config import GameConfig

from .session import GameSession
from .settings import GameSettings

CLIENT_DIR = Path(__file__).resolve().parent / "client"


def create_game_app(
    config: GameConfig,
    settings: GameSettings,
    record_path: str | Path | None = None,
) -> FastAPI:
    session = GameSession(config, settings, record_path=record_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        game_task = asyncio.create_task(session.run())
        heartbeat_task = asyncio.create_task(session.heartbeat())
        try:
            yield
        finally:
            session.close()
            game_task.cancel()
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await game_task
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task

    app = FastAPI(title="SwarmWorld Game", version="0.1.0", lifespan=lifespan)
    app.state.session = session

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/play/")

    if CLIENT_DIR.is_dir():
        app.mount("/play", StaticFiles(directory=CLIENT_DIR, html=True), name="play")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "tick": session.simulation.tick,
            "agents": session.simulation.population.size,
            "artifacts": session.simulation.artifacts.count,
            "viewers": len(session.hub.clients),
            "replay": session.replay_manifest(),
            "game": session.game_state_payload(),
            **session.control_state(),
        }

    @app.get("/snapshot")
    async def snapshot() -> dict[str, Any]:
        return session.simulation.snapshot(display_limit=2048)

    @app.get("/history")
    async def history() -> dict[str, Any]:
        return session.dynamics.packet(
            sample_interval=session.config.server.render_every
        )

    @app.get("/replay/manifest")
    async def replay_manifest() -> dict[str, Any]:
        return session.replay_manifest()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        connection_id = uuid4().hex
        await session.hub.add(websocket)
        await session.hub.send(
            websocket,
            {
                "type": "snapshot",
                "snapshot": session.simulation.snapshot(display_limit=2048),
                "control": session.control_state(),
                "dynamics_history": session.dynamics.packet(
                    sample_interval=session.config.server.render_every
                ),
                "replay_manifest": session.replay_manifest(),
                "game": session.game_state_payload(),
            },
        )
        for frame in tuple(session.replay_frames):
            sent = await session.hub.send(
                websocket,
                {
                    "type": "replay_frame",
                    "frame": frame,
                    "replay_manifest": session.replay_manifest(),
                },
            )
            if not sent:
                return
        await session.hub.send(
            websocket,
            {
                "type": "replay_ready",
                "replay_manifest": session.replay_manifest(),
                "game": session.game_state_payload(),
            },
        )
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                if not isinstance(payload, dict):
                    continue
                reply = session.handle_game_command(payload, connection_id)
                state_changed = bool(reply.pop("game_state_changed", False))
                if reply.get("type") == "control":
                    await session.hub.broadcast(reply)
                else:
                    await session.hub.send(websocket, reply)
                if state_changed:
                    await session.hub.broadcast(
                        {"type": "game_state", **session.game_state_payload()}
                    )
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            session.hub.remove(websocket)
            if session.release_connection(connection_id):
                await session.hub.broadcast(
                    {"type": "game_state", **session.game_state_payload()}
                )

    return app
