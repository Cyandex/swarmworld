"""Entry point for the human-playable game server.

Run from the repository root (or anywhere — the path bootstrap below finds the
repo):

    python game/serve.py --config game/configs/arcade-scripted.yaml

Then open the printed URL. The canonical trace and the human-provenance sidecar
are written under ``runs/game/`` unless ``--no-record`` is given.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="SwarmWorld human-play game server")
    parser.add_argument(
        "--config",
        required=True,
        help="game profile YAML (a standard biofoundry config plus a game: section)",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="trace output path (default: runs/game/<profile>-<timestamp>.jsonl[.gz])",
    )
    parser.add_argument(
        "--no-record", action="store_true", help="disable trace and sidecar recording"
    )
    parser.add_argument("--host", default=None, help="override server.host")
    parser.add_argument("--port", type=int, default=None, help="override server.port")
    args = parser.parse_args()

    import uvicorn

    from game.app import create_game_app
    from game.settings import load_game_profile

    config, settings = load_game_profile(args.config)
    if args.host is not None:
        config.server.host = args.host
    if args.port is not None:
        config.server.port = args.port

    record_path: Path | None
    if args.no_record:
        record_path = None
    elif args.record is not None:
        record_path = Path(args.record)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = ".jsonl.gz" if config.trace.compression == "gzip" else ".jsonl"
        record_path = (
            REPO_ROOT / "runs" / "game" / f"{Path(args.config).stem}-{stamp}{suffix}"
        )

    app = create_game_app(config, settings, record_path=record_path)

    host, port = config.server.host, config.server.port
    print(f"SwarmWorld game server on http://{host}:{port}/play/")
    print(f"  websocket:            ws://{host}:{port}/ws")
    print("  spectator observatory: cd web && npm run dev  (same ws URL)")
    print(f"  human agent slot:     {settings.human_agent}")
    mode = (
        "turn-based (require_response)"
        if settings.require_response
        else (
            f"prompted ({settings.input_timeout_seconds:g}s window)"
            if settings.input_timeout_seconds > 0
            else "real-time (never waits)"
        )
    )
    print(f"  pacing:               {mode}")
    if record_path is not None:
        print(f"  trace:                {record_path}")
        print(f"  human sidecar:        {record_path.name}.human.jsonl")
    else:
        print("  recording:            disabled")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
