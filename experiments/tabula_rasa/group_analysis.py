"""Do trained agents stay together? Measures spatial grouping of a trained run.

Agents cannot see each other; they can only hear calls (language on) or be fed (sharing).
Runs without language are therefore blind to each other, and any clustering there comes
from shared food patches alone. Grouping beyond that baseline is social behaviour.

Every tick, for the living agents we record the distance to their nearest living
neighbour. It is compared with the same number of agents placed uniformly at random on
walkable tiles, which corrects for population size (fewer survivors -> larger gaps).
Agents spawn close together, so even random walkers score above 1 on that index; the
fair comparison is against the random/scripted baselines and the blind (no-language)
runs, not against 1.

Usage:
    python experiments/tabula_rasa/group_analysis.py runs/A_poison runs/B_language ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabula_rasa as tr  # noqa: E402


def nearest_neighbour(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    d = np.abs(xs[:, None] - xs[None, :]) + np.abs(ys[:, None] - ys[None, :])
    np.fill_diagonal(d, 10**6)
    return d.min(axis=1)


def measure(policy: str, cfg: dict, seeds: list[int], model=None) -> dict:
    rng = np.random.default_rng(0)
    nn_obs, nn_null, contact, hearing = [], [], [], []
    for seed in seeds:
        world = tr.World(seed, cfg["ticks"], poison=cfg["poison"], language=cfg["language"],
                         sharing=cfg.get("sharing", False))
        walk_y, walk_x = np.nonzero(world.sim.world.walkable)
        policy_rng = np.random.default_rng(seed)
        done = False
        while not done:
            alive = world.alive()
            pop = world.sim.population
            if alive.sum() >= 2:
                xs, ys = pop.x[alive].astype(int), pop.y[alive].astype(int)
                nn = nearest_neighbour(xs, ys)
                nn_obs.append(nn.mean())
                contact.append((nn <= 1).mean())
                hearing.append((nn <= tr.HEAR_RADIUS).mean())
                pick = rng.integers(0, len(walk_x), alive.sum())
                nn_null.append(nearest_neighbour(walk_x[pick], walk_y[pick]).mean())
            symbols = np.zeros(world.n, dtype=np.int64)
            if policy == "gpt":
                with torch.no_grad():
                    act_d, sym_d, _ = tr.distributions(model, torch.from_numpy(world.contexts()),
                                                       cfg["language"] != "off")
                acts = act_d.sample().numpy()
                if sym_d is not None:
                    symbols = sym_d.sample().numpy()
            elif policy == "random":
                acts = policy_rng.integers(0, tr.N_ACTIONS, world.n)
            else:
                acts = tr.scripted_actions(world, policy_rng)
            _, done, _ = world.step(np.where(alive, acts, 0), symbols)
    obs, null = float(np.mean(nn_obs)), float(np.mean(nn_null))
    return {
        "mean_nearest_neighbour": round(obs, 2),
        "random_placement_baseline": round(null, 2),
        "grouping_index": round(null / obs, 3),  # >1: closer together than chance
        "share_in_contact_(<=1)": round(float(np.mean(contact)), 3),
        "share_within_hearing_(<=6)": round(float(np.mean(hearing)), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--baselines", action="store_true", help="also measure random and scripted")
    args = ap.parse_args()
    torch.manual_seed(0)
    seeds = list(range(1000, 1000 + args.seeds))
    results = {}
    for run in map(Path, args.runs):
        cfg = json.loads((run / "report.json").read_text())["config"]
        tr.configure(cfg.get("sharing", False))
        model = tr.TinyGPT()
        model.load_state_dict(torch.load(run / "tinygpt.pt"))
        model.eval()
        results[run.name] = measure("gpt", cfg, seeds, model)
        print(run.name, results[run.name], flush=True)
    if args.baselines:
        cfg = {"ticks": 400, "poison": True, "language": "off"}
        tr.configure(False)
        for policy in ("random", "scripted"):
            results[policy] = measure(policy, cfg, seeds)
            print(policy, results[policy], flush=True)
    out = Path(args.runs[0]).parent / "group_analysis.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
