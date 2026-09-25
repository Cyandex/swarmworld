"""A gradient-free agent in the spirit of Netta (github.com/Cyandex/netta).

Netta learns language without gradients: it counts what it has actually lived, backs off
from long to short contexts, keeps a fixed share of honest ignorance (epsilon = 0.1) and
never invents continuations it has not seen. This file applies the same principles to
survival in SwarmWorld, in the same world, senses, reward and training budget as the
TinyGPT/PPO runs in tabula_rasa.py:

  * Memory is a table of counts, nothing else: for every lived (situation, action) the
    number of times it happened and the sum of the discounted return that followed
    (+0.01 per tick alive, -1 on death, gamma 0.99 - the PPO reward).
  * Situations are read at three levels of detail (full current frame, a coarse frame,
    tile + inventory) plus an empty context. The value of an action backs off like
    Netta's trigram -> bigram -> unigram: Q_k = (sum_k + M * Q_{k-1}) / (n_k + M).
  * With probability EPSILON the agent acts from ignorance (uniformly); otherwise it
    takes the action whose lived value is best.
  * Counts are updated after each episode (Monte Carlo). No weights, no loss, no
    optimiser; the same run is identical every time.

This is an adaptation of Netta's principles to acting, not Netta's code: Netta predicts
bytes, this agent chooses actions.

Usage:
    python experiments/tabula_rasa/netta_agent.py --poison --out runs/N_A_poison
    python experiments/tabula_rasa/netta_agent.py --poison --see-others --out runs/N_P2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabula_rasa as tr  # noqa: E402

EPSILON = 0.1  # Netta's share of honest ignorance
M = 4.0  # pseudo-count with which a coarser level speaks for an unlived action
GAMMA = 0.99


def levels() -> list[list[str]]:
    """Situation keys from fine to coarse. Language fields are left out (language off)."""
    senses = [k for k in ("pred_dir", "pred_dist", "mate_dir", "mate_dist", "mates_near")
              if k in tr.FIELDS]
    full = ["tile_res", "tile_mass", "adj_n", "adj_e", "adj_s", "adj_w", "energy", "held",
            "last_action", "last_ok", "energy_trend"] + senses
    coarse = ["tile_res", "held", "energy_band", "energy_trend"] + \
             [k for k in ("pred_dir", "mate_dir") if k in senses]
    return [full, coarse, ["tile_res", "held"]]


class Memory:
    def __init__(self, n_actions: int):
        self.levels = levels()
        self.n_actions = n_actions
        self.tables = [defaultdict(lambda: np.zeros((2, n_actions))) for _ in self.levels]
        self.root = np.zeros((2, n_actions))

    def keys(self, frame: np.ndarray) -> list[tuple]:
        vals = {k: int(frame[j]) - tr.OFFSETS[k] for j, k in enumerate(tr.FIELDS)}
        vals["energy_band"] = vals["energy"] // 3
        return [tuple(vals[k] for k in lvl) for lvl in self.levels]

    def value(self, keys: list[tuple]) -> np.ndarray:
        n, s = self.root
        q = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        for table, key in zip(reversed(self.tables), reversed(keys)):
            entry = table.get(key)
            if entry is not None:
                q = (entry[1] + M * q) / (entry[0] + M)
        return q

    def lived(self, keys: list[tuple], action: int, ret: float) -> None:
        self.root[0, action] += 1
        self.root[1, action] += ret
        for table, key in zip(self.tables, keys):
            entry = table[key]
            entry[0, action] += 1
            entry[1, action] += ret

    def size(self) -> dict:
        return {f"level_{i}": len(t) for i, t in enumerate(self.tables)}


def run_episode(memory: Memory, seed: int, args, epsilon: float, learn: bool):
    world = tr.World(seed, args.ticks, poison=args.poison, predators=args.predators)
    rng = np.random.default_rng(seed + 104729)
    trace = [[] for _ in range(world.n)]  # (keys, action, reward) per agent
    meals, choices, done = Counter(), Counter(), False
    while not done:
        alive = world.alive()
        acts = np.zeros(world.n, dtype=np.int64)
        keys = [None] * world.n
        for i in np.nonzero(alive)[0]:
            keys[i] = memory.keys(world.history[i, -1])
            if rng.random() < epsilon:
                acts[i] = rng.integers(tr.N_ACTIONS)
            else:
                q = memory.value(keys[i])
                best = np.flatnonzero(q >= q.max() - 1e-12)
                acts[i] = rng.choice(best)
            choices[tr.ACTION_NAMES[acts[i]]] += 1
        rew, done, eaten = world.step(acts, np.zeros(world.n, dtype=np.int64))
        meals.update(eaten)
        for i in np.nonzero(alive)[0]:
            trace[i].append((keys[i], int(acts[i]), float(rew[i])))
    if learn:
        for steps in trace:
            ret = 0.0
            for keys, action, r in reversed(steps):
                ret = r + GAMMA * ret
                memory.lived(keys, action, ret)
    pop = world.sim.population
    death = np.where(pop.death_tick >= 0, pop.death_tick, world.sim.tick)
    return {"mean_lifespan": float(death.mean()), "survivors": int(pop.active.sum()),
            "agents": world.n, "meals": dict(meals), "social": dict(world.social),
            "action_mix": {k: round(v / max(1, sum(choices.values())), 3)
                           for k, v in choices.most_common()}}


def evaluate(memory, seeds, args, epsilon):
    runs = [run_episode(memory, s, args, epsilon, learn=False) for s in seeds]
    meals, social = Counter(), Counter()
    for r in runs:
        meals.update(r["meals"]); social.update(r["social"])
    return {"mean_lifespan": round(float(np.mean([r["mean_lifespan"] for r in runs])), 1),
            "survival_rate": round(sum(r["survivors"] for r in runs) / sum(r["agents"] for r in runs), 3),
            "meals": dict(meals.most_common()), "social": dict(social),
            "action_mix": runs[0]["action_mix"]}


def food_probe(memory: Memory) -> dict:
    """Lived value of eating each held material (only that material in the inventory,
    mid energy), read from the coarsest situation level (tile + inventory, any tile)."""
    out = {}
    table = memory.tables[-1]
    for bit, r in enumerate(tr.RESOURCES):
        n = s = np.zeros(tr.N_ACTIONS)
        for (tile, held), entry in table.items():
            if held == 1 << bit:
                n, s = n + entry[0], s + entry[1]
        a = tr.EAT_OFFSET + bit
        wait = s[0] / n[0] if n[0] else float("nan")
        out[r.name] = {"lived_eats": int(n[a]),
                       "eat_minus_wait": round(float(s[a] / n[a] - wait), 3) if n[a] and n[0] else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--poison", action="store_true")
    ap.add_argument("--predators", action="store_true")
    ap.add_argument("--see-others", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="offset of the training world seeds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tr.configure(False, args.predators, args.see_others)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    memory = Memory(tr.N_ACTIONS)
    eval_seeds = list(range(1000, 1000 + args.eval_seeds))
    curve, t0 = [], time.time()
    for ep in range(args.episodes):
        stats = run_episode(memory, args.seed * 100000 + ep, args, EPSILON, learn=True)
        curve.append({"episode": ep, "mean_lifespan": stats["mean_lifespan"],
                      "survivors": stats["survivors"]})
        if ep % 10 == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  lifespan {stats['mean_lifespan']:6.1f}  survivors {stats['survivors']:2d}/12"
                  f"  meals {dict(Counter(stats['meals']).most_common(4))}  {memory.size()}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    report = {"config": vars(args), "epsilon": EPSILON, "backoff_pseudo_count": M,
              "memory": memory.size(),
              "after_training_eps_0.1": evaluate(memory, eval_seeds, args, EPSILON),
              "after_training_greedy": evaluate(memory, eval_seeds, args, 0.0),
              "food_probe": food_probe(memory), "curve": curve}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
