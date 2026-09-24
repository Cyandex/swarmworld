"""Tabula-rasa agents: a tiny nanoGPT-style transformer trained from scratch in SwarmWorld.

No pretrained weights, no English prompts, no built-in knowledge. Each agent sees the
world only as a stream of discrete tokens (local tile, neighbours, energy, inventory,
its previous action and whether that action worked). The only learning signal is
staying alive under the metabolic economy. Everything the policy "knows" about the
world - e.g. which materials are edible - has to be discovered through interaction.

Usage:
    python experiments/tabula_rasa/tabula_rasa.py --episodes 120 --out runs/tabula_rasa
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from biofoundry.config import load_config
from biofoundry.simulation import FEEDSTOCK_COMPOSITION, BioFoundrySimulation
from biofoundry.types import ActionType, AgentAction, Direction, Resource

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = [r for r in Resource if r != Resource.NONE]  # 8 materials

# --- action space: 14 discrete choices ------------------------------------------------
ACTIONS: list[dict] = [{"verb": ActionType.WAIT}]
ACTIONS += [{"verb": ActionType.MOVE, "direction": d} for d in
            (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)]
ACTIONS += [{"verb": ActionType.HARVEST}]
ACTIONS += [{"verb": ActionType.METABOLIZE, "resource": r, "amount": 0.25} for r in RESOURCES]
N_ACTIONS = len(ACTIONS)
ACTION_NAMES = ["WAIT", "N", "E", "S", "W", "HARVEST"] + [f"EAT_{r.name}" for r in RESOURCES]

# --- token vocabulary: each field gets its own id range (a tiny "world language") -----
FIELDS = {
    "tile_res": 9,
    "tile_mass": 4,
    "adj_n": 10, "adj_e": 10, "adj_s": 10, "adj_w": 10,  # 9 resources + blocked
    "energy": 10,
    "held": 256,  # bitmask of which of the 8 materials are in inventory
    "last_action": N_ACTIONS,
    "last_ok": 2,
    "energy_trend": 3,
}
OFFSETS, _o = {}, 0
for _name, _size in FIELDS.items():
    OFFSETS[_name] = _o
    _o += _size
VOCAB = _o
TOK_PER_TICK = len(FIELDS)
CONTEXT_TICKS = 6
BLOCK = TOK_PER_TICK * CONTEXT_TICKS


def make_config(max_ticks: int):
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg.simulation.max_ticks = max_ticks
    cfg.science.enabled = False  # pure survival world, no research layer
    eco = cfg.economy
    eco.enabled = True
    eco.mortality_enabled = True
    eco.respawn_enabled = False
    eco.passive_cost = 0.004  # doing nothing kills you in ~250 ticks
    eco.move_cost = 0.01
    eco.default_action_cost = 0.004
    return cfg


class World:
    """Wraps the authoritative simulation and turns it into per-agent token streams."""

    def __init__(self, seed: int, max_ticks: int):
        self.sim = BioFoundrySimulation(make_config(max_ticks))
        self.sim.reset(seed)
        self.n = self.sim.population.size
        self.history = np.zeros((self.n, CONTEXT_TICKS, TOK_PER_TICK), dtype=np.int64)
        self.last_action = np.zeros(self.n, dtype=np.int64)
        self.last_ok = np.ones(self.n, dtype=np.int64)
        self.prev_energy = self.sim.population.energy.copy()
        for i in range(self.n):
            tick_tokens = self._encode(i, trend=1)
            self.history[i, :] = tick_tokens  # pad context with first frame

    def _res_at(self, x: int, y: int) -> int:
        w = self.sim.world
        if not (0 <= x < w.width and 0 <= y < w.height) or not w.walkable[y, x]:
            return 9
        return int(w.resource_kind[y, x]) if w.resource_mass[y, x] > 0.05 else 0

    def _encode(self, i: int, trend: int) -> np.ndarray:
        pop, w = self.sim.population, self.sim.world
        x, y = int(pop.x[i]), int(pop.y[i])
        cap = w.resource_capacity[y, x]
        frac = w.resource_mass[y, x] / cap if cap > 0 else 0.0
        held = 0
        for bit, r in enumerate(RESOURCES):
            if pop.inventory[i, int(r)] > 0.05:
                held |= 1 << bit
        vals = {
            "tile_res": min(self._res_at(x, y), 8),
            "tile_mass": min(int(frac * 4), 3),
            "adj_n": self._res_at(x, y - 1), "adj_e": self._res_at(x + 1, y),
            "adj_s": self._res_at(x, y + 1), "adj_w": self._res_at(x - 1, y),
            "energy": min(int(pop.energy[i] * 10), 9),
            "held": held,
            "last_action": int(self.last_action[i]),
            "last_ok": int(self.last_ok[i]),
            "energy_trend": trend,
        }
        return np.asarray([OFFSETS[k] + v for k, v in vals.items()], dtype=np.int64)

    def contexts(self) -> np.ndarray:
        return self.history.reshape(self.n, BLOCK).copy()

    def alive(self) -> np.ndarray:
        return self.sim.population.active.copy()

    def step(self, action_ids: np.ndarray) -> tuple[np.ndarray, bool, list]:
        alive_before = self.alive()
        ids = self.sim.population.agent_ids
        actions = {ids[i]: AgentAction.from_value(dict(ACTIONS[a])) for i, a in enumerate(action_ids)}
        result = self.sim.step(actions)
        rejected = {e.payload.get("agent") for e in result.events if e.kind == "action_rejected"}
        eaten = [e.payload for e in result.events if e.kind == "resource_metabolized"]
        alive_after = self.alive()
        energy = self.sim.population.energy
        # survival is the only reward: +0.01 per tick alive, -1 on death
        reward = np.where(alive_after, 0.01, 0.0) - np.where(alive_before & ~alive_after, 1.0, 0.0)
        for i in range(self.n):
            if not alive_after[i]:
                continue
            self.last_action[i] = action_ids[i]
            self.last_ok[i] = 0 if self.sim.population.agent_ids[i] in rejected else 1
            delta = float(energy[i] - self.prev_energy[i])
            trend = 0 if delta < -1e-6 else (2 if delta > 1e-6 else 1)
            self.history[i, :-1] = self.history[i, 1:]
            self.history[i, -1] = self._encode(i, trend)
        self.prev_energy = energy.copy()
        done = self.sim.tick >= self.sim.config.simulation.max_ticks or not alive_after.any()
        return reward.astype(np.float32), done, eaten


# --- nanoGPT-style model ---------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        mask = torch.triu(torch.ones(BLOCK, BLOCK, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        h = self.ln1(x)
        t = x.shape[1]
        x = x + self.attn(h, h, h, attn_mask=self.mask[:t, :t], need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(BLOCK, d)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.policy = nn.Linear(d, N_ACTIONS)
        self.value = nn.Linear(d, 1)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        x = self.ln(self.blocks(x))[:, -1]
        return self.policy(x), self.value(x).squeeze(-1)


# --- policies --------------------------------------------------------------------------
EDIBLE = {r for r in RESOURCES if float(FEEDSTOCK_COMPOSITION[r][[0, 1, 2, 4]].sum()) >= 0.20}


def scripted_actions(world: World, rng) -> np.ndarray:
    """Hand-written baseline WITH innate knowledge of which materials are edible."""
    pop, out = world.sim.population, np.zeros(world.n, dtype=np.int64)
    best = sorted(EDIBLE, key=lambda r: -float(FEEDSTOCK_COMPOSITION[r][[0, 1, 2, 4]].sum()))
    for i in range(world.n):
        x, y = int(pop.x[i]), int(pop.y[i])
        held = [r for r in best if pop.inventory[i, int(r)] > 0.05]
        if pop.energy[i] < 0.7 and held:
            out[i] = 6 + RESOURCES.index(held[0])
        elif world._res_at(x, y) in {int(r) for r in EDIBLE} and pop.inventory[i].sum() < 3:
            out[i] = 5
        else:
            adj = [world._res_at(x, y - 1), world._res_at(x + 1, y),
                   world._res_at(x, y + 1), world._res_at(x - 1, y)]
            good = [k for k, r in enumerate(adj) if r in {int(e) for e in EDIBLE}]
            ok = [k for k, r in enumerate(adj) if r != 9]
            out[i] = 1 + (rng.choice(good) if good else rng.choice(ok))
    return out


def run_episode(policy, seed: int, max_ticks: int, model=None, rng=None, record=False):
    world = World(seed, max_ticks)
    buf = {"ctx": [], "act": [], "logp": [], "val": [], "rew": [], "mask": []}
    eaten, choices, done = Counter(), Counter(), False
    while not done:
        alive = world.alive()
        if policy == "gpt":
            ctx = torch.from_numpy(world.contexts())
            with torch.no_grad():
                logits, value = model(ctx)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            acts = act.numpy()
            if record:
                buf["ctx"].append(ctx); buf["act"].append(act)
                buf["logp"].append(dist.log_prob(act)); buf["val"].append(value)
        elif policy == "random":
            acts = rng.integers(0, N_ACTIONS, world.n)
        elif policy == "wait":
            acts = np.zeros(world.n, dtype=np.int64)
        else:
            acts = scripted_actions(world, rng)
        acts = np.where(alive, acts, 0)
        for a in acts[alive]:
            choices[ACTION_NAMES[a]] += 1
        rew, done, eats = world.step(acts)
        for e in eats:
            eaten[Resource(int(e["resource"])).name] += 1
        if record:
            buf["rew"].append(torch.from_numpy(rew)); buf["mask"].append(torch.from_numpy(alive))
    pop = world.sim.population
    death = np.where(pop.death_tick >= 0, pop.death_tick, world.sim.tick)
    stats = {
        "mean_lifespan": float(death.mean()),
        "survivors": int(pop.active.sum()),
        "agents": world.n,
        "meals": dict(eaten),
        "action_mix": {k: round(v / max(1, sum(choices.values())), 3) for k, v in choices.most_common()},
    }
    return stats, buf


def ppo_update(model, opt, buf, gamma=0.99, lam=0.95, epochs=4, clip=0.2):
    T = len(buf["rew"])
    rew, mask = torch.stack(buf["rew"]), torch.stack(buf["mask"]).float()
    val = torch.stack(buf["val"])
    adv = torch.zeros_like(rew)
    last = torch.zeros(rew.shape[1])
    for t in reversed(range(T)):
        nxt = val[t + 1] * mask[t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = rew[t] + gamma * nxt - val[t]
        last = delta + gamma * lam * (mask[t + 1] if t + 1 < T else 0) * last
        adv[t] = last
    ret = adv + val
    keep = mask.bool().flatten()
    ctx = torch.stack(buf["ctx"]).flatten(0, 1)[keep]
    act = torch.stack(buf["act"]).flatten()[keep]
    old = torch.stack(buf["logp"]).flatten()[keep]
    adv, ret = adv.flatten()[keep], ret.flatten()[keep]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = ctx.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).split(512):
            logits, v = model(ctx[idx])
            dist = torch.distributions.Categorical(logits=logits)
            ratio = (dist.log_prob(act[idx]) - old[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            loss = pg + 0.5 * F.mse_loss(v, ret[idx]) - 0.01 * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()


def edibility_probe(model) -> dict:
    """What does the trained model 'believe'? Hungry agent holding exactly one material:
    probability that it chooses to eat it."""
    out = {}
    for bit, r in enumerate(RESOURCES):
        frame = np.asarray([OFFSETS[k] + v for k, v in {
            "tile_res": 0, "tile_mass": 0, "adj_n": 0, "adj_e": 0, "adj_s": 0, "adj_w": 0,
            "energy": 3, "held": 1 << bit, "last_action": 5, "last_ok": 1, "energy_trend": 0,
        }.items()])
        ctx = torch.from_numpy(np.tile(frame, CONTEXT_TICKS))[None]
        with torch.no_grad():
            p = torch.softmax(model(ctx)[0], -1)[0]
        out[r.name] = {"edible": r in EDIBLE, "p_eat_it": round(float(p[6 + bit]), 3),
                       "p_eat_anything": round(float(p[6:].sum()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "runs" / "tabula_rasa"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    torch.set_num_threads(4)

    model = TinyGPT()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT: {n_params:,} parameters, vocab {VOCAB}, context {BLOCK} tokens")

    eval_seeds = list(range(1000, 1000 + args.eval_seeds))

    def evaluate(policy):
        runs = [run_episode(policy, s, args.ticks, model=model, rng=np.random.default_rng(s))[0]
                for s in eval_seeds]
        meals = Counter()
        for r in runs:
            meals.update(r["meals"])
        return {"mean_lifespan": round(float(np.mean([r["mean_lifespan"] for r in runs])), 1),
                "survival_rate": round(sum(r["survivors"] for r in runs) / sum(r["agents"] for r in runs), 3),
                "meals": dict(meals.most_common()), "action_mix": runs[0]["action_mix"]}

    report = {"baselines": {p: evaluate(p) for p in ("wait", "random", "scripted")}}
    report["gpt_before_training"] = evaluate("gpt")
    report["probe_before"] = edibility_probe(model)
    print(json.dumps(report["baselines"], indent=1))

    curve, t0 = [], time.time()
    for ep in range(args.episodes):
        stats, buf = run_episode("gpt", seed=ep, max_ticks=args.ticks, model=model, record=True)
        ppo_update(model, opt, buf)
        curve.append({"episode": ep, "mean_lifespan": stats["mean_lifespan"], "survivors": stats["survivors"]})
        if ep % 10 == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  lifespan {stats['mean_lifespan']:6.1f}  survivors {stats['survivors']:2d}/12"
                  f"  meals {dict(Counter(stats['meals']).most_common(3))}  [{time.time() - t0:.0f}s]", flush=True)

    report["gpt_after_training"] = evaluate("gpt")
    report["probe_after"] = edibility_probe(model)
    report["curve"] = curve
    report["params"] = n_params
    (out / "report.json").write_text(json.dumps(report, indent=1))
    torch.save(model.state_dict(), out / "tinygpt.pt")
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
