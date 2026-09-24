"""Tabula-rasa agents: a tiny nanoGPT-style transformer trained from scratch in SwarmWorld.

No pretrained weights, no English prompts, no built-in knowledge. Each agent sees the
world only as a stream of discrete tokens (local tile, neighbours, energy, inventory,
its previous action and whether that action worked). The only learning signal is
staying alive under the metabolic economy. Everything the policy "knows" about the
world - e.g. which materials are edible - has to be discovered through interaction.

Optional treatments:
  --poison     CHITIN (common, looks like food) drains energy instead of giving it.
  --language   on:    every tick each agent may emit one raw symbol (A-Z, 0-9) or stay
                      silent; agents hear the nearest speaker within radius 6, plus its
                      direction and distance. Symbols carry no predefined meaning.
               muted: agents still "speak", but nobody hears anything (control).
  --team-weight  share of reward that comes from the group's survival. Without a shared
                 stake, a speaker gains nothing from informing others.

Usage:
    python experiments/tabula_rasa/tabula_rasa.py --episodes 300 --out runs/tabula_rasa
    python experiments/tabula_rasa/tabula_rasa.py --poison --language on --team-weight 0.5
"""

from __future__ import annotations

import argparse
import json
import string
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
EAT_OFFSET = 6

# --- poison ----------------------------------------------------------------------------
POISON = Resource.CHITIN
POISON_DAMAGE = 0.20  # energy lost per 0.25 mass eaten

# --- language: raw symbols without any built-in meaning --------------------------------
ALPHABET = ["_"] + list(string.ascii_uppercase) + list(string.digits)  # "_" = silence
N_SYMBOLS = len(ALPHABET)
HEAR_RADIUS = 6

# --- token vocabulary: each field gets its own id range (a tiny "world language") -----
FIELDS = {
    "tile_res": 9,
    "tile_mass": 4,
    "adj_n": 10, "adj_e": 10, "adj_s": 10, "adj_w": 10,  # 9 resources + blocked
    "energy": 10,
    "held": 256,  # bitmask of which of the 8 materials are in inventory
    "last_action": N_ACTIONS,
    "last_ok": 2,
    "energy_trend": 5,  # big drop, drop, flat, gain, big gain
    "said": N_SYMBOLS,  # own last symbol (lets multi-tick "words" form)
    "heard": N_SYMBOLS,  # nearest speaker's symbol this tick
    "heard_dir": 5,  # none, N, E, S, W
    "heard_dist": 4,
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


def trend_bin(delta: float) -> int:
    if delta < -0.05:
        return 0
    if delta < -1e-6:
        return 1
    if delta <= 1e-6:
        return 2
    return 3 if delta < 0.05 else 4


class World:
    """Wraps the authoritative simulation and turns it into per-agent token streams."""

    def __init__(self, seed: int, max_ticks: int, poison: bool = False, language: str = "off"):
        self.sim = BioFoundrySimulation(make_config(max_ticks))
        self.sim.reset(seed)
        self.poison, self.language = poison, language
        self.n = self.sim.population.size
        self.history = np.zeros((self.n, CONTEXT_TICKS, TOK_PER_TICK), dtype=np.int64)
        self.last_action = np.zeros(self.n, dtype=np.int64)
        self.last_ok = np.ones(self.n, dtype=np.int64)
        self.said = np.zeros(self.n, dtype=np.int64)
        self.heard = np.zeros((self.n, 3), dtype=np.int64)  # symbol, dir, dist
        self.prev_energy = self.sim.population.energy.copy()
        for i in range(self.n):
            self.history[i, :] = self._encode(i, trend=2)  # pad context with first frame

    def _res_at(self, x: int, y: int) -> int:
        w = self.sim.world
        if not (0 <= x < w.width and 0 <= y < w.height) or not w.walkable[y, x]:
            return 9
        return int(w.resource_kind[y, x]) if w.resource_mass[y, x] > 0.05 else 0

    def situation(self, i: int) -> dict:
        """Ground truth about an agent's situation, used only for analysing speech."""
        pop = self.sim.population
        x, y = int(pop.x[i]), int(pop.y[i])
        return {"tile": min(self._res_at(x, y), 8), "energy": min(int(pop.energy[i] * 10), 9)}

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
            "said": int(self.said[i]),
            "heard": int(self.heard[i, 0]),
            "heard_dir": int(self.heard[i, 1]),
            "heard_dist": int(self.heard[i, 2]),
        }
        return np.asarray([OFFSETS[k] + v for k, v in vals.items()], dtype=np.int64)

    def contexts(self) -> np.ndarray:
        return self.history.reshape(self.n, BLOCK).copy()

    def alive(self) -> np.ndarray:
        return self.sim.population.active.copy()

    def _deliver_speech(self, symbols: np.ndarray, alive: np.ndarray) -> None:
        self.heard[:] = 0
        if self.language != "on":
            return
        pop = self.sim.population
        speakers = [j for j in range(self.n) if alive[j] and symbols[j] != 0]
        for i in range(self.n):
            best = None
            for j in speakers:
                if j == i:
                    continue
                dx, dy = int(pop.x[j] - pop.x[i]), int(pop.y[j] - pop.y[i])
                dist = abs(dx) + abs(dy)
                if dist <= HEAR_RADIUS and (best is None or dist < best[0]):
                    best = (dist, j, dx, dy)
            if best is None:
                continue
            dist, j, dx, dy = best
            if dist == 0:
                direction = 0
            elif abs(dy) >= abs(dx):
                direction = 1 if dy < 0 else 3
            else:
                direction = 2 if dx > 0 else 4
            self.heard[i] = (symbols[j], direction, min(dist // 2, 3))

    def step(self, action_ids: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, bool, dict]:
        pop = self.sim.population
        alive_before = self.alive()
        ids = pop.agent_ids
        poisoned = []
        actions = {}
        for i, a in enumerate(action_ids):
            spec = dict(ACTIONS[a])
            if (self.poison and alive_before[i] and spec["verb"] == ActionType.METABOLIZE
                    and spec["resource"] == POISON and pop.inventory[i, int(POISON)] > 0.05):
                eaten = min(0.25, float(pop.inventory[i, int(POISON)]))
                pop.inventory[i, int(POISON)] -= np.float32(eaten)
                pop.energy[i] = np.float32(max(0.0, float(pop.energy[i]) - POISON_DAMAGE * eaten / 0.25))
                poisoned.append(i)
                spec = {"verb": ActionType.WAIT}
            actions[ids[i]] = AgentAction.from_value(spec)
        result = self.sim.step(actions)
        rejected = {e.payload.get("agent") for e in result.events if e.kind == "action_rejected"}
        meals = Counter(Resource(int(e.payload["resource"])).name
                        for e in result.events if e.kind == "resource_metabolized")
        if poisoned:
            meals[f"{POISON.name} (poison)"] += len(poisoned)
        alive_after = self.alive()
        self._deliver_speech(np.where(alive_before, symbols, 0), alive_before)
        energy = pop.energy
        # survival is the only reward: +0.01 per tick alive, -1 on death
        reward = np.where(alive_after, 0.01, 0.0) - np.where(alive_before & ~alive_after, 1.0, 0.0)
        for i in range(self.n):
            if not alive_after[i]:
                continue
            self.last_action[i] = action_ids[i]
            self.last_ok[i] = 0 if ids[i] in rejected else 1
            self.said[i] = symbols[i]
            trend = trend_bin(float(energy[i] - self.prev_energy[i]))
            self.history[i, :-1] = self.history[i, 1:]
            self.history[i, -1] = self._encode(i, trend)
        self.prev_energy = energy.copy()
        done = self.sim.tick >= self.sim.config.simulation.max_ticks or not alive_after.any()
        return reward.astype(np.float32), done, meals


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
        self.speech = nn.Linear(d, N_SYMBOLS)
        self.value = nn.Linear(d, 1)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        x = self.ln(self.blocks(x))[:, -1]
        return self.policy(x), self.speech(x), self.value(x).squeeze(-1)


def distributions(model, ctx, speak: bool):
    logits, speech, value = model(ctx)
    act = torch.distributions.Categorical(logits=logits)
    sym = torch.distributions.Categorical(logits=speech) if speak else None
    return act, sym, value


# --- policies --------------------------------------------------------------------------
def edible(poison: bool) -> set:
    good = {r for r in RESOURCES if float(FEEDSTOCK_COMPOSITION[r][[0, 1, 2, 4]].sum()) >= 0.20}
    return good - {POISON} if poison else good


def scripted_actions(world: World, rng) -> np.ndarray:
    """Hand-written baseline WITH innate knowledge of which materials are edible."""
    food = edible(world.poison)
    food_ids = {int(r) for r in food}
    pop, out = world.sim.population, np.zeros(world.n, dtype=np.int64)
    best = sorted(food, key=lambda r: -float(FEEDSTOCK_COMPOSITION[r][[0, 1, 2, 4]].sum()))
    for i in range(world.n):
        x, y = int(pop.x[i]), int(pop.y[i])
        held = [r for r in best if pop.inventory[i, int(r)] > 0.05]
        if pop.energy[i] < 0.7 and held:
            out[i] = EAT_OFFSET + RESOURCES.index(held[0])
        elif world._res_at(x, y) in food_ids and pop.inventory[i].sum() < 3:
            out[i] = 5
        else:
            adj = [world._res_at(x, y - 1), world._res_at(x + 1, y),
                   world._res_at(x, y + 1), world._res_at(x - 1, y)]
            good = [k for k, r in enumerate(adj) if r in food_ids]
            ok = [k for k, r in enumerate(adj) if r != 9]
            out[i] = 1 + (rng.choice(good) if good else rng.choice(ok))
    return out


def run_episode(policy, seed, args, model=None, rng=None, record=False, log_speech=False):
    world = World(seed, args.ticks, poison=args.poison, language=args.language)
    speak = args.language != "off"
    buf = {"ctx": [], "act": [], "sym": [], "logp": [], "val": [], "rew": [], "mask": []}
    meals, choices, done = Counter(), Counter(), False
    speech_log = []  # (speaker situation, symbol) and (heard symbol, next action)
    while not done:
        alive = world.alive()
        symbols = np.zeros(world.n, dtype=np.int64)
        if policy == "gpt":
            ctx = torch.from_numpy(world.contexts())
            with torch.no_grad():
                act_d, sym_d, value = distributions(model, ctx, speak)
            act = act_d.sample()
            logp = act_d.log_prob(act)
            if speak:
                sym = sym_d.sample()
                logp = logp + sym_d.log_prob(sym)
                symbols = sym.numpy()
            acts = act.numpy()
            if record:
                buf["ctx"].append(ctx); buf["act"].append(act)
                buf["sym"].append(torch.from_numpy(symbols))
                buf["logp"].append(logp); buf["val"].append(value)
        elif policy == "random":
            acts = rng.integers(0, N_ACTIONS, world.n)
        elif policy == "wait":
            acts = np.zeros(world.n, dtype=np.int64)
        else:
            acts = scripted_actions(world, rng)
        acts = np.where(alive, acts, 0)
        if log_speech:
            for i in np.nonzero(alive)[0]:
                speech_log.append({**world.situation(i), "symbol": int(symbols[i]),
                                   "heard": int(world.heard[i, 0]), "action": int(acts[i]),
                                   "agent": int(i), "tick": world.sim.tick})
        for a in acts[alive]:
            choices[ACTION_NAMES[a]] += 1
        rew, done, eaten = world.step(acts, symbols)
        meals.update(eaten)
        if record:
            buf["rew"].append(torch.from_numpy(rew)); buf["mask"].append(torch.from_numpy(alive))
    pop = world.sim.population
    death = np.where(pop.death_tick >= 0, pop.death_tick, world.sim.tick)
    stats = {
        "mean_lifespan": float(death.mean()),
        "survivors": int(pop.active.sum()),
        "agents": world.n,
        "meals": dict(meals),
        "action_mix": {k: round(v / max(1, sum(choices.values())), 3) for k, v in choices.most_common()},
    }
    return stats, buf, speech_log


def ppo_update(model, opt, buf, speak, team_weight, gamma=0.99, lam=0.95, epochs=4, clip=0.2):
    T = len(buf["rew"])
    rew, mask = torch.stack(buf["rew"]), torch.stack(buf["mask"]).float()
    if team_weight > 0:  # blend own reward with the group's average reward
        rew = (1 - team_weight) * rew + team_weight * rew.mean(dim=1, keepdim=True)
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
    sym = torch.stack(buf["sym"]).flatten()[keep]
    old = torch.stack(buf["logp"]).flatten()[keep]
    adv, ret = adv.flatten()[keep], ret.flatten()[keep]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = ctx.shape[0]
    for _ in range(epochs):
        for idx in torch.randperm(n).split(512):
            act_d, sym_d, v = distributions(model, ctx[idx], speak)
            logp, ent = act_d.log_prob(act[idx]), act_d.entropy()
            if speak:
                logp, ent = logp + sym_d.log_prob(sym[idx]), ent + sym_d.entropy()
            ratio = (logp - old[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
            loss = pg + 0.5 * F.mse_loss(v, ret[idx]) - 0.01 * ent.mean()
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
            "energy": 3, "held": 1 << bit, "last_action": 5, "last_ok": 1, "energy_trend": 1,
            "said": 0, "heard": 0, "heard_dir": 0, "heard_dist": 0,
        }.items()])
        ctx = torch.from_numpy(np.tile(frame, CONTEXT_TICKS))[None]
        with torch.no_grad():
            p = torch.softmax(model(ctx)[0], -1)[0]
        out[r.name] = {"p_eat_it": round(float(p[EAT_OFFSET + bit]), 3)}
    return out


def mutual_information(pairs: list[tuple[int, int]]) -> float:
    """I(X;Y) in bits from observed (x, y) pairs."""
    if not pairs:
        return 0.0
    n = len(pairs)
    joint = Counter(pairs)
    px, py = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    return sum(c / n * np.log2(c * n / (px[x] * py[y])) for (x, y), c in joint.items())


def speech_analysis(log: list[dict], rng) -> dict:
    """Is there a pattern? Compare real MI against the same data with symbols shuffled."""
    spoken = [e for e in log if e["symbol"] != 0]
    usage = Counter(ALPHABET[e["symbol"]] for e in log)
    total = sum(usage.values())

    def mi_vs_shuffle(xs, ys):
        real = mutual_information(list(zip(xs, ys)))
        ys = list(ys)
        shuffled = []
        for _ in range(20):
            rng.shuffle(ys)
            shuffled.append(mutual_information(list(zip(xs, ys))))
        return {"bits": round(real, 4), "shuffled_bits": round(float(np.mean(shuffled)), 4),
                "shuffled_p95": round(float(np.percentile(shuffled, 95)), 4)}

    # speaker side: does the symbol depend on what the speaker is standing on / how hungry?
    sym = [e["symbol"] for e in log]
    # listener side: does hearing a symbol change what the listener does?
    heard = [e for e in log if e["heard"] != 0]
    # per-tile favourite symbol ("vocabulary")
    vocab = {}
    for tile in range(9):
        c = Counter(ALPHABET[e["symbol"]] for e in log if e["tile"] == tile)
        if sum(c.values()) >= 50:
            name = Resource(tile).name if tile else "EMPTY"
            top = c.most_common(3)
            vocab[name] = [(s, round(k / sum(c.values()), 3)) for s, k in top]
    # bigrams: consecutive symbols by the same agent ("words")
    by_agent = {}
    for e in sorted(log, key=lambda e: (e["agent"], e["tick"])):
        by_agent.setdefault(e["agent"], []).append(e["symbol"])
    bigrams = Counter()
    for seq in by_agent.values():
        for a, b in zip(seq, seq[1:]):
            if a and b:
                bigrams[ALPHABET[a] + ALPHABET[b]] += 1
    probs = np.asarray([c / total for c in usage.values()])
    return {
        "silence_rate": round(usage["_"] / total, 3),
        "symbols_used_over_1pct": sum(1 for s, c in usage.items() if s != "_" and c / total > 0.01),
        "entropy_bits": round(float(-(probs * np.log2(probs)).sum()), 3),
        "top_symbols": [(s, round(c / total, 3)) for s, c in usage.most_common(8)],
        "mi_symbol_vs_tile": mi_vs_shuffle([e["tile"] for e in log], sym),
        "mi_symbol_vs_energy": mi_vs_shuffle([e["energy"] for e in log], sym),
        "mi_heard_vs_action": mi_vs_shuffle([e["heard"] for e in heard], [e["action"] for e in heard]),
        "favourite_symbols_by_tile": vocab,
        "top_bigrams": bigrams.most_common(8),
        "utterances": len(spoken),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--poison", action="store_true")
    ap.add_argument("--language", choices=["off", "on", "muted"], default="off")
    ap.add_argument("--team-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs" / "tabula_rasa"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)

    model = TinyGPT()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT: {n_params:,} parameters, vocab {VOCAB}, context {BLOCK} tokens, "
          f"poison={args.poison} language={args.language} team_weight={args.team_weight}")

    eval_seeds = list(range(1000, 1000 + args.eval_seeds))

    def evaluate(policy, log_speech=False):
        runs, logs = [], []
        for s in eval_seeds:
            stats, _, log = run_episode(policy, s, args, model=model, rng=np.random.default_rng(s),
                                        log_speech=log_speech)
            runs.append(stats); logs.extend(log)
        meals = Counter()
        for r in runs:
            meals.update(r["meals"])
        result = {"mean_lifespan": round(float(np.mean([r["mean_lifespan"] for r in runs])), 1),
                  "survival_rate": round(sum(r["survivors"] for r in runs) / sum(r["agents"] for r in runs), 3),
                  "meals": dict(meals.most_common()), "action_mix": runs[0]["action_mix"]}
        return result, logs

    report = {"config": vars(args), "params": n_params,
              "baselines": {p: evaluate(p)[0] for p in ("wait", "random", "scripted")}}
    report["gpt_before_training"] = evaluate("gpt")[0]
    report["probe_before"] = edibility_probe(model)
    print(json.dumps({k: (v["mean_lifespan"], v["survival_rate"]) for k, v in report["baselines"].items()}))

    curve, t0 = [], time.time()
    for ep in range(args.episodes):
        stats, buf, _ = run_episode("gpt", seed=ep, args=args, model=model, record=True)
        ppo_update(model, opt, buf, args.language != "off", args.team_weight)
        curve.append({"episode": ep, "mean_lifespan": stats["mean_lifespan"],
                      "survivors": stats["survivors"], "meals": stats["meals"]})
        if ep % 10 == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  lifespan {stats['mean_lifespan']:6.1f}  survivors {stats['survivors']:2d}/12"
                  f"  meals {dict(Counter(stats['meals']).most_common(4))}  [{time.time() - t0:.0f}s]", flush=True)

    report["gpt_after_training"], log = evaluate("gpt", log_speech=True)
    report["probe_after"] = edibility_probe(model)
    if args.language != "off":
        report["speech"] = speech_analysis(log, np.random.default_rng(0))
    report["curve"] = curve
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "speech_log.json").write_text(json.dumps(log))
    torch.save(model.state_dict(), out / "tinygpt.pt")
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
