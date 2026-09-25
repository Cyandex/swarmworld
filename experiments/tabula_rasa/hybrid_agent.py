"""Hybrid agent: a Netta-style memory (hippocampus) + TinyGPT (cortex) as one unit.

  * Memory (netta_agent.Memory): counts every lived (situation, action) and the return
    that followed. Learns from one experience, forgets nothing, cannot generalise.
  * TinyGPT (tabula_rasa.TinyGPT): sees the last 6 ticks as tokens. Learns slowly, but
    generalises to situations it has never seen.
  * Decision, every tick and per agent: familiarity w = n / (n + K), where n is how often
    the memory has lived this exact situation. With probability w the memory decides
    (best lived action, backed off like Netta's n-grams); otherwise TinyGPT decides by
    sampling its policy. Known situation -> memory, unknown -> generalisation.
    The memory's ignorance share (epsilon 0.1) still applies when it decides.
  * Sleep, after every episode: the memory's knowledge is replayed into TinyGPT. For
    recently lived contexts the target is the memory's action preference,
    softmax(Q_memory / TAU), weighted by how well the memory knows the situation. TinyGPT
    never sees the reward directly; everything it knows comes through the memory.

The same trained unit is evaluated three ways: hybrid, memory alone, TinyGPT alone.

Usage:
    python experiments/tabula_rasa/hybrid_agent.py --poison --out runs/H_A
    python experiments/tabula_rasa/hybrid_agent.py --poison --predators --see-others \\
        --device cuda --out runs/H_P1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netta_agent as na  # noqa: E402
import tabula_rasa as tr  # noqa: E402

K = 10.0  # visits at which memory and TinyGPT get equal say (w = n / (n + K))
TAU = 0.05  # temperature of the memory's action preference taught to TinyGPT
REPLAY_EPISODES = 20  # sleep replays the contexts of the last N episodes
BATCH = 512


def familiarity(memory: na.Memory, keys: list[tuple]) -> float:
    entry = memory.tables[0].get(keys[0])
    n = float(entry[0].sum()) if entry is not None else 0.0
    return n / (n + K)


def memory_choice(memory, keys, rng, epsilon):
    if rng.random() < epsilon:
        return int(rng.integers(tr.N_ACTIONS))
    q = memory.value(keys)
    return int(rng.choice(np.flatnonzero(q >= q.max() - 1e-12)))


def run_episode(memory, model, seed, args, mode, learn, replay=None):
    """mode: 'hybrid', 'memory' or 'gpt'."""
    world = tr.World(seed, args.ticks, poison=args.poison, predators=args.predators)
    rng = np.random.default_rng(seed + 104729)
    trace = [[] for _ in range(world.n)]
    ctx_log, key_log = [], []
    meals, deciders, done = Counter(), Counter(), False
    while not done:
        alive = world.alive()
        live = np.nonzero(alive)[0]
        acts = np.zeros(world.n, dtype=np.int64)
        ctx = world.contexts()
        with torch.no_grad():
            logits = model(torch.from_numpy(ctx[live]).to(args.device))[0]
            gpt_acts = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        keys = {}
        for j, i in enumerate(live):
            keys[i] = memory.keys(world.history[i, -1])
            w = {"hybrid": familiarity(memory, keys[i]), "memory": 1.0, "gpt": 0.0}[mode]
            if rng.random() < w:
                acts[i] = memory_choice(memory, keys[i], rng, na.EPSILON)
                deciders["memory"] += 1
            else:
                acts[i] = gpt_acts[j]
                deciders["gpt"] += 1
            if learn:
                ctx_log.append(ctx[i]); key_log.append(keys[i])
        rew, done, eaten = world.step(acts, np.zeros(world.n, dtype=np.int64))
        meals.update(eaten)
        for i in live:
            trace[i].append((keys[i], int(acts[i]), float(rew[i])))
    if learn:
        for steps in trace:  # the memory learns what this day taught
            ret = 0.0
            for keys, action, r in reversed(steps):
                ret = r + na.GAMMA * ret
                memory.lived(keys, action, ret)
        replay.append((np.stack(ctx_log), key_log))
    pop = world.sim.population
    death = np.where(pop.death_tick >= 0, pop.death_tick, world.sim.tick)
    total = max(1, sum(deciders.values()))
    return {"mean_lifespan": float(death.mean()), "survivors": int(pop.active.sum()),
            "agents": world.n, "meals": dict(meals), "social": dict(world.social),
            "memory_decided": round(deciders["memory"] / total, 3)}


def sleep(memory, model, opt, replay, rng, device, samples) -> float:
    """Replay recent days: TinyGPT learns the memory's preferences, weighted by how
    familiar the memory is with each situation."""
    ctxs = np.concatenate([c for c, _ in replay])
    keys = [k for _, ks in replay for k in ks]
    pick = rng.choice(len(keys), min(samples, len(keys)), replace=False)
    q = np.stack([memory.value(keys[p]) for p in pick])
    target = torch.softmax(torch.from_numpy((q - q.max(1, keepdims=True)) / TAU), 1).float()
    weight = torch.tensor([familiarity(memory, keys[p]) for p in pick], dtype=torch.float32)
    x = torch.from_numpy(ctxs[pick])
    losses = []
    for b in range(0, len(pick), BATCH):
        logits = model(x[b:b + BATCH].to(device))[0]
        ce = -(target[b:b + BATCH].to(device) * F.log_softmax(logits, 1)).sum(1)
        w = weight[b:b + BATCH].to(device)
        loss = (w * ce).sum() / w.sum().clamp_min(1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(memory, model, seeds, args, mode):
    runs = [run_episode(memory, model, s, args, mode, learn=False) for s in seeds]
    meals, social = Counter(), Counter()
    for r in runs:
        meals.update(r["meals"]); social.update(r["social"])
    return {"mean_lifespan": round(float(np.mean([r["mean_lifespan"] for r in runs])), 1),
            "survival_rate": round(sum(r["survivors"] for r in runs) / sum(r["agents"] for r in runs), 3),
            "memory_decided": round(float(np.mean([r["memory_decided"] for r in runs])), 3),
            "meals": dict(meals.most_common()), "social": dict(social)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=400)
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--poison", action="store_true")
    ap.add_argument("--predators", action="store_true")
    ap.add_argument("--see-others", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--sleep-samples", type=int, default=4096, help="contexts replayed per sleep")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    tr.configure(False, args.predators, args.see_others)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    memory = na.Memory(tr.N_ACTIONS)
    model = tr.TinyGPT().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    replay, rng = deque(maxlen=REPLAY_EPISODES), np.random.default_rng(args.seed)
    eval_seeds = list(range(1000, 1000 + args.eval_seeds))
    curve, t0 = [], time.time()
    for ep in range(args.episodes):
        model.eval()
        stats = run_episode(memory, model, args.seed * 100000 + ep, args, "hybrid", True, replay)
        model.train()
        loss = sleep(memory, model, opt, replay, rng, args.device, args.sleep_samples)
        model.eval()
        curve.append({"episode": ep, "mean_lifespan": stats["mean_lifespan"],
                      "survivors": stats["survivors"], "memory_decided": stats["memory_decided"],
                      "sleep_loss": round(loss, 4)})
        if ep % 10 == 0 or ep == args.episodes - 1:
            print(f"ep {ep:4d}  lifespan {stats['mean_lifespan']:6.1f}  survivors {stats['survivors']:2d}/12"
                  f"  memory decided {stats['memory_decided']:.0%}  sleep loss {loss:.3f}"
                  f"  meals {dict(Counter(stats['meals']).most_common(3))}  [{time.time() - t0:.0f}s]",
                  flush=True)
    report = {"config": vars(args), "K": K, "tau": TAU, "memory": memory.size(),
              **{f"eval_{m}": evaluate(memory, model, eval_seeds, args, m)
                 for m in ("hybrid", "memory", "gpt")},
              "curve": curve}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    torch.save(model.state_dict(), out / "tinygpt.pt")
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
