# Tabula-rasa agents

A ~127k-parameter nanoGPT-style transformer trained **from scratch** (no pretrained
weights, no text prompts) to survive in SwarmWorld's metabolic economy. The agent sees
the world only as discrete tokens (tile, neighbours, energy, held materials, last action
and whether it succeeded); the only reward is staying alive. Which materials are edible
is never told to the agent.

```bash
pip install torch
python experiments/tabula_rasa/tabula_rasa.py --episodes 300 --out runs/tabula_rasa
python experiments/tabula_rasa/tabula_rasa.py --episodes 300 --poison --language on --team-weight 0.5 --out runs/B_language
python experiments/tabula_rasa/group_analysis.py runs/A_poison runs/B_language --baselines
```

## Experiment 1: survival

The table below comes from an earlier version of the script (14 actions, 3-level energy
trend, no speech tokens), commit `279c4a9`.

First result (12 agents, 400 ticks, 5 held-out seeds, 300 training episodes, ~35 min CPU):

| policy                              | mean lifespan | survival rate |
|-------------------------------------|--------------:|--------------:|
| WAIT only                           | 249           | 0 %           |
| random                              | 254           | 35 %          |
| TinyGPT before training             | 212           | 17 %          |
| **TinyGPT after training**          | **343**       | **68 %**      |
| scripted, with built-in food knowledge | 355        | 85 %          |

Caveats: training is noisy (single run, per-episode survival oscillates between 4 and
12 of 12), and failed `METABOLIZE` attempts cost no energy, so the policy keeps
"trying" inedible materials instead of learning to avoid them.

## Experiment 2: poison, raw-symbol language, feeding

CHITIN (second most common material, previously food) now drains 0.20 energy per bite.
Agents may emit one of 36 meaningless symbols (A-Z, 0-9) or stay silent every tick and
hear the nearest speaker within 6 tiles (symbol, direction, distance). Agents cannot
see each other otherwise. One training run of 300 episodes per condition, evaluated on
5 held-out seeds (60 agents); all conditions have poison.

| run | language | reward | feeding | lifespan | survival (eval) | survival (last 50 training eps) | P(eat chitin) before -> after | grouping index |
|---|---|---|---|--:|--:|--:|---|--:|
| A | off | 50% team | - | 277 | 15 % | 26 % | 0.14 -> 0.01 | 1.06 |
| B | on | 50% team | - | 295 | 40 % | 37 % | 0.14 -> 0.06 | 1.25 |
| C | muted (nobody hears) | 50% team | - | 292 | 32 % | 42 % | 0.14 -> 0.01 | 1.14 |
| D | on | own only | - | 307 | 45 % | 41 % | 0.14 -> 0.01 | 1.03 |
| E | on | own only | yes, +10% bonus | 293 | 30 % | 33 % | 0.07 -> 0.00 | 1.14 |
| random | - | - | - | 207 | 23 % | - | - | 1.26 |
| scripted (knows food) | - | - | - | 295 | 65 % | - | - | 1.24 |

Speech: mutual information in bits (95th percentile of 20 symbol shuffles in brackets).

| run | symbol vs speaker's tile | symbol vs speaker's energy | heard symbol vs listener's action |
|---|--:|--:|--:|
| B | 0.114 (0.011) | 0.083 (0.014) | 0.052 (0.048) |
| C (muted) | 0.094 (0.012) | 0.162 (0.015) | - |
| D | 0.062 (0.012) | 0.112 (0.014) | 0.054 (0.050) |
| E | 0.053 (0.010) | 0.123 (0.014) | 0.103 (0.100) |

Findings:

- **Poison avoidance is learned in every condition**, from the energy drop alone:
  chitin eaten per 50 training episodes falls from 200-280 to roughly 15-100.
- **Symbols do depend on the speaker's situation, but the muted control shows the same
  pattern.** The speaker-side correlation is therefore a by-product of the shared network
  ("a growling stomach"), not communication. Listeners react only marginally: in B, D and
  E the effect is just above the shuffle threshold.
- **No measurable survival benefit from language** (B vs muted C) and **no herding**:
  trained agents are no closer together than random walkers or food-following scripted
  agents. With scarce food and no benefit to proximity, spreading out is the sensible
  strategy.
- **Feeding hardly happens** (10 successful feeds in E's evaluation) because agents are
  rarely adjacent; failed FEED attempts cost nothing, so they are spammed.

Caveats: one training run per condition, and evaluation survival disagrees with late
training survival by up to 11 points, so differences between A-E of this size are within
run-to-run noise.

## Experiment 3: predators

Three predators chase agents within 6 tiles; an attack on an adjacent agent succeeds
with probability 0.6 / (1 + companions within 2 tiles) and costs 0.6 energy. Agents
sense the nearest predator within 4 tiles. `--see-others` additionally lets agents sense
the nearest other agent within 4 tiles and the number within 2. All runs: poison on,
reward = own survival only, one training run of 300 episodes each.

| run | predators | sees others | language | feeding | survival (eval / last 50 training eps) | grouping index | in contact (<=1 tile) |
|---|:-:|:-:|:-:|:-:|--:|--:|--:|
| E (from exp. 2) | - | - | on | yes | 30 % / 33 % | 1.16 | 0.1 % |
| F | yes | - | on | yes | 23 % / 27 % | 1.08 | 1.5 % |
| G | yes | yes | on | yes | 25 % / 23 % | 1.33 | 2.8 % |
| P1 | yes | yes | off | - | 68 % / 60 % | 1.30 | 11.6 % |
| P2 | - | yes | off | - | 72 % / 79 % | 1.53 | 14.1 % |
| random (no predators) | - | - | - | - | 23 % | 1.26 | 1.4 % |
| scripted (no predators) | - | - | - | - | 65 % | 1.24 | 2.8 % |

Findings:

- **Seeing each other, not danger, produced grouping.** P2 (no predators) is the most
  clustered run so far (14 % of agent-ticks in direct contact, 10x random walkers) and
  survives best. Adding predators (P1) slightly *reduced* grouping. A plausible reading is
  local enhancement: other agents mark where food is.
- **Predators mostly hit loners**: of 75 attacks in P1's evaluation, 66 were on agents
  with no companion nearby.
- **Flight rather than herding**: when a predator is within 4 tiles, the share of MOVE
  actions rises in P1 (21 % -> 29 %) and F (6 % -> 22 %), not in G (5 % -> 3 %). P1 has a
  strong "run west" reflex (84-88 % W with a predator adjacent in a probe).
- **Alarm cues on the speaker side**: symbols carry information about a nearby predator
  (F 0.029 bits vs 0.002 shuffled p95; G 0.011 vs 0.002), with distinct symbols when a
  predator is near. There was no muted control for F/G, so this may be the same
  by-product seen for hunger. Listeners do not react measurably (heard symbol vs action
  below the shuffle p95 in both).
- **Feeding stays rare**: 0 successful feeds in F, 16 in G (E: 10).
- Poison is still avoided in eval (chitin is 0-3 % of meals), although chitin eaten
  during training rose again in P1/P2 as agents lived longer and ate more.
- The 22-action + speech runs (E, F, G) survive far worse than the 14-action runs
  without speech (P1, P2); action space size and the extra speech output are confounded
  with the treatments here.

`edibility_probe` ignored the predator/companion tokens before this commit, so the
"P(eat chitin)" probe values in the predator runs' `report.json` files are invalid.
