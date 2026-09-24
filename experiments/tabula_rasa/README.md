# Tabula-rasa agents

A ~127k-parameter nanoGPT-style transformer trained **from scratch** (no pretrained
weights, no text prompts) to survive in SwarmWorld's metabolic economy. The agent sees
the world only as discrete tokens (tile, neighbours, energy, held materials, last action
and whether it succeeded); the only reward is staying alive. Which materials are edible
is never told to the agent.

```bash
pip install torch
python experiments/tabula_rasa/tabula_rasa.py --episodes 300 --out runs/tabula_rasa
```

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
