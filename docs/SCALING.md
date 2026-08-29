# Scaling

## What can scale

The dense world mechanics can scale to thousands of agents on a laptop because
positions, energy, inventories, and fields are numeric arrays. An LLM population
has a different cost model: model requests, context length, and memory retrieval
will dominate long before movement and diffusion do.

Therefore “10,000 agents” means one of several regimes:

1. 10,000 scripted or learned motor policies;
2. 10,000 agents with sparse, staggered LLM macroturns;
3. a smaller set of active LLM reasoners plus many cached-plan actors;
4. cohort or representative cognition with individual physical state;
5. an aggregate population model for the largest runs.

Results must label the regime and never equate them silently.

## Implemented techniques

- NumPy structure-of-arrays population state.
- Vectorized movement and field diffusion.
- Sparse loops only over submitted semantic actions and artifacts.
- Fixed-capacity artifact arrays.
- Bounded memory, plan queues, program registers, and instructions.
- Delta events with configurable snapshot interval.
- A separate renderer cadence and a display-agent limit.

## Initial local benchmark

Development machine: Apple Silicon macOS, Python 3.12, PyTorch conda environment,
CPU simulator, 100 microticks, science instrumentation enabled, no LLM and no renderer.
This is an engineering smoke
benchmark, not a publication result.

| Agents | Ticks/s | Agent-steps/s |
|---:|---:|---:|
| 16 | 2,990 | 47,845 |
| 64 | 970 | 62,100 |
| 256 | 261 | 66,886 |
| 1,024 | 64 | 65,904 |
| 4,096 | 14 | 58,706 |

The exact command was:

```bash
biofoundry benchmark --config configs/demo.yaml \
  --agents 16 64 256 1024 4096 --ticks 100
```

The plateau near 60,000 instrumented agent-steps/s reflects amortized field-update
cost followed by per-agent observation and event overhead. Longer
benchmarks, multiple repetitions, memory measurements, and confidence intervals
are required before making performance claims.

## Next bottlenecks

1. Constructing a separate PettingZoo observation dictionary for every agent.
2. Large-volume text events and memories.
3. Spatial recipient queries when most agents communicate simultaneously.
4. Per-artifact DSL execution after artifact counts become very large.
5. Full protocol snapshots and renderer load above a few thousand visible entities.

Planned mitigations include batched observation tensors, compressed semantic IDs,
sorted-cell spatial indices, grouped program execution, snapshot compression, and
renderer level of detail.

## LLM serving

For a high-throughput supported accelerator deployment, use a batching server such as
vLLM. mistral.rs is the cross-platform path: its distributed binaries support Metal on
Apple Silicon, CUDA on supported NVIDIA Linux systems, and CPU fallback. Both expose
the Responses API required by SwarmWorld. The complete setup is in
[LOCAL_MODEL_SERVING.md](LOCAL_MODEL_SERVING.md). Model-service throughput must be
benchmarked separately from simulator throughput.

The macroturn interval is a scientific control as well as a performance parameter.
Scaling comparisons should hold total tokens, model requests, or wall-clock budget
constant according to the hypothesis being tested.
