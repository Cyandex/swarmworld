# Configuration profiles

Every YAML file is a complete run profile loaded through the same validated
`GameConfig`. CLI flags may override population size, seed, horizon, decision schedule,
and explicit budgets without editing the source file. A study manifest stores the
fully resolved configuration—not merely the YAML filename.

| Profile | Purpose | Agents | World | Model |
|---|---|---:|---|---|
| `demo.yaml` | deterministic smoke tests, oracle checks, and live renderer demo | 12 | fixed 48 × 36, central workspace | disabled |
| `scale.yaml` | scripted simulator benchmark | 1,024 default | 128 × 96 | disabled |
| `openai-gpt-5.6-luna.yaml` | 12-agent fixed-world hosted-model baseline | 12 | fixed 48 × 36, public infrastructure | `gpt-5.6-luna` |
| `openai-gpt-5.6-luna-technology-ecology-50.yaml` | live society and revision-9 flagship base | 50 | procedural 72 × 54, distributed hidden laboratories | `gpt-5.6-luna` |
| `openai-gpt-5.6-luna-procedural-metabolism.yaml` | separate survival, turnover, and inheritance treatment | 12 | procedural 48 × 36 | `gpt-5.6-luna` |
| `mistralrs-gemma-4-e4b-technology-ecology-50.yaml` | 50-agent local/remote open-weight study through mistral.rs on Metal, CUDA, or CPU | 50 | procedural 72 × 54, distributed hidden laboratories | `google/gemma-4-E4B-it` as `default` |
| `vllm-gemma-4-e4b-technology-ecology-50.yaml` | equivalent 50-agent open-weight study through vLLM | 50 | procedural 72 × 54, distributed hidden laboratories | `google/gemma-4-E4B-it` as `swarm-gemma-4-e4b` |
| `local-gemma-4-12b-mistralrs.yaml` | stronger local-model capacity pilot | 8 | fixed 48 × 36 | server alias `default` |
| `local-gemma-4-e4b-mistralrs.yaml` | local E4B capacity control | 12 | fixed 48 × 36 | server alias `default` |
| `local-gemma-4-mistralrs.yaml` | local E2B capacity control | 12 | fixed 48 × 36 | server alias `default` |

The technology-ecology profile leaves model-call and action-attempt budgets unset.
`research-study --decision-schedule fixed` overrides its live event-driven scheduler
for matched paper comparisons. `--world-scaling fixed` keeps its dimensions fixed while
retaining a new paired procedural layout for each seed.

The OpenAI profiles read `OPENAI_API_KEY`. Open-weight profiles target
`http://127.0.0.1:8000/v1` and use mistral.rs transport workarounds documented in
[STRUCTURED_OUTPUTS.md](../docs/STRUCTURED_OUTPUTS.md) only where that backend requires
them. The complete mistral.rs and vLLM setup is in
[LOCAL_MODEL_SERVING.md](../docs/LOCAL_MODEL_SERVING.md). Confirm server flags,
tokenizer files, and served model aliases against the installed serving release before
a long run.

Do not repurpose a completed confirmatory YAML in place. Copy it under a new experiment
name, change the seed block and output directory, and retain both resolved manifests.
Fixed/procedural generation, centralized/distributed workspaces, metabolism, and
decision scheduling alter the scientific treatment and must be reported separately.
