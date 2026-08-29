# Open-weight model serving

SwarmWorld does not load model weights itself. It sends non-streaming OpenAI Responses
API requests to the `llm.base_url` in a run profile. Local inference is therefore three
independent processes:

```text
open-weight model server (:8000/v1/responses)
                 |
                 v
BioFoundry simulation or live server
                 |
                 v
optional Three.js or Godot renderer
```

The checked-in Gemma 4 E4B profiles are:

- `configs/mistralrs-gemma-4-e4b-technology-ecology-50.yaml`: mistral.rs on Apple
  Silicon/Metal, NVIDIA CUDA, or CPU;
- `configs/vllm-gemma-4-e4b-technology-ecology-50.yaml`: vLLM on a supported accelerator.

Both reproduce the same 50-agent world, science settings, and 12-action macro-plan
capacity as the hosted-model technology-ecology profile. This keeps action opportunity
fixed when the model and serving stack change.

## 1. Prepare SwarmWorld

The `local` extra installs Transformers, which the tested mistral.rs profile uses to
identify whitespace-only tokenizer IDs for strict JSON generation:

```bash
conda activate PyTorch
cd SwarmWorld
python -m pip install -e ".[dev,analysis,local]"
```

Accept the `google/gemma-4-E4B-it` model terms on Hugging Face, then make a read token
available to the model server and, on first use, to the tokenizer loader:

```bash
export HF_TOKEN="hf_..."
```

Neither the token nor model weights are stored in a SwarmWorld trace.

## 2. Option A: mistral.rs on Metal or CUDA

mistral.rs distributes hardware-specific binaries. The same installation command
selects Metal on Apple Silicon, CUDA on supported NVIDIA Linux systems, and CPU when no
accelerator build applies:

```bash
curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh

mistralrs --version
mistralrs doctor
```

The installed binary can inspect the model and recommend quantization/device mapping:

```bash
mistralrs tune \
  --model-id google/gemma-4-E4B-it \
  --quant 4 \
  --profile balanced
```

Start a loopback-only Responses server on port 8000:

```bash
mistralrs serve \
  --model-id google/gemma-4-E4B-it \
  --token-source env:HF_TOKEN \
  --quant 4 \
  --host 127.0.0.1 \
  --port 8000 \
  --no-ui \
  --max-seq-len 32768 \
  --max-seqs 2
```

No Metal- or CUDA-specific flag is needed. mistral.rs selects the accelerator compiled
into its platform binary and automatically maps model layers. `--quant 4` prefers a
published four-bit UQFF and otherwise performs in-situ quantization. Omit it when native
precision fits and is part of the frozen experiment. On a larger CUDA system, increase
both server `--max-seqs` and YAML `llm.concurrency` only before freezing the study.

The profile uses the single-model mistral.rs API alias `default`, strict JSON Schema,
safe numeric transport, and the tokenizer-specific compact-JSON guard:

```yaml
llm:
  base_url: http://127.0.0.1:8000/v1
  model: default
  structured_output: true
  grammar_safe_numbers: true
  tokenizer: google/gemma-4-E4B-it
```

Do not launch `mistralrs serve --agent`: SwarmWorld needs model inference only and never
delegates shell, Python, filesystem, or network tools to the inference server.

## 3. Option B: vLLM

vLLM should run in its own environment because its compiled accelerator and PyTorch
packages need to match. The SwarmWorld process remains in the `PyTorch` conda
environment and communicates with vLLM over HTTP.

On a supported Linux accelerator host:

```bash
uv venv --python 3.12 .venv-vllm
source .venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto
```

Start Gemma 4 E4B and give it the stable API alias used by the checked-in profile:

```bash
vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name swarm-gemma-4-e4b \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling
```

For a remote inference host, bind deliberately, add authentication/TLS at the network
boundary, and change only `llm.base_url` in a copied profile. If vLLM is started with
`--api-key`, export the same value as `VLLM_API_KEY` before starting SwarmWorld.

The vLLM profile uses:

```yaml
llm:
  base_url: http://127.0.0.1:8000/v1
  model: swarm-gemma-4-e4b
  api_key_env: VLLM_API_KEY
  structured_output: true
```

vLLM performs canonical JSON-schema constrained decoding directly, so the
mistral.rs-specific numeric and whitespace transport guards are disabled.

## 4. Verify the actual Responses contract

First verify the advertised model name:

```bash
curl http://127.0.0.1:8000/v1/models
```

Then probe the same Responses API and strict-output shape used by SwarmWorld. Set
`SWARM_MODEL=default` for mistral.rs or
`SWARM_MODEL=swarm-gemma-4-e4b` for vLLM:

```bash
export SWARM_MODEL="default"

curl http://127.0.0.1:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${SWARM_MODEL}\",
    \"input\": [{\"role\": \"user\", \"content\": \"Return ok=true.\"}],
    \"max_output_tokens\": 64,
    \"store\": false,
    \"text\": {
      \"format\": {
        \"type\": \"json_schema\",
        \"name\": \"swarmworld_probe\",
        \"strict\": true,
        \"schema\": {
          \"type\": \"object\",
          \"properties\": {\"ok\": {\"type\": \"boolean\"}},
          \"required\": [\"ok\"],
          \"additionalProperties\": false
        }
      }
    }
  }"
```

A successful result has `status: "completed"` and an `output_text` content part
containing `{"ok":true}`. `biofoundry doctor` validates the YAML and local Python
dependencies; it does not make a paid or local model request, so this probe is the
endpoint check.

## 5. Run a live local-model society

Leave the inference server running. In a second terminal:

```bash
conda activate PyTorch
cd SwarmWorld

biofoundry serve \
  --config configs/mistralrs-gemma-4-e4b-technology-ecology-50.yaml \
  --record runs/mistralrs-gemma-4-e4b-live.jsonl.gz
```

For vLLM, replace the config and trace name:

```bash
biofoundry serve \
  --config configs/vllm-gemma-4-e4b-technology-ecology-50.yaml \
  --record runs/vllm-gemma-4-e4b-live.jsonl.gz
```

In a third terminal:

```bash
cd SwarmWorld/web
npm ci
npm run dev
```

Open the exact URL printed by Vite.

## 6. Run the matched validity study

Use a distinct output directory for each model/backend. For mistral.rs:

```bash
biofoundry research-study \
  --config configs/mistralrs-gemma-4-e4b-technology-ecology-50.yaml \
  --policy llm \
  --conditions full no-communication no-program-forking independent-search \
  --population-sizes 4 16 \
  --world-scaling fixed \
  --decision-schedule fixed \
  --seeds 3001 3002 \
  --ticks 400 \
  --held-out-evaluation-seeds 9001 9002 9003 9004 \
  --output-dir runs/gemma-4-e4b-mistralrs-validity-pilot
```

For vLLM, change the config and output directory:

```bash
biofoundry research-study \
  --config configs/vllm-gemma-4-e4b-technology-ecology-50.yaml \
  --policy llm \
  --conditions full no-communication no-program-forking independent-search \
  --population-sizes 4 16 \
  --world-scaling fixed \
  --decision-schedule fixed \
  --seeds 3001 3002 \
  --ticks 400 \
  --held-out-evaluation-seeds 9001 9002 9003 9004 \
  --output-dir runs/gemma-4-e4b-vllm-validity-pilot
```

Analyze only after `study-summary.json` exists:

```bash
biofoundry analyze-study \
  runs/gemma-4-e4b-mistralrs-validity-pilot/study-summary.json \
  --output-dir runs/gemma-4-e4b-mistralrs-validity-pilot/analysis
```

## 7. Experimental interpretation

- Use one backend and one frozen server configuration across all conditions in a study.
- Record the model repository revision, server version, quantization, context length,
  maximum sequences, hardware, SwarmWorld commit, and resolved study manifest.
- Quantized and native-precision runs are different model conditions.
- A vLLM/mistral.rs comparison measures the complete inference stack, not merely the
  Gemma checkpoint; kernel precision, batching, constrained decoding, and sampling may
  differ.
- Do not pool Gemma 4 E4B results with Luna results. Treat model family/capacity as a
  separate factor and preserve paired world seeds when comparing them.
- Increase concurrency for throughput only before freezing a confirmatory protocol.
  Concurrency does not grant extra scheduled decisions, but it can change wall time and
  may expose backend-specific sampling behavior.
