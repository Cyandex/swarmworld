# Structured action plans

The real-agent configuration uses OpenAI Responses API Structured Outputs rather
than JSON mode. Its request contains:

```json
{
  "text": {
    "format": {
      "type": "json_schema",
      "name": "biofoundry_agent_plan",
      "strict": true,
      "schema": {"...": "complete schema"}
    }
  }
}
```

The authoritative schema lives in
[`src/biofoundry/structured_output.py`](../src/biofoundry/structured_output.py).
It is generated from the same Python enums and artifact-VM constants used by the
simulator, which prevents the model contract from silently drifting away from the
executable contract.

## Root contract

Every response has exactly two root properties:

```json
{
  "research_state": {
    "goal": "self-chosen objective",
    "hypothesis": "falsifiable expectation or empty string",
    "progress": "agent-authored assessment",
    "next_checkpoint": "observable reassessment condition",
    "collaboration_need": "desired peer evidence/help or empty string",
    "evidence_ids": ["stable IDs the agent elects to retain"]
  },
  "plan": ["one to the configured maximum of complete action objects"]
}
```

`research_state` is private notebook state, authored and revised by the model. The
simulator does not fill it, score its wording, or grant capabilities from it. Retaining
the latest update gives the otherwise stateless Responses call reflection/planning
continuity without assigning roles, hypotheses, routes, or recipes in engine code.
`evidence_ids` is bounded to sixteen entries and supplies explicit agent-selected
pointers for experience retrieval and causal audit; invalid or unknown IDs are dropped.

`llm.max_plan_actions` bounds one open-loop macro-plan, not an agent's lifetime actions
or scientific vocabulary. It is configurable from 1 to 16. The 12-agent OpenAI profile
uses 16, the 50-agent technology-ecology profile uses 12, and the local Gemma 4 12B
profile uses 6 to prevent its constrained decoder from expanding an entire research
program into one truncated response. Event-driven or fixed periodic macroturns continue
throughout the episode, and each retained action must remain valid without intervening
evidence.

Every action contains all sixteen fields: `verb`, `direction`, `resource`,
`artifact`, `target_x`, `target_y`, `target_artifact_id`, `target_agent_id`, `reply_to`,
`amount`, `message`, `recipe`, `program`, `insight`, `artifact_spec`, and
`causal_parents`. Model-facing enums use
semantic names rather than opaque integer codes. Irrelevant direction/resource/artifact
fields use `STAY`/`NONE`, amount uses zero, coordinates use `-1`, text and target IDs use
an empty string, optional structured payloads use `null`, and unused causal parents use
an empty array.

The nested schemas are also complete and closed:

- private research state contains the five bounded text fields plus bounded evidence IDs;
- recipes contain typed resource/mass inputs, bounded process steps, an output
  form, and design principles;
- insights contain content, kind, and bounded salience;
- artifact programs contain a name, nullable exact parent program ID, and one to 64
  instructions;
- artifact specifications contain an unrestricted identity and architecture hypothesis
  plus bounded continuous geometry; no finite technology catalog is exposed;
- each instruction is one of five exact shapes: sensor read, constant, copy,
  binary arithmetic/comparison, or actuator invocation.

Every object sets `additionalProperties: false`, and every property is required.
Optional object values are unions with `null`. This is the strict-schema shape
required by the [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

Every non-enum text leaf also has an explicit, generous `maxLength`. This is a
termination constraint, not a technology vocabulary: agents can still name and
describe arbitrary inventions, but a constrained local decoder cannot repeat inside a
legal JSON string until the episode's token ceiling and thereby discard an otherwise
valid plan. Recipe design principles are likewise limited to four short entries;
processing operations, materials, artifact geometry, and executable programs remain
separate fields rather than being hidden in prose. The host validates these bounds
identically for local and hosted models.

## Validation boundary

Structured Outputs eliminates syntactic and shape failures: missing fields, prose
instead of JSON, invalid enums, string-valued insights, malformed recipes, and
invalid artifact instruction forms. The host validates decoded responses against
the same schema a second time, protecting local or compatible servers whose strict
decoding implementation differs. The schema does not—and should not—try to encode
the current world state. The deterministic simulator still rejects actions whose
dynamic preconditions fail, such as building without sufficient inventory,
operating at the wrong station, or targeting an absent artifact.
Addressed actions additionally require the target agent to be inside the action's local
radius. A `reply_to` ID must identify a message visible to the acting agent. Every
non-`WAIT` verb may use it: communication forms a response, while a successful world
action forms a request-fulfillment edge and a failed action does not. A fork
parent must exist in that agent's authored, observed, taught, or inherited skill index.

Material enum values are representational vocabulary, not evidence of availability.
The per-agent prompt lists only personally grounded material names. Every independently
proposed recipe input must have been directly observed or personally possessed. A
combined design may instead cite publications whose authors personally grounded a
material. Conserved shared-depot mass enables fabrication but does not itself create
empirical knowledge. `HARVEST` always uses `resource="NONE"` and samples whatever is
present on the current tile. The simulator does not infer that unobserved materials are
absent or prescribe how an agent should react.

All rejections remain recorded in run logs, so experiments can separate model
contract errors from meaningful failures of situated planning. No earlier experimental
result is carried into the new collective-science study.

## Configuration

Enable strict output with:

```yaml
llm:
  structured_output: true
  json_mode: false
```

For mistral.rs 0.8.x, the included local profile additionally enables
`grammar_safe_numbers`. That transport adapter prevents nonterminating constrained
integer/decimal generation by carrying numeric leaves as finite decimal strings. The
policy converts them back and validates the result against the canonical strict schema
before constructing an action. Transport patterns preserve canonical 0--1 bounds for
intensities and geometry controls, preventing decoder-only out-of-range values. OpenAI
and servers without this decoder issue continue
to receive the canonical numeric schema. The local profile also sets
`json_whitespace_logit_bias` and a tokenizer ID. The provider derives tokens that decode
only to whitespace and applies the configured negative bias during strict JSON output;
compact JSON never requires those tokens between fields. This avoids mistral.rs accepting
whitespace forever at a grammar boundary while leaving ordinary content tokens and the
canonical validation boundary intact. The local profiles reserve 4,096 output tokens so
a legal multi-action plan with a publication or artifact specification can close its
JSON object; every individual string and array remains explicitly bounded by the schema.

Action payloads are verb-discriminated. For example, `HARVEST` can only carry
`resource="NONE"`, `MOVE` cannot select `STAY`, an `INSPECT` cannot smuggle in a recipe,
and a `PUBLISH` must carry a complete insight object. This catches static contract errors
at generation time; dynamic failures such as operating away from a station remain world
feedback that agents must learn from.

`json_mode` is retained only for older replay headers and servers that do not
support strict schemas. Do not enable both flags. Strict plans are larger than
single loose actions, so the OpenAI example allocates 4096 output tokens.
