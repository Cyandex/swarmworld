# Experimental design

For operational commands, start with [QUICKSTART.md](QUICKSTART.md), then follow the
worked [analysis guide](ANALYSIS.md). The current flagship
[cultural-gain scaling protocol](FLAGSHIP_EXPERIMENT.md) compares an
interacting swarm with the best of the same number of isolated searches, applies
condition-honest action schemas, and evaluates frozen societies under unseen stress
schedules. The technological-ecosystem extension is specified in
[TECHNOLOGICAL_ECOSYSTEM_EXPERIMENT.md](TECHNOLOGICAL_ECOSYSTEM_EXPERIMENT.md).

## Rules shared by all experiments

- The independent simulation seed is the statistical unit.
- Prompts, responses, actions, failures, events, snapshots, and state hashes are saved.
- Success is computed by deterministic world state, never by an LLM judge.
- Development and confirmatory seeds are disjoint.
- Functional quality is reported separately from collaboration-dependent provenance.
- Every seed is included, including model errors and zero-artifact episodes.
- Scaling starts only after one causal mechanism is reliable at fixed population size.

## Follow-on experiments

After the open-invention mechanism is established, sweep population size,
communication radius and topology, memory capacity, macroturn interval, resource
scarcity, environmental heterogeneity, agent dropout, model family, and decoding seed.
Additional controls should include a centralized planner with observation parity,
shuffled or replayed messages, and artifact-only stigmergy.

The release intentionally excludes obsolete internal runbooks and generated paper
assets. Completed paper-study manifests and results live in the external dataset
described in [DATA.md](../DATA.md).
