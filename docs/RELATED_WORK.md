# Related work and novelty boundary

Scope note: this is a design-positioning map, not a claim that an exhaustive literature
search proves absolute novelty. The paper should date its final search, document search
terms, and compare claims against the primary sources linked below.

## Design requirements taken from prior environments

The experiment adopts mechanisms that earlier systems showed cannot be treated as
optional implementation details:

- [Voyager](https://voyager.minedojo.org/) motivates explicit environment feedback,
  self-verification, and reusable executable skills. SwarmWorld applies these to
  multi-agent material artifacts rather than one Minecraft agent.
- [Generative Agents / Smallville](https://arxiv.org/abs/2304.03442) motivates a persistent
  observation-memory-planning loop. SwarmWorld separates private working, episodic,
  notebook, spatial, and cultural memory so each layer can be ablated.
- [Project Sid](https://arxiv.org/abs/2411.00114) emphasizes action awareness and
  social awareness in large agent societies. SwarmWorld exposes expected-versus-
  observed action outcomes, immediate recovery, local peer state, and a task board.
  Sid's reported roles are inferred after the run from agents' self-generated goals;
  SwarmWorld therefore treats specialization as a permutation-tested action statistic,
  not as an LLM-labeled role narrative.
- [DiscoveryWorld](https://github.com/allenai/discoveryworld) scores both task
  completion and component scientific procedures. SwarmWorld likewise uses explicit
  milestones, test results, critical causal records, and a separate outcome score.
  DiscoveryWorld also withholds fine-grained scorecard state from the acting agent.
  SwarmWorld follows that separation: exact thresholds and global completion counts
  remain evaluator-only.
- [SOTOPIA](https://arxiv.org/abs/2310.11667) motivates multidimensional evaluation
  rather than a single subjective success score.
- [GovSim](https://arxiv.org/abs/2404.16698) motivates communication ablations and
  long-horizon consequences. Here messages, public memory, and material pooling are
  independently removable.
- [MultiAgentBench](https://arxiv.org/abs/2503.01935) motivates milestone-based
  key-performance indicators and later comparison of communication topologies.
- [ScienceAgentBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html)
  motivates scoring each component of a scientific workflow rather than accepting a
  plausible final answer.

These precedents determine what the agents must actually learn: grounded action
affordances, a process-property relation from experiments, useful social allocation,
and an executable controller whose later world effects can be measured.

## Code-generating agents

[Voyager](https://arxiv.org/abs/2305.16291) is the closest embodied precedent: a
single Minecraft agent writes executable skills, tests them, and stores successful
programs in a reusable library. BioFoundry differs by studying a decentralized
population, persistent programmed material artifacts, environmental tick dynamics,
and causal contributor provenance.

[AutoGen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/command-line-code-executors.html)
supports conversational agents that write Python or shell programs executed as
local or Docker processes. BioFoundry deliberately does not offer general host-code
execution; agent programs are capability-bounded world objects.

[MetaGPT](https://arxiv.org/abs/2308.00352) and related software-company systems
coordinate agents to produce software artifacts. Their code is a task deliverable,
not an autonomous material object embedded in a spatial ecology.

## Generative societies

[Generative Agents / Smallville](https://arxiv.org/abs/2304.03442) demonstrates memory,
reflection, planning, and emergent social coordination in a simulated town.
[Concordia](https://github.com/google-deepmind/concordia) uses a Game Master to
resolve natural-language actions. BioFoundry instead uses an explicit deterministic
scientific state machine, simultaneous environment actions, and executable artifact
dynamics.

[AgentVerse](https://arxiv.org/abs/2308.10848),
[MetaGPT](https://arxiv.org/abs/2308.00352), and
[Magentic-One](https://arxiv.org/abs/2411.04468) are useful controls for engineered
coordination: they dynamically group agents, prescribe software-company roles and
procedures, or rely on a central orchestrator. They do not establish spontaneous
specialization by initially homogeneous, locally observing agents, so those mechanisms
belong in centralized or role-assigned baselines rather than the treatment condition.

## Scientific worlds and multi-agent interfaces

[DiscoveryWorld](https://github.com/allenai/discoveryworld) evaluates scientific
discovery agents in a purpose-built environment. BioFoundry does not reuse its
scientific framing or scenarios. The focus is decentralized technological culture
and causal composition rather than individual task completion.

[PettingZoo](https://pettingzoo.farama.org/main/api/parallel/) supplies only the
multi-agent API contract. It is infrastructure and not a novelty claim.

## Focused novelty hypothesis

The potentially new experimental object is the conjunction of:

1. homogeneous LLM agents with local biological observations;
2. decentralized communication and multiple separable memory substrates;
3. typed processing recipes that transform conserved feedstocks;
4. agent-authored bounded programs installed in persistent material artifacts;
5. autonomous artifact execution on every later world tick;
6. reuse, modification, and spatial/ecological selection of those artifacts;
7. complete provenance and counterfactual removal of claimed contributors.

The strongest claimed contribution should be narrower than “LLM swarms are novel”:
SwarmWorld tests *causal closure* across epistemic, material, and executable layers.
A successful result must link personally grounded publications from distinct agents to
a conserved physical build, an agent-authored controller, and later measured field
behavior. Communication, depot, feedback, navigation, and single-agent controls then
ask which links are actually necessary.

The initial open-invention experiment makes the conjunction more specific: agents are
not offered a catalog of technologies or geometries. They must author the material's
identity, biological analogies, architecture parameters, processing recipe, predicted
effects, and later-tick program. A causally distributed artifact must be tested, carry
personally grounded evidence from distinct authors, and drive measured field services.
Physical contributors, public evidence authors, explicit combination, and depot use
remain separate diagnostics rather than mandatory ceremony. Functional success remains
separately measurable by a single agent, preventing the stricter lineage definition
from manufacturing an apparent swarm advantage.

Individual ingredients are established. A focused search cannot prove absolute
novelty, so publications should present this as a testable conjunction and compare
directly against the closest embodied-code and generative-society precedents.
