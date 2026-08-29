# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately through the repository's GitHub
Security tab by opening a private vulnerability report. Do not open a public issue
until a maintainer has assessed the report.

Include the affected version or commit, environment, minimal reproduction, expected
impact, and any suggested mitigation. Avoid attaching real credentials, private model
outputs, or sensitive trace data.

## Scope

Security reports may concern the Python server and WebSocket interface, browser or
Godot clients, trace parsing, configuration handling, dependency exposure, or unsafe
handling of model-provider credentials. Research-result disputes and ordinary model
behavior are not security vulnerabilities unless they create a concrete security or
privacy risk.

SwarmWorld reads provider credentials from environment variables selected by a local
configuration. Never store credential values in YAML, prompts, traces, terminal
captures, or commits. Run untrusted traces, scenario packages, and model endpoints in
an appropriately isolated environment.
