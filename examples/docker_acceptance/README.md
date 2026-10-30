# Docker acceptance fixture

This deterministic, dependency-free XOR interaction ablation exists only to verify
PaperPilot's production experiment boundary. It writes one `metrics.json` file and
makes no network calls. The acceptance launcher requires a preloaded image resolved
to an immutable local image ID, Docker `--network=none`, a read-only container root,
no capabilities, no-new-privileges, and explicit CPU/memory/PID/output limits.

It is not a scientific benchmark and its 100% candidate accuracy must never be used
as a model-quality claim.
