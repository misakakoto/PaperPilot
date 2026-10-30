# Architecture

PaperPilot has a local workbench/API, task-specific adapters, a shared durable
runtime, registered tools and persistent state. Ordinary reading, active evidence
gathering, prototype development and imported-result diagnosis are separate paths.
Role names denote responsibilities; this is not a claim of parallel autonomous agents.

## Evidence and context

The active path separates paper discovery, evidence navigation and original
reading. P handles identify papers, S handles locate source units, R handles
identify actual read ranges and Q handles select exact quotes. Previews cannot
become citations. The ledger tracks requirements, proposed support and gaps;
literal source checks do not establish semantic entailment.

ContextCompiler assembles the current goal, constraints, dependencies and unresolved
feedback. Input, output and protected reserves are checked against cumulative
usage; unknown external attempts hold budget. Final evidence views select complete
source quotes with explicit omissions. Full originals stay in local storage.

## Tasks, changes and recovery

EpisodeRunner persists model decisions and local tool receipts separately from
state submission. Saved known responses can be applied after interruption without
another model request. Unknown external results stop progression rather than being
automatically resent. This guarantees neither remote-service exactly-once nor
unlimited context.

Prototype tasks form a dependency graph across reading, design, implementation,
validation and review. ChangeImpact previews affected tasks and reusable artifacts.
Version changes invalidate dependent results without deleting prior artifacts or
resetting spending. A receipt must match the current candidate and validation contract.

## Execution and limitations

Selected reading evidence can be handed to a separately submitted prototype task.
Frozen candidates require specific approval. The prototype Docker path disables
network/local fallback and applies resource/time constraints; its validator checks
engineering interfaces, not scientific quality. No corpus, weights or trained
checkpoint is included in the repository.

Active evidence is opt-in; ordinary QA does not automatically switch to it.
The retrieval backend is selected explicitly by the task configuration.
