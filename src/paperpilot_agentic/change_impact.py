"""Explain dependency invalidation before committing a revision."""

from .prototype_workspace import digest


def analyze(original, updated, kind):
    before, after = original["work"], updated["work"]
    roots = []
    for key, task in after["tasks"].items():
        old = before["tasks"].get(key)
        if old is None or any(
            old.get(k) != task.get(k) for k in ("goal", "acceptance", "capability", "depends_on")
        ):
            roots.append(key)
    affected, paths = set(roots), {key: [key] for key in roots}
    while True:
        new = {
            key
            for key, t in after["tasks"].items()
            if key not in affected and set(t["depends_on"]) & affected
        }
        if not new:
            break
        for key in sorted(new):
            parent = sorted(set(after["tasks"][key]["depends_on"]) & affected)[0]
            paths[key] = paths[parent] + [key]
        affected |= new
    invalidated, reused = [], []
    for key, task in before["tasks"].items():
        if task["status"] != "DONE":
            continue
        current = after["tasks"].get(key, {})
        if current.get("status") == "DONE" and current.get("output") == task["output"]:
            reused.append(task["output"])
        else:
            invalidated.append(key)
    return {
        "version": "change-impact-v1",
        "kind": kind,
        "expected_revision": before["revision"],
        "next_revision": after["revision"],
        "changed_task_ids": sorted(roots),
        "affected_task_ids": sorted(affected),
        "invalidated_task_ids": sorted(invalidated),
        "reusable_artifact_ids": sorted(set(reused)),
        "propagation_paths": paths,
        "before_work_sha256": digest(before),
        "after_work_sha256": digest(after),
    }
