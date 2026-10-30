"""Immutable runtime behavior selection; legacy states are never migrated on load."""

from .prototype_workspace import digest

PROFILE = "research-runtime-v2"
_CONFIG = {
    "version": PROFILE,
    "context_compiler": "task-budget-v1",
    "recovery": "request-bound-v1",
    "patch_output_tokens": 3000,
    "truncated_output_tokens": 5000,
    "max_output_truncations": 2,
}
DEFAULT_PROFILE = "research-runtime-v3"
_PROFILES = {
    PROFILE: _CONFIG,
    DEFAULT_PROFILE: {
        **_CONFIG,
        "version": DEFAULT_PROFILE,
        "context_compiler": "task-budget-v2",
        "retain_unstored_feedback": True,
        "source_object_default_config": True,
    },
}


def attach(state, identity):
    if identity is None:
        return
    if not isinstance(identity, str) or identity not in _PROFILES:
        raise ValueError("Unknown runtime profile")
    config = _PROFILES[identity]
    state["runtime_profile"] = {"id": identity, "config": dict(config), "sha256": digest(config)}


def enabled(state):
    profile = state.get("runtime_profile")
    if profile is None:
        return False
    if not isinstance(profile, dict) or not isinstance(profile.get("id"), str):
        raise ValueError("Runtime profile identity changed or is unknown")
    config = _PROFILES.get(profile["id"])
    if config is None:
        raise ValueError("Runtime profile identity changed or is unknown")
    expected = {"id": profile["id"], "config": config, "sha256": digest(config)}
    if profile != expected:
        raise ValueError("Runtime profile identity changed or is unknown")
    return True


def feature(state, name):
    return enabled(state) and state["runtime_profile"]["config"].get(name, False) is True


def diagnostics(state):
    return {
        key: state.get(key)
        for key in ("runtime_profile", "context_manifest", "change_impact", "recovery_trace")
        if key in state
    }
