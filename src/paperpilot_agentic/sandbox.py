from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .models import ExperimentProtocol, SandboxPolicy, utc_now

_SECRET_RE = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL)", re.I)

try:  # Unix-only resource limits; Docker remains the preferred isolation backend.
    import resource
except Exception:  # pragma: no cover - Windows
    resource = None


@dataclass(slots=True)
class ExperimentResult:
    experiment_id: str
    status: str
    backend: str
    exit_code: int | None
    timed_out: bool
    started_at: str
    completed_at: str
    duration_seconds: float
    protocol_hash: str
    preregistration_path: str
    result_path: str
    stdout_path: str
    stderr_path: str
    metrics: dict[str, Any] = field(default_factory=dict)
    criteria: list[dict[str, Any]] = field(default_factory=list)
    accepted: bool | None = None
    source_hash_before: str = ""
    source_hash_after: str = ""
    source_unchanged: bool = True
    workspace_hash_after: str = ""
    network_isolated: bool = False
    isolation_notes: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    seed_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hash_tree(root: Path, *, ignore: Iterable[str] = ()) -> str:
    ignored = set(ignore)
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
        rel = path.relative_to(root)
        if any(part in ignored for part in rel.parts):
            continue
        digest.update(rel.as_posix().encode("utf-8"))
        if path.is_symlink():
            digest.update(b"LINK")
            digest.update(os.readlink(path).encode("utf-8", errors="replace"))
        elif path.is_file():
            digest.update(b"FILE")
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError:
                digest.update(b"UNREADABLE")
        elif path.is_dir():
            digest.update(b"DIR")
    return digest.hexdigest()


def _copy_source(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise PermissionError(
                f"Experiment source contains a symbolic link and cannot be copied safely: {path}"
            )
    ignore = shutil.ignore_patterns(
        ".git",
        ".paperpilot_agentic",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
    )
    shutil.copytree(source, destination, symlinks=False, ignore=ignore)


def _tree_size(*roots: Path) -> int:
    total = 0
    for root in roots:
        if root.is_file():
            with contextlib.suppress(OSError):
                total += root.stat().st_size
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                with contextlib.suppress(OSError):
                    total += path.stat().st_size
    return total


def _terminate_process(process: subprocess.Popen[bytes], *, docker_name: str | None = None) -> None:
    if docker_name:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "kill", docker_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
    if process.poll() is not None:
        return
    if os.name == "posix" and docker_name is None:
        with contextlib.suppress(Exception):
            os.killpg(process.pid, signal.SIGKILL)
    elif os.name == "nt" and docker_name is None:
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
    with contextlib.suppress(Exception):
        process.kill()


def _validate_protocol(protocol: ExperimentProtocol) -> None:
    protected = protocol.policy.read_only_files
    if (not isinstance(protected, (list, tuple)) or len(protected) > 32
            or any(not isinstance(name, str) or len(name) > 240
                   or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name)
                   or any(part in {'.', '..'} for part in name.split('/')) for name in protected)
            or len(set(protected)) != len(protected)):
        raise ValueError('Invalid sandbox read-only file paths')
    if protected and (protocol.policy.backend != 'docker' or protocol.policy.allow_local_fallback):
        raise ValueError('Read-only file protection requires explicit Docker with no local fallback')
    if not protocol.command or not all(isinstance(item, str) and item for item in protocol.command):
        raise ValueError("Experiment command must be a non-empty argv list")
    if len(protocol.command) > 128:
        raise ValueError("Experiment command contains too many arguments")
    executable = Path(protocol.command[0]).name
    allowed = set(protocol.policy.allowed_executables)
    if executable not in allowed:
        raise PermissionError(f"Executable {executable!r} is not allowlisted: {sorted(allowed)}")
    for key, value in protocol.environment.items():
        if _SECRET_RE.search(key):
            raise PermissionError(f"Experiment environment may not inject credential-like variable {key!r}")
        if not isinstance(value, str):
            raise TypeError(f"Environment value for {key!r} must be a string")
    if protocol.policy.network not in {"none", "host"}:
        raise ValueError("Sandbox network policy must be 'none' or 'host'")


def _sanitized_environment(policy: SandboxPolicy, extra: Mapping[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in policy.env_allowlist:
        if key in os.environ and not _SECRET_RE.search(key):
            env[key] = os.environ[key]
    env.setdefault("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.update({str(k): str(v) for k, v in extra.items()})
    env["PAPERPILOT_SANDBOX"] = "1"
    env["PAPERPILOT_NETWORK_POLICY"] = policy.network
    return env


def _preexec(policy: SandboxPolicy):
    def apply() -> None:
        os.setsid()
        if resource is None:
            return
        with contextlib.suppress(Exception):
            resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_seconds, policy.cpu_seconds + 1))
        memory = int(policy.memory_mb) * 1024 * 1024
        with contextlib.suppress(Exception):
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        max_output = int(policy.max_output_mb) * 1024 * 1024
        with contextlib.suppress(Exception):
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_output, max_output))
        with contextlib.suppress(Exception):
            resource.setrlimit(resource.RLIMIT_NPROC, (policy.max_processes, policy.max_processes))
        with contextlib.suppress(Exception):
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        with contextlib.suppress(Exception):
            os.umask(0o077)

    return apply


def _docker_image_available(image: str) -> bool:
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _read_json_metrics(workspace: Path, output: Path, metric_files: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for relative in metric_files:
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Metric path must be relative: {relative}")
        candidates = [workspace / rel, output / rel]
        metric_path = next((path for path in candidates if path.is_file()), None)
        if metric_path is None:
            continue
        target = output / "metrics" / rel.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if metric_path.resolve() != target.resolve():
            shutil.copy2(metric_path, target)
        else:
            target = metric_path
        try:
            parsed = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                metrics.update(dict(parsed))
            else:
                metrics[rel.stem] = parsed
        except Exception as exc:
            metrics[f"{rel.stem}_parse_error"] = f"{type(exc).__name__}: {exc}"
        artifacts.append(
            {
                "kind": "experiment_metrics",
                "path": str(target.resolve()),
                "sha256": _hash_tree(target.parent) if target.parent.is_dir() else "",
                "metadata": {"source_name": relative},
            }
        )
    return metrics, artifacts


def _lookup_metric(metrics: Mapping[str, Any], dotted: str) -> Any:
    value: Any = metrics
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted)
        value = value[part]
    return value


def _evaluate_criteria(metrics: Mapping[str, Any], criteria: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool | None]:
    if not criteria:
        return [], None
    outcomes: list[dict[str, Any]] = []
    for raw in criteria:
        item = dict(raw)
        metric = str(item.get("metric", ""))
        operator = str(item.get("operator", ">="))
        expected = item.get("value")
        try:
            actual = _lookup_metric(metrics, metric)
            if actual is None or expected is None:
                raise ValueError('Metric and target must be available')
            if any(isinstance(value, float) and not math.isfinite(value) for value in (actual, expected)):
                raise ValueError('Non-finite metric or target cannot evaluate a hypothesis')
            if operator == ">=":
                passed = actual >= expected
            elif operator == ">":
                passed = actual > expected
            elif operator == "<=":
                passed = actual <= expected
            elif operator == "<":
                passed = actual < expected
            elif operator == "==":
                tolerance = float(item.get("tolerance", 0.0))
                if not math.isfinite(tolerance) or tolerance < 0:
                    raise ValueError('Tolerance must be finite and non-negative')
                if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                    passed = abs(float(actual) - float(expected)) <= tolerance
                else:
                    passed = actual == expected
            elif operator == "!=":
                passed = actual != expected
            else:
                raise ValueError(f"Unsupported operator {operator!r}")
            error = None
        except Exception as exc:
            actual = None
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        outcomes.append({**item, "actual": actual, "passed": bool(passed), "error": error})
    return outcomes, (None if any(item['error'] for item in outcomes)
                      else all(item["passed"] for item in outcomes))


class SandboxExperimentRunner:
    """Runs preregistered commands in Docker or an explicitly marked local fallback.

    Docker uses a read-only container root, no capabilities, no network, PID/memory/CPU
    limits and a copied workspace. The local fallback still copies the source and applies
    OS resource limits, but cannot guarantee network isolation; this limitation is
    recorded in every result instead of being hidden.
    """

    def __init__(self, *, cancel_requested: Callable[[], bool] | None = None):
        self.cancel_requested = cancel_requested or (lambda: False)

    def run(self, protocol: ExperimentProtocol) -> ExperimentResult:
        if self.cancel_requested():
            raise RuntimeError("Experiment cancelled before launch")
        _validate_protocol(protocol)
        if protocol.seed_sweep:
            from .seed_sweep import run_seed_sweep

            return run_seed_sweep(protocol, self.run)
        source = Path(protocol.source_dir).expanduser().resolve()
        output = Path(protocol.output_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Experiment source directory not found: {source}")
        output.mkdir(parents=True, exist_ok=True)
        experiment_id = "experiment_" + protocol.fingerprint.split(":", 1)[1][:24]
        experiment_output = output / experiment_id
        experiment_output.mkdir(parents=True, exist_ok=True)
        source_hash_before = _hash_tree(
            source,
            ignore={".git", ".paperpilot_agentic", "__pycache__", ".pytest_cache"},
        )
        cached_result_path = experiment_output / "result.json"
        if cached_result_path.is_file():
            try:
                cached_data = json.loads(cached_result_path.read_text(encoding="utf-8"))
                if (
                    cached_data.get("status") == "COMPLETED"
                    and cached_data.get("protocol_hash") == protocol.fingerprint
                    and cached_data.get("source_hash_before") == source_hash_before
                ):
                    return ExperimentResult(**cached_data)
            except Exception:
                # A partial/corrupt result is never trusted; the preregistered run may
                # recover under the persistent tool lease instead.
                pass

        protocol_data = protocol.to_dict()
        preregistration = {
            "experiment_id": experiment_id,
            "registered_at": utc_now(),
            "protocol_hash": protocol.fingerprint,
            "protocol": protocol_data,
        }
        prereg_path = experiment_output / "preregistration.json"
        prereg_path.write_text(json.dumps(preregistration, ensure_ascii=False, indent=2), encoding="utf-8")

        started_at = utc_now()
        started = time.monotonic()
        stdout_path = experiment_output / "stdout.log"
        stderr_path = experiment_output / "stderr.log"
        backend = "local"
        network_isolated = False
        notes: list[str] = []
        exit_code: int | None = None
        timed_out = False
        error: str | None = None

        with tempfile.TemporaryDirectory(prefix="paperpilot-sandbox-") as temp:
            temp_root = Path(temp)
            workspace = temp_root / "workspace"
            _copy_source(source, workspace)
            policy = protocol.policy
            use_docker = policy.backend == "docker" or (
                policy.backend == "auto" and _docker_image_available(policy.docker_image)
            )
            docker_name: str | None = None
            if use_docker:
                if not _docker_image_available(policy.docker_image):
                    raise RuntimeError(
                        "Docker image is unavailable locally; implicit image pulls are disabled"
                    )
                backend = "docker"
                network_isolated = policy.network == "none"
                docker_name = f"paperpilot-{experiment_id[-16:]}"
                command = [
                    "docker",
                    "run",
                    "--rm",
                    f"--name={docker_name}",
                    "--init",
                    "--read-only",
                    "--cap-drop=ALL",
                    "--security-opt=no-new-privileges",
                    f"--network={policy.network}",
                    f"--cpus={max(0.1, policy.cpu_seconds / max(1, policy.timeout_seconds)):.2f}",
                    f"--memory={int(policy.memory_mb)}m",
                    f"--pids-limit={int(policy.max_processes)}",
                    "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
                    f"--volume={workspace}:/workspace:rw",
                    "--workdir=/workspace",
                ]
                for relative in policy.read_only_files:
                    protected_file = workspace / relative
                    if not protected_file.is_file():
                        raise ValueError(f'Missing sandbox read-only file: {relative}')
                    command.append(f'--volume={protected_file}:/workspace/{relative}:ro')
                if policy.read_only_files:
                    notes.append('Host-designated input files mounted read-only: '+', '.join(policy.read_only_files))
                env = _sanitized_environment(policy, protocol.environment)
                # Never pass the Windows host PATH into a Linux container.  Apart
                # from leaking host layout, it makes even image-provided commands
                # such as ``python`` unresolvable (exit 127).  The experiment cannot
                # override PATH because protocol environment keys are allow-listed.
                env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                env["HOME"] = "/tmp/paperpilot-home"
                for key, value in env.items():
                    command.extend(["--env", f"{key}={value}"])
                command.extend([policy.docker_image, *protocol.command])
                preexec = None
            else:
                if policy.backend == "docker":
                    raise RuntimeError(f"Docker image is unavailable locally: {policy.docker_image}")
                if not policy.allow_local_fallback:
                    raise RuntimeError("Docker sandbox unavailable and local fallback is disabled")
                backend = "local-resource-limited"
                network_isolated = False
                if policy.network == "none":
                    notes.append(
                        "Local fallback cannot guarantee network isolation; use a preloaded Docker image for enforced --network=none."
                    )
                command = list(protocol.command)
                env = _sanitized_environment(policy, protocol.environment)
                sandbox_home = workspace / ".home"
                sandbox_home.mkdir(mode=0o700, exist_ok=True)
                env["HOME"] = str(sandbox_home)
                notes.append(
                    "Local fallback applies resource limits and a copied workspace but cannot guarantee host-filesystem isolation; Docker is required for a hard boundary."
                )
                preexec = _preexec(policy) if os.name == "posix" else None
                if os.name != "posix":
                    notes.append("OS resource limits are unavailable on this platform")

            try:
                workspace_size_before = _tree_size(workspace)
                output_limit = max(1, int(policy.max_output_mb)) * 1024 * 1024
                with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                    if self.cancel_requested():
                        raise RuntimeError("Experiment cancelled before launch")
                    process = subprocess.Popen(
                        command,
                        cwd=str(workspace) if backend != "docker" else None,
                        env=env if backend != "docker" else None,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        shell=False,
                        preexec_fn=preexec,
                        creationflags=(
                            subprocess.CREATE_NEW_PROCESS_GROUP
                            if os.name == "nt" and backend != "docker"
                            else 0
                        ),
                    )
                    deadline = time.monotonic() + max(1, policy.timeout_seconds)
                    while True:
                        if self.cancel_requested():
                            error = "Experiment cancelled by user; owned process terminated"
                            _terminate_process(process, docker_name=docker_name)
                            exit_code = process.wait(timeout=15)
                            break
                        generated_size = max(0, _tree_size(workspace) - workspace_size_before)
                        generated_size += _tree_size(stdout_path, stderr_path)
                        if generated_size > output_limit:
                            error = f"Experiment output exceeded limit of {policy.max_output_mb} MiB"
                            notes.append(error)
                            _terminate_process(process, docker_name=docker_name)
                            exit_code = process.wait(timeout=15)
                            break
                        polled = process.poll()
                        if polled is not None:
                            exit_code = polled
                            break
                        if time.monotonic() >= deadline:
                            timed_out = True
                            error = f"Experiment exceeded timeout of {policy.timeout_seconds}s"
                            _terminate_process(process, docker_name=docker_name)
                            exit_code = process.wait(timeout=15)
                            break
                        time.sleep(0.1)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            metrics, artifacts = _read_json_metrics(workspace, experiment_output, protocol.metric_files)
            workspace_hash_after = _hash_tree(workspace, ignore={"__pycache__", ".pytest_cache"})

        source_hash_after = _hash_tree(
            source,
            ignore={".git", ".paperpilot_agentic", "__pycache__", ".pytest_cache"},
        )
        source_unchanged = source_hash_before == source_hash_after
        if not source_unchanged:
            notes.append("Original source directory changed during execution; result is marked unsafe")
        criteria, accepted = _evaluate_criteria(metrics, protocol.acceptance_criteria)
        duration = time.monotonic() - started
        completed_at = utc_now()
        status = "COMPLETED" if exit_code == 0 and not timed_out and error is None and source_unchanged else "FAILED"
        result_path = experiment_output / "result.json"
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=status,
            backend=backend,
            exit_code=exit_code,
            timed_out=timed_out,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=round(duration, 6),
            protocol_hash=protocol.fingerprint,
            preregistration_path=str(prereg_path.resolve()),
            result_path=str(result_path.resolve()),
            stdout_path=str(stdout_path.resolve()),
            stderr_path=str(stderr_path.resolve()),
            metrics=metrics,
            criteria=criteria,
            accepted=accepted,
            source_hash_before=source_hash_before,
            source_hash_after=source_hash_after,
            source_unchanged=source_unchanged,
            workspace_hash_after=workspace_hash_after,
            network_isolated=network_isolated,
            isolation_notes=notes,
            artifacts=[
                *artifacts,
                {"kind": "experiment_preregistration", "path": str(prereg_path.resolve()), "metadata": {}},
                {"kind": "experiment_stdout", "path": str(stdout_path.resolve()), "metadata": {}},
                {"kind": "experiment_stderr", "path": str(stderr_path.resolve()), "metadata": {}},
            ],
            error=error,
        )
        result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        result.artifacts.append(
            {"kind": "experiment_result", "path": str(result_path.resolve()), "metadata": {"status": status}}
        )
        # Rewrite once so the result artifact is itself listed in the manifest.
        result_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result
