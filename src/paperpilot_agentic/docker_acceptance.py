"""Fixed, zero-API acceptance protocol for the production Docker sandbox."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .models import ExperimentProtocol, SandboxPolicy
from .sandbox import ExperimentResult

_IMAGE_ID_RE = re.compile(r"sha256:[a-f0-9]{64}")


class DockerAcceptanceUnavailable(RuntimeError):
    """The host cannot currently provide the required local Docker boundary."""


def resolve_local_docker_image(image: str) -> tuple[str, str]:
    """Resolve a preloaded image to an immutable ID without pulling anything."""

    if not shutil.which("docker"):
        raise DockerAcceptanceUnavailable("Docker CLI is not installed")
    try:
        version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        raise DockerAcceptanceUnavailable("Docker daemon preflight failed") from exc
    if version.returncode != 0 or not version.stdout.strip():
        raise DockerAcceptanceUnavailable("Docker daemon is unavailable")
    inspected = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    image_id = inspected.stdout.strip()
    if inspected.returncode != 0 or not _IMAGE_ID_RE.fullmatch(image_id):
        raise DockerAcceptanceUnavailable(
            "Required Docker image is not preloaded; implicit pull is forbidden"
        )
    return image_id, version.stdout.strip()


def build_docker_acceptance_protocol(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    image_id: str,
    seed: int = 7,
) -> ExperimentProtocol:
    """Create a digest-bound, preregistered XOR interaction ablation."""

    if not _IMAGE_ID_RE.fullmatch(str(image_id)):
        raise ValueError("Docker acceptance requires an immutable local image ID")
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir() or not (source / "experiment.py").is_file():
        raise FileNotFoundError("Docker acceptance fixture is incomplete")
    return ExperimentProtocol(
        name="paperpilot-docker-xor-interaction-ablation",
        command=[
            "python",
            "experiment.py",
            "--seed",
            str(int(seed)),
            "--output",
            "metrics.json",
        ],
        source_dir=str(source),
        output_dir=str(output),
        metric_files=["metrics.json"],
        acceptance_criteria=[
            {"metric": "baseline_accuracy", "operator": "<=", "value": 0.5},
            {"metric": "candidate_accuracy", "operator": ">=", "value": 1.0},
            {"metric": "improvement", "operator": ">=", "value": 0.5},
        ],
        hypothesis=(
            "A preregistered interaction feature solves the fixed XOR slice while "
            "the single-feature baseline remains at chance."
        ),
        seed=int(seed),
        policy=SandboxPolicy(
            backend="docker",
            docker_image=str(image_id),
            timeout_seconds=60,
            cpu_seconds=30,
            memory_mb=256,
            max_output_mb=4,
            max_processes=32,
            network="none",
            allow_local_fallback=False,
            allowed_executables=("python",),
            env_allowlist=("LANG", "LC_ALL"),
        ),
    )


def audit_docker_acceptance_result(
    result: ExperimentResult | Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the real result attests every production boundary."""

    value = result.to_dict() if isinstance(result, ExperimentResult) else dict(result)
    checks = {
        "completed": str(value.get("status")) == "COMPLETED",
        "docker_backend": str(value.get("backend")) == "docker",
        "network_isolated": value.get("network_isolated") is True,
        "source_unchanged": value.get("source_unchanged") is True,
        "criteria_accepted": value.get("accepted") is True,
        "exit_code_zero": int(value.get("exit_code", -1)) == 0,
        "not_timed_out": value.get("timed_out") is False,
        "no_runtime_error": not value.get("error"),
    }
    metrics = dict(value.get("metrics") or {})
    checks.update(
        {
            "baseline_metric": metrics.get("baseline_accuracy") == 0.5,
            "candidate_metric": metrics.get("candidate_accuracy") == 1.0,
            "improvement_metric": metrics.get("improvement") == 0.5,
            "fixed_seed": metrics.get("seed") == 7,
            "fixed_sample_count": metrics.get("samples") == 256,
        }
    )
    return {
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "checks": checks,
        "metrics": metrics,
    }
