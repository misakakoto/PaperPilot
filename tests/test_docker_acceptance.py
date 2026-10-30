from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperpilot_agentic import sandbox
from paperpilot_agentic.docker_acceptance import (
    DockerAcceptanceUnavailable,
    audit_docker_acceptance_result,
    build_docker_acceptance_protocol,
    resolve_local_docker_image,
)
from paperpilot_agentic.models import ExperimentProtocol
from paperpilot_agentic.sandbox import ExperimentResult, SandboxExperimentRunner


IMAGE_ID = "sha256:" + "a" * 64


def _fixture_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "experiment.py").write_text("print('fixture')\n", encoding="utf-8")
    return source


def test_docker_acceptance_protocol_is_digest_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    protocol = build_docker_acceptance_protocol(
        source_dir=_fixture_source(tmp_path),
        output_dir=tmp_path / "output",
        image_id=IMAGE_ID,
    )

    assert protocol.policy.backend == "docker"
    assert protocol.policy.docker_image == IMAGE_ID
    assert protocol.policy.network == "none"
    assert protocol.policy.allow_local_fallback is False
    assert protocol.policy.allowed_executables == ("python",)


def test_docker_image_resolution_never_pulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        calls.append(command)
        if command[1] == "version":
            return SimpleNamespace(returncode=0, stdout="29.1.3\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=IMAGE_ID + "\n", stderr="")

    monkeypatch.setattr("paperpilot_agentic.docker_acceptance.shutil.which", lambda _: "docker")
    monkeypatch.setattr("paperpilot_agentic.docker_acceptance.subprocess.run", fake_run)

    image_id, version = resolve_local_docker_image("python:3.11-slim")

    assert image_id == IMAGE_ID
    assert version == "29.1.3"
    assert calls == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "image", "inspect", "python:3.11-slim", "--format", "{{.Id}}"],
    ]
    assert all("pull" not in call for command in calls for call in command)


def test_docker_image_resolution_fails_closed_without_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("paperpilot_agentic.docker_acceptance.shutil.which", lambda _: "docker")
    monkeypatch.setattr(
        "paperpilot_agentic.docker_acceptance.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    )

    with pytest.raises(DockerAcceptanceUnavailable, match="daemon"):
        resolve_local_docker_image("python:3.11-slim")


def test_production_runner_refuses_implicit_image_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fixture_source(tmp_path)
    protocol = ExperimentProtocol.from_dict(
        {
            "name": "no-pull",
            "command": ["python", "experiment.py"],
            "source_dir": str(source),
            "output_dir": str(tmp_path / "output"),
            "policy": {
                "backend": "docker",
                "docker_image": "python:3.11-slim",
                "network": "none",
                "allow_local_fallback": False,
            },
        }
    )
    monkeypatch.setattr(sandbox, "_docker_image_available", lambda _image: False)

    with pytest.raises(RuntimeError, match="implicit image pulls are disabled"):
        SandboxExperimentRunner().run(protocol)


def test_docker_acceptance_audit_requires_all_attestations(tmp_path: Path) -> None:
    result = ExperimentResult(
        experiment_id="experiment_fixture",
        status="COMPLETED",
        backend="docker",
        exit_code=0,
        timed_out=False,
        started_at="2026-08-29T00:00:00+00:00",
        completed_at="2026-08-29T00:00:01+00:00",
        duration_seconds=1.0,
        protocol_hash="sha256:fixture",
        preregistration_path=str(tmp_path / "protocol.json"),
        result_path=str(tmp_path / "result.json"),
        stdout_path=str(tmp_path / "stdout.log"),
        stderr_path=str(tmp_path / "stderr.log"),
        metrics={
            "baseline_accuracy": 0.5,
            "candidate_accuracy": 1.0,
            "improvement": 0.5,
            "seed": 7,
            "samples": 256,
        },
        criteria=[{"passed": True}],
        accepted=True,
        source_unchanged=True,
        network_isolated=True,
    )

    assert audit_docker_acceptance_result(result)["status"] == "PASSED"
    result.network_isolated = False
    audit = audit_docker_acceptance_result(result)
    assert audit["status"] == "FAILED"
    assert audit["checks"]["network_isolated"] is False


def test_docker_runner_uses_container_path_not_windows_host_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fixture_source(tmp_path)
    (source / "metrics.json").write_text("{}", encoding="utf-8")
    protocol = build_docker_acceptance_protocol(
        source_dir=source,
        output_dir=tmp_path / "output",
        image_id=IMAGE_ID,
    )
    captured: dict[str, Any] = {}

    class CompletedProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(command: list[str], **_kwargs: Any) -> CompletedProcess:
        captured["command"] = command
        return CompletedProcess()

    monkeypatch.setattr(sandbox, "_docker_image_available", lambda _image: True)
    monkeypatch.setattr(sandbox.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("PATH", r"C:\\Windows\\System32;C:\\Program Files\\Docker")
    SandboxExperimentRunner().run(protocol)

    command = captured["command"]
    path_env = next(
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--env" and command[index + 1].startswith("PATH=")
    )
    assert path_env == "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    assert "C:\\Windows" not in path_env
