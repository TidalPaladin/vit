"""Contract tests for the repository's GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).parents[1]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
MAKEFILE = REPOSITORY_ROOT / "Makefile"
CIRCLECI_CONFIGURATION = REPOSITORY_ROOT / ".circleci" / "config.yml"
CODECOV_CONFIGURATION = REPOSITORY_ROOT / "codecov.yml"
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV_ACTION = "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
UPLOAD_ARTIFACT_ACTION = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
UV_VERSION = "0.11.28"
UV_CHECKSUM = "e490a6464492183c5d4534a5527fb4440f7f2bb2f228162ad7e4afe076dc0224"


def load_workflow(name: str) -> dict[str, Any]:
    """Load a workflow as a mapping."""
    with (WORKFLOW_DIRECTORY / name).open() as workflow_file:
        workflow = yaml.safe_load(workflow_file)
    assert isinstance(workflow, dict)
    return workflow


def workflow_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every step from every job in a workflow."""
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_ci_workflow_contract() -> None:
    workflow = load_workflow("ci.yml")

    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert workflow["on"]["pull_request"]["branches"] == ["master"]
    assert workflow["on"]["push"]["branches"] == ["master"]
    assert workflow["concurrency"]["group"] == "ci-${{ github.event.pull_request.number || github.ref }}"
    assert workflow["concurrency"]["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"

    jobs = workflow["jobs"]
    assert jobs["quality"]["name"] == "Quality"
    assert jobs["quality"]["runs-on"] == "ubuntu-24.04"
    assert jobs["quality"]["timeout-minutes"] == 15
    assert jobs["tests"]["strategy"]["matrix"]["include"] == [
        {"python-version": "3.11", "artifact-suffix": "py311"},
        {"python-version": "3.14", "artifact-suffix": "py314"},
    ]
    assert jobs["tests"]["runs-on"] == "ubuntu-24.04"
    assert jobs["tests"]["timeout-minutes"] == 25
    assert jobs["required"]["name"] == "Required"
    assert jobs["required"]["needs"] == ["quality", "tests"]
    assert jobs["required"]["if"] == "${{ always() }}"
    assert jobs["required"]["runs-on"] == "ubuntu-24.04"
    assert jobs["required"]["timeout-minutes"] == 5

    required_step = jobs["required"]["steps"][0]
    assert required_step["env"] == {
        "QUALITY_RESULT": "${{ needs.quality.result }}",
        "TESTS_RESULT": "${{ needs.tests.result }}",
    }
    assert '"$QUALITY_RESULT" = success' in required_step["run"]
    assert '"$TESTS_RESULT" = success' in required_step["run"]

    artifact_steps = [step for step in workflow_steps(workflow) if step.get("uses") == UPLOAD_ARTIFACT_ACTION]
    assert len(artifact_steps) == 1
    assert artifact_steps[0]["with"]["name"] == "coverage-${{ matrix.artifact-suffix }}-${{ github.sha }}"
    assert artifact_steps[0]["with"]["retention-days"] == 7


@pytest.mark.parametrize("workflow_name", ["ci.yml", "dependency-audit.yml", "production-build.yml"])
def test_workflows_pin_actions_and_harden_checkout(workflow_name: str) -> None:
    workflow = load_workflow(workflow_name)
    steps = workflow_steps(workflow)

    for step in steps:
        action = step.get("uses")
        if action is None:
            continue
        assert action in {CHECKOUT_ACTION, SETUP_UV_ACTION, UPLOAD_ARTIFACT_ACTION}
        if action == CHECKOUT_ACTION:
            assert step["with"]["persist-credentials"] is False
        elif action == SETUP_UV_ACTION:
            assert step["with"]["version"] == UV_VERSION
            assert step["with"]["checksum"] == UV_CHECKSUM


def test_ci_cache_is_restored_for_forks_but_saved_only_by_trusted_writers() -> None:
    workflow = load_workflow("ci.yml")
    setup_steps = [step for step in workflow_steps(workflow) if step.get("uses") == SETUP_UV_ACTION]

    assert setup_steps
    for step in setup_steps:
        assert step["with"]["enable-cache"] is True
        assert step["with"]["restore-cache"] is True
        assert step["with"]["save-cache"] == (
            "${{ github.event_name != 'pull_request' || "
            "github.event.pull_request.head.repo.full_name == github.repository }}"
        )
        assert step["with"]["cache-dependency-glob"] == "pyproject.toml\nuv.lock"
        assert "uv-0.11.28" in step["with"]["cache-suffix"]
        assert "torch-2.13.0" in step["with"]["cache-suffix"]


def test_dependency_health_workflow_contract() -> None:
    workflow = load_workflow("dependency-audit.yml")

    assert workflow["name"] == "Dependency health"
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"] == [{"cron": "17 6 * * 1"}]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["validation_id"]["required"] is False
    assert workflow["concurrency"] == {"group": "dependency-health", "cancel-in-progress": False}

    jobs = workflow["jobs"]
    assert jobs["security-audit"]["name"] == "Security audit"
    assert jobs["security-audit"]["runs-on"] == "ubuntu-24.04"
    assert jobs["security-audit"]["timeout-minutes"] == 20
    assert jobs["deprecation-report"]["name"] == "Deprecation report"
    assert jobs["deprecation-report"]["runs-on"] == "ubuntu-24.04"
    assert jobs["deprecation-report"]["timeout-minutes"] == 25

    artifact_steps = [step for step in workflow_steps(workflow) if step.get("uses") == UPLOAD_ARTIFACT_ACTION]
    assert {step["with"]["name"] for step in artifact_steps} == {
        "dependency-security-${{ github.run_id }}",
        "dependency-deprecations-${{ github.run_id }}",
    }
    assert all(step["with"]["retention-days"] == 7 for step in artifact_steps)


def test_production_build_workflow_contract() -> None:
    workflow = load_workflow("production-build.yml")

    assert workflow["name"] == "Production build"
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["on"]) == {"schedule", "workflow_dispatch"}
    assert workflow["on"]["schedule"] == [{"cron": "37 5 * * 0"}]
    assert workflow["on"]["workflow_dispatch"]["inputs"]["validation_id"]["required"] is False
    assert workflow["concurrency"] == {"group": "production-build", "cancel-in-progress": False}

    jobs = workflow["jobs"]
    assert jobs["distributions"]["name"] == "Distributions"
    assert jobs["distributions"]["runs-on"] == "ubuntu-24.04"
    assert jobs["distributions"]["timeout-minutes"] == 30
    assert jobs["cpu-compile"]["name"] == "CPU compile"
    assert jobs["cpu-compile"]["runs-on"] == "ubuntu-24.04"
    assert jobs["cpu-compile"]["timeout-minutes"] == 30

    compile_steps = [step for step in jobs["cpu-compile"]["steps"] if step.get("run") == "make test-compile-cpu"]
    assert len(compile_steps) == 1
    assert compile_steps[0]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert compile_steps[0]["env"]["TORCHDYNAMO_DISABLE"] == "0"

    artifact_steps = [step for step in workflow_steps(workflow) if step.get("uses") == UPLOAD_ARTIFACT_ACTION]
    assert len(artifact_steps) == 1
    assert artifact_steps[0]["with"]["name"] == "vit-distributions-${{ github.sha }}"
    assert artifact_steps[0]["with"]["retention-days"] == 7


def test_circleci_and_codecov_are_removed_after_cutover() -> None:
    assert not CIRCLECI_CONFIGURATION.exists()
    assert not CODECOV_CONFIGURATION.exists()


def test_makefile_selects_only_cpu_compile_tests() -> None:
    makefile = MAKEFILE.read_text()

    assert "test-compile-cpu:" in makefile
    assert 'TORCHDYNAMO_DISABLE="0"' in makefile
    assert 'CUDA_VISIBLE_DEVICES=""' in makefile
    assert '-m "compile and not cuda"' in makefile
