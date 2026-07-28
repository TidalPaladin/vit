import subprocess
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).parents[1]
DEPENDENCY_AUDIT_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "dependency-audit.yml"


def test_dependency_audit_workflow_is_scheduled_and_covers_supported_python_bounds() -> None:
    workflow = yaml.safe_load(DEPENDENCY_AUDIT_WORKFLOW.read_text())

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["schedule"] == [{"cron": "17 6 * * 1"}]
    assert set(workflow["jobs"]) == {"security-audit", "deprecation-report"}
    security_command = next(
        step["run"]
        for step in workflow["jobs"]["security-audit"]["steps"]
        if step.get("name") == "Run every security scanner"
    )
    assert "tools/run_security_audit.py" in security_command


def test_dependency_audit_target_runs_scanner_on_requested_python() -> None:
    result = subprocess.run(
        ["make", "--dry-run", "audit-dependencies", "PYTHON_VERSION=3.11"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert '--python "3.11"' in result.stdout
    assert "--only-group ci-security" in result.stdout
    assert "pip-audit" in result.stdout


@pytest.mark.parametrize(("scanner", "should_succeed"), [("true", True), ("false", False)])
def test_dependency_audit_target_propagates_scanner_status(scanner: str, should_succeed: bool) -> None:
    result = subprocess.run(
        ["make", "audit-dependencies", "UV=true", f"PIP_AUDIT={scanner}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode == 0) is should_succeed, result.stderr


def test_dependency_audit_target_propagates_export_failure() -> None:
    """The scanner must not mask a failed locked-dependency export."""
    result = subprocess.run(
        ["make", "audit-dependencies", "UV=false", "PIP_AUDIT=true"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
