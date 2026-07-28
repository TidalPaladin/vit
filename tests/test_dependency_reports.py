"""Tests for dependency-health report helpers."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from tools.dependency_report import DependencyReportError, build_report
from tools.run_security_audit import Scan, run_scans, scans_succeeded


def test_security_orchestration_runs_every_scanner_before_failing(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    scans = (
        Scan("python-3.11", ("audit", "3.11"), tmp_path / "python-3.11.json"),
        Scan("python-3.14", ("audit", "3.14"), tmp_path / "python-3.14.json"),
        Scan("workflows", ("zizmor",), tmp_path / "workflows.json"),
    )

    def runner(command: tuple[str, ...]) -> CompletedProcess[str]:
        calls.append(command)
        report_path = scans[len(calls) - 1].report_path
        report_path.write_text(json.dumps({"findings": []}))
        return CompletedProcess(command, 1 if len(calls) == 1 else 0)

    results = run_scans(scans, runner=runner)

    assert calls == [scan.command for scan in scans]
    assert [result.exit_code for result in results] == [1, 0, 0]
    assert all(result.report_written for result in results)
    assert not scans_succeeded(results)


def test_security_orchestration_rejects_a_missing_report(tmp_path: Path) -> None:
    scan = Scan("workflows", ("zizmor",), tmp_path / "workflows.json")

    results = run_scans((scan,), runner=lambda command: CompletedProcess(command, 0))

    assert not results[0].report_written
    assert not scans_succeeded(results)


def test_dependency_findings_do_not_fail_reporting(tmp_path: Path) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        """
[build-system]
requires = ["build-tool==1.0.0"]

[project]
requires-python = ">=3.11,<3.15"
dependencies = ["runtime==2.0.0"]

[project.optional-dependencies]
extra = ["inactive==3.0.0"]

[dependency-groups]
dev = ["future-only==4.0.0"]
""".strip()
    )
    responses: dict[tuple[str, str], dict[str, Any]] = {
        ("build-tool", "1.0.0"): {"info": {"classifiers": [], "requires_python": ">=3.11", "yanked": True}},
        ("runtime", "2.0.0"): {"info": {"classifiers": [], "requires_python": ">=3.11", "yanked": False}},
        (
            "inactive",
            "3.0.0",
        ): {
            "info": {
                "classifiers": ["Development Status :: 7 - Inactive"],
                "requires_python": ">=3.11",
                "yanked": False,
            }
        },
        ("future-only", "4.0.0"): {"info": {"classifiers": [], "requires_python": ">=3.15", "yanked": False}},
    }

    report = build_report(
        pyproject_path,
        python_versions=("3.11", "3.14"),
        fetch_project=lambda name, version: responses[(name, version)],
    )

    assert report["status"] == "findings"
    assert {finding["kind"] for finding in report["findings"]} == {
        "inactive",
        "requires-python",
        "yanked",
    }


def test_dependency_reporting_raises_on_network_or_parse_errors(tmp_path: Path) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        """
[project]
requires-python = ">=3.11,<3.15"
dependencies = ["runtime==2.0.0"]
""".strip()
    )

    def failing_fetcher(name: str, version: str) -> dict[str, Any]:
        raise OSError(f"offline while fetching {name} {version}")

    with pytest.raises(DependencyReportError, match="offline"):
        build_report(pyproject_path, python_versions=("3.14",), fetch_project=failing_fetcher)
