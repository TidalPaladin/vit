"""Run every locked security scanner and write a single execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


Runner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]
PYTHON_VERSIONS = ("3.11", "3.14")
VULNERABILITY_SERVICE = "PyPI advisory database"


@dataclass(frozen=True)
class Scan:
    """One scanner invocation and its expected machine-readable report."""

    name: str
    command: tuple[str, ...]
    report_path: Path


@dataclass(frozen=True)
class ScanResult:
    """Recorded result for one scanner invocation."""

    name: str
    command: tuple[str, ...]
    exit_code: int
    report_path: str
    report_written: bool


def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True)


def run_scans(scans: Sequence[Scan], *, runner: Runner = _default_runner) -> tuple[ScanResult, ...]:
    """Run all scanners even when an earlier scanner reports a finding."""
    results = []
    for scan in scans:
        completed = runner(scan.command)
        results.append(
            ScanResult(
                name=scan.name,
                command=scan.command,
                exit_code=completed.returncode,
                report_path=str(scan.report_path),
                report_written=scan.report_path.is_file(),
            )
        )
    return tuple(results)


def scans_succeeded(results: Sequence[ScanResult]) -> bool:
    """Return whether every scanner completed cleanly and wrote its report."""
    return all(result.exit_code == 0 and result.report_written for result in results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_version(command: tuple[str, ...]) -> dict[str, str | int]:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    output = (completed.stdout or completed.stderr).strip()
    return {"command": " ".join(command), "exit_code": completed.returncode, "output": output}


def _project_metadata(pyproject_path: Path) -> tuple[str, list[str]]:
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    dependency_groups = pyproject.get("dependency-groups", {})
    return str(pyproject["project"]["requires-python"]), ["project", *sorted(dependency_groups)]


def _build_scans(report_directory: Path) -> tuple[Scan, ...]:
    scans = [
        Scan(
            name=f"pip-audit-python-{python_version}",
            command=(
                "make",
                "audit-dependencies",
                f"PYTHON_VERSION={python_version}",
                f"AUDIT_REPORT={report_directory / f'pip-audit-python-{python_version}.json'}",
            ),
            report_path=report_directory / f"pip-audit-python-{python_version}.json",
        )
        for python_version in PYTHON_VERSIONS
    ]
    scans.append(
        Scan(
            name="zizmor-workflows",
            command=("make", "audit-workflows", f"ZIZMOR_REPORT={report_directory / 'zizmor-workflows.json'}"),
            report_path=report_directory / "zizmor-workflows.json",
        )
    )
    return tuple(scans)


def main() -> int:
    """Run the security audit and write its reproducibility manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    report_directory = args.report_dir.resolve()
    report_directory.mkdir(parents=True, exist_ok=True)
    scans = _build_scans(report_directory)
    python_bounds, included_dependency_groups = _project_metadata(repository_root / "pyproject.toml")

    results = run_scans(scans)
    security_tool_prefix = (
        "uv",
        "run",
        "--isolated",
        "--frozen",
        "--only-group",
        "ci-security",
        "--python",
        "3.14",
    )
    tools = {
        "uv": _tool_version(("uv", "--version")),
        "pip-audit": _tool_version((*security_tool_prefix, "pip-audit", "--version")),
        "zizmor": _tool_version((*security_tool_prefix, "zizmor", "--version")),
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository_root": str(repository_root),
        "uv_lock_sha256": _sha256(repository_root / "uv.lock"),
        "python_bounds": python_bounds,
        "python_versions": list(PYTHON_VERSIONS),
        "included_dependency_groups": included_dependency_groups,
        "vulnerability_service": VULNERABILITY_SERVICE,
        "tools": tools,
        "scans": [asdict(result) for result in results],
    }
    (report_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tools_succeeded = all(tool["exit_code"] == 0 for tool in tools.values())
    return 0 if scans_succeeded(results) and tools_succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
