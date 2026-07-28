"""Report yanked, inactive, and Python-incompatible direct dependency pins."""

from __future__ import annotations

import argparse
import json
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version


FetchProject = Callable[[str, str], Mapping[str, Any]]
PYPI_JSON_BASE = "https://pypi.org/pypi"
INACTIVE_CLASSIFIER = "Development Status :: 7 - Inactive"


class DependencyReportError(RuntimeError):
    """Raised when dependency metadata cannot be collected or interpreted."""


@dataclass(frozen=True)
class PinnedDependency:
    """An exact dependency pin and the pyproject section that declared it."""

    name: str
    version: str
    requirement: str
    source: str


def _requirement_strings(values: object, source: str) -> Iterable[tuple[str, str]]:
    if not isinstance(values, list):
        raise DependencyReportError(f"{source} must be an array")
    for value in values:
        if isinstance(value, str):
            yield value, source
        elif not (isinstance(value, dict) and set(value) == {"include-group"}):
            raise DependencyReportError(f"Unsupported dependency entry in {source}: {value!r}")


def _declared_requirements(pyproject: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    build_system = pyproject.get("build-system", {})
    project = pyproject.get("project", {})
    yield from _requirement_strings(build_system.get("requires", []), "build-system.requires")
    yield from _requirement_strings(project.get("dependencies", []), "project.dependencies")

    for group_name, requirements in project.get("optional-dependencies", {}).items():
        yield from _requirement_strings(requirements, f"project.optional-dependencies.{group_name}")
    for group_name, requirements in pyproject.get("dependency-groups", {}).items():
        yield from _requirement_strings(requirements, f"dependency-groups.{group_name}")


def _exact_version(requirement: Requirement) -> str | None:
    specifiers = list(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==" or "*" in specifiers[0].version:
        return None
    return specifiers[0].version


def _collect_pins(pyproject: Mapping[str, Any]) -> tuple[PinnedDependency, ...]:
    pins: dict[tuple[str, str, str], PinnedDependency] = {}
    for requirement_text, source in _declared_requirements(pyproject):
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement as error:
            raise DependencyReportError(f"Invalid requirement in {source}: {requirement_text}") from error
        version = _exact_version(requirement)
        if version is None:
            continue
        key = (requirement.name.lower(), version, source)
        pins[key] = PinnedDependency(
            name=requirement.name,
            version=version,
            requirement=requirement_text,
            source=source,
        )
    return tuple(sorted(pins.values(), key=lambda pin: (pin.name.lower(), pin.version, pin.source)))


def fetch_pypi_project(name: str, version: str) -> Mapping[str, Any]:
    """Fetch PyPI metadata for one exact project version."""
    encoded_name = urllib.parse.quote(name, safe="")
    encoded_version = urllib.parse.quote(version, safe="")
    request = urllib.request.Request(
        f"{PYPI_JSON_BASE}/{encoded_name}/{encoded_version}/json",
        headers={"Accept": "application/json", "User-Agent": "vit-dependency-report/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise DependencyReportError(f"Unable to fetch PyPI metadata for {name} {version}: {error}") from error
    if not isinstance(payload, dict):
        raise DependencyReportError(f"PyPI returned an invalid payload for {name} {version}")
    return payload


def _marker_applies(requirement_text: str, python_version: str) -> bool:
    requirement = Requirement(requirement_text)
    if requirement.marker is None:
        return True
    environment: dict[str, str] = {str(key): str(value) for key, value in default_environment().items()}
    environment["python_version"] = python_version
    environment["python_full_version"] = f"{python_version}.0"
    return requirement.marker.evaluate(environment)


def _requires_python_supports(specifier: str | None, python_version: str) -> bool:
    if not specifier:
        return True
    try:
        return Version(f"{python_version}.0") in SpecifierSet(specifier)
    except InvalidSpecifier as error:
        raise DependencyReportError(f"Invalid requires-python metadata {specifier!r}") from error


def build_report(
    pyproject_path: Path,
    *,
    python_versions: Sequence[str],
    fetch_project: FetchProject = fetch_pypi_project,
) -> dict[str, Any]:
    """Build a report; findings are data, while collection errors raise."""
    try:
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DependencyReportError(f"Unable to parse {pyproject_path}: {error}") from error

    pins = _collect_pins(pyproject)
    findings: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    metadata_cache: dict[tuple[str, str], Mapping[str, Any]] = {}

    for pin in pins:
        cache_key = (pin.name.lower(), pin.version)
        if cache_key not in metadata_cache:
            try:
                metadata_cache[cache_key] = fetch_project(pin.name, pin.version)
            except DependencyReportError:
                raise
            except Exception as error:
                raise DependencyReportError(
                    f"Unable to fetch metadata for {pin.name} {pin.version}: {error}"
                ) from error

        payload = metadata_cache[cache_key]
        info = payload.get("info")
        if not isinstance(info, Mapping):
            raise DependencyReportError(f"PyPI metadata for {pin.name} {pin.version} has no info mapping")

        classifiers = info.get("classifiers") or []
        if not isinstance(classifiers, list) or not all(isinstance(value, str) for value in classifiers):
            raise DependencyReportError(f"PyPI classifiers for {pin.name} {pin.version} are invalid")
        requires_python_value = info.get("requires_python")
        requires_python = requires_python_value if isinstance(requires_python_value, str) else None
        yanked = bool(info.get("yanked", False))

        package = asdict(pin) | {
            "requires_python": requires_python,
            "yanked": yanked,
            "inactive": INACTIVE_CLASSIFIER in classifiers,
        }
        packages.append(package)
        if yanked:
            findings.append({"kind": "yanked", "dependency": pin.name, "version": pin.version, "source": pin.source})
        if INACTIVE_CLASSIFIER in classifiers:
            findings.append({"kind": "inactive", "dependency": pin.name, "version": pin.version, "source": pin.source})

        for python_version in python_versions:
            if _marker_applies(pin.requirement, python_version) and not _requires_python_supports(
                requires_python, python_version
            ):
                findings.append(
                    {
                        "kind": "requires-python",
                        "dependency": pin.name,
                        "version": pin.version,
                        "source": pin.source,
                        "python_version": python_version,
                        "requires_python": requires_python,
                    }
                )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pyproject": str(pyproject_path),
        "python_versions": list(python_versions),
        "pypi_service": PYPI_JSON_BASE,
        "status": "findings" if findings else "ok",
        "packages": packages,
        "findings": findings,
    }


def _markdown_summary(report: Mapping[str, Any]) -> str:
    findings = report["findings"]
    lines = [
        "## Dependency deprecation report",
        "",
        f"Checked {len(report['packages'])} exact direct pins for Python {', '.join(report['python_versions'])}.",
        "",
    ]
    if not findings:
        lines.append("No yanked releases, inactive classifiers, or `requires-python` conflicts were found.")
    else:
        lines.extend(("| Finding | Dependency | Version | Source | Detail |", "|---|---|---|---|---|"))
        for finding in findings:
            detail = ""
            if finding["kind"] == "requires-python":
                detail = f"Python {finding['python_version']} not in `{finding['requires_python'] or 'unspecified'}`"
            lines.append(
                f"| {finding['kind']} | {finding['dependency']} | {finding['version']} | "
                f"{finding['source']} | {detail} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write JSON and Markdown dependency-health reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--python-version", action="append", dest="python_versions")
    args = parser.parse_args()
    python_versions = tuple(args.python_versions or ("3.11", "3.14"))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = build_report(args.pyproject, python_versions=python_versions)
    except DependencyReportError as error:
        error_report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "error",
            "error": str(error),
        }
        args.json_output.write_text(json.dumps(error_report, indent=2, sort_keys=True) + "\n")
        args.summary_output.write_text(f"## Dependency deprecation report\n\nCollection failed: {error}\n")
        return 2

    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.summary_output.write_text(_markdown_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
