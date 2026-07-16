"""Tests for installed-distribution metadata and contents."""

import shlex
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.validate_wheel import validate_console_script_targets


PROJECT_ROOT = Path(__file__).parents[1]
BENCHMARK_DEPENDENCIES = {"matplotlib==3.11.0", "tqdm==4.68.4"}
EXPLAINABILITY_DEPENDENCIES = {"captum==0.9.0", "matplotlib==3.11.0", "pillow==12.3.0"}
BENCHMARK_PACKAGES = {"vit", "benchmark"}
ENTRY_POINTS_PATH = "vit-0.1.1.dist-info/entry_points.txt"
OPTIONAL_DEPENDENCY_BLOCKER = """
import importlib
import importlib.abc
import importlib.util
import sys


class OptionalDependencyBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] in {"captum", "matplotlib", "PIL", "tqdm"}:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named {module.__name__!r}", name=module.__name__)


sys.meta_path.insert(0, OptionalDependencyBlocker())
entrypoint = importlib.import_module(sys.argv[1])
sys.argv = [sys.argv[1], "--help"]
entrypoint.main()
"""


def test_benchmark_distribution_configuration() -> None:
    """The wheel must ship benchmark modules and expose their dependencies as an extra."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    optional_dependencies = set(pyproject["project"]["optional-dependencies"]["benchmarking"])
    wheel_packages = set(pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])

    assert optional_dependencies == BENCHMARK_DEPENDENCIES
    assert BENCHMARK_PACKAGES <= wheel_packages


def test_explainability_distribution_configuration() -> None:
    """The explainability extra and artifact CLI must be present in package metadata."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    optional_dependencies = set(pyproject["project"]["optional-dependencies"]["explainability"])
    scripts = pyproject["project"]["scripts"]

    assert optional_dependencies == EXPLAINABILITY_DEPENDENCIES
    assert scripts["vit-explain"] == "vit.explain.cli:main"


def test_readme_git_install_commands_are_single_requirements() -> None:
    """Each direct-reference requirement must be one shell argument to pip."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    install_commands = [line for line in readme.splitlines() if line.startswith("pip install")]

    assert install_commands
    for command in install_commands:
        arguments = shlex.split(command)
        assert len(arguments) == 3
        assert arguments[:2] == ["pip", "install"]
        assert " @ git+https://github.com/TidalPaladin/vit.git" in arguments[2]


@pytest.mark.parametrize("entrypoint", ["benchmark.cli", "benchmark.component_cli", "vit.explain.cli"])
def test_benchmark_entrypoint_help_without_optional_dependencies(entrypoint: str) -> None:
    """Always-installed benchmark commands must provide help without the benchmarking extra."""
    result = subprocess.run(
        [sys.executable, "-c", OPTIONAL_DEPENDENCY_BLOCKER, entrypoint],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_wheel_validation_rejects_missing_console_script_module(tmp_path: Path) -> None:
    """Artifact validation must detect a console script whose module was not shipped."""
    wheel_path = tmp_path / "vit-0.1.1-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(ENTRY_POINTS_PATH, "[console_scripts]\nvit-benchmark = benchmark.cli:main\n")

    with pytest.raises(ValueError, match="benchmark.cli"):
        validate_console_script_targets(wheel_path)


def test_wheel_validation_accepts_packaged_console_script_module(tmp_path: Path) -> None:
    """A package module satisfies the console-script artifact contract."""
    wheel_path = tmp_path / "vit-0.1.1-py3-none-any.whl"
    with ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(ENTRY_POINTS_PATH, "[console_scripts]\nvit-benchmark = benchmark.cli:main\n")
        wheel.writestr("benchmark/__init__.py", "")
        wheel.writestr("benchmark/cli.py", "def main(): ...\n")

    validate_console_script_targets(wheel_path)
