"""Validate that wheel console-script targets are present in the artifact."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
from pathlib import Path
from zipfile import ZipFile


ENTRY_POINTS_SUFFIX = ".dist-info/entry_points.txt"


def validate_console_script_targets(wheel_path: Path) -> None:
    """Raise when a wheel omits a module targeted by a console script."""
    with ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())
        entry_point_files = [path for path in wheel_files if path.endswith(ENTRY_POINTS_SUFFIX)]
        if len(entry_point_files) != 1:
            raise ValueError(f"Expected one entry_points.txt in {wheel_path}, found {len(entry_point_files)}")

        parser = ConfigParser(interpolation=None)
        parser.read_string(wheel.read(entry_point_files[0]).decode())

        missing_modules = []
        for target in parser["console_scripts"].values():
            module_name = target.partition(":")[0].strip()
            module_path = module_name.replace(".", "/")
            module_candidates = {f"{module_path}.py", f"{module_path}/__init__.py"}
            if module_candidates.isdisjoint(wheel_files):
                missing_modules.append(module_name)

    if missing_modules:
        formatted_modules = ", ".join(sorted(missing_modules))
        raise ValueError(f"Console-script target modules are missing from {wheel_path}: {formatted_modules}")


def main() -> int:
    """Validate one or more wheel paths supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()

    for wheel_path in args.wheels:
        validate_console_script_targets(wheel_path)
        print(f"Validated console-script targets in {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
