# Security

## Dependency audits

Run the public-vulnerability audit against both supported Python bounds:

```console
make audit-dependencies PYTHON_VERSION=3.11
make audit-dependencies PYTHON_VERSION=3.14
```

The target exports every locked dependency group from `uv.lock`, omits the local project, and passes the pinned
requirements to `pip-audit`. A known public vulnerability makes the command fail. GitHub Actions runs the same checks
when dependency inputs change, on updates to `master`, on manual request, and every Monday.

No vulnerability IDs are ignored. Any future `--ignore-vuln` entry must identify the affected package, explain why the
advisory does not apply, link to supporting evidence, and include a review date in this file.

## Coverage

| Surface | Coverage decision |
| --- | --- |
| Direct and transitive Python packages | Audited from all locked groups for Python 3.11 and 3.14 on Linux. |
| Public vulnerability data | `pip-audit==2.10.1` queries PyPI's public vulnerability service. |
| Project source | Not covered by `pip-audit`; formatting, linting, typing, and tests remain separate quality gates. |
| Native libraries bundled inside wheels | Not covered unless the vulnerability is reported against the Python package. Review upstream package advisories during dependency updates. |
| Non-Linux dependency variants | Not covered by the scheduled audit because the supported CI environment is Linux. Review platform-specific changes when adding another CI platform. |
| CircleCI images and the Codecov orb | Not covered by `pip-audit`. Review their release and security notices when changing the versioned CI references. |
| GitHub Actions | `actions/checkout` is pinned to a full commit SHA. Review the upstream release before changing it. |
