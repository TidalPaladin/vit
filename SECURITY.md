# Security

## Dependency audits

Run the public-vulnerability audit against both supported Python bounds:

```console
make audit-dependencies PYTHON_VERSION=3.11
make audit-dependencies PYTHON_VERSION=3.14
```

The target exports every locked dependency group from `uv.lock`, omits the local project, and passes the pinned
requirements to `pip-audit==2.10.1`. A known public vulnerability makes the command fail. The independent
`dependency-audit.yml` workflow runs both Python bounds every Monday at 06:17 UTC and on manual request.

The same workflow runs `zizmor==1.28.0` against every GitHub Actions workflow and records scanner versions, commands,
the UTC query time, the `uv.lock` SHA-256 digest, the supported Python bounds, and the PyPI advisory service. It also
checks direct, optional, build, and dependency-group pins for PyPI yanks, inactive classifiers, and `requires-python`
conflicts. Those lifecycle findings are informational; network, parsing, and command errors fail the job. Reports are
stored as GitHub Actions artifacts for seven days.

Run the additional checks locally with:

```console
make audit-workflows
make report-deprecations REPORT_DIR=/tmp/vit-dependency-report
make test-deprecations
```

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
| CircleCI images and the Codecov orb | Retained only for the CI migration overlap and not covered by `pip-audit`. |
| GitHub Actions | Every action is pinned to a full commit SHA. Locked `zizmor` checks workflow structure and permissions weekly. |
| CUDA execution | Deferred. Required CI is Linux CPU-only until a suitable self-hosted CUDA runner is available. |
