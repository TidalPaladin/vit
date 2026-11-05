# Dependency maintenance: 3 November 2026

Resolves [issue #127](https://github.com/TidalPaladin/vit/issues/127) by updating the audit-tool dependency `urllib3` from 2.7.0 to 2.8.0. A broad lock refresh also removes the other findings in the latest weekly audit. The supported Python range expands to 3.11 through 3.15 (`>=3.11,<3.16`). Public library interfaces stay the same.

## Recent audit evidence

The [2 November Dependency health run](https://github.com/TidalPaladin/vit/actions/runs/41479226915) failed its security job. Its deprecation report passed. The 5, 12, 19, and 26 October security runs also failed. These remote results apply to commit `1d86bcef3a8a17d97f81a3685583a6310ce87769`, not the current local checkout.

Before the change, the repository security runner reported 55 distinct advisory IDs across seven packages in both labeled boundary reports. The workflow scanner passed. Inspection showed that the old runner executed both package scans under the newest Python, so its Python 3.11 report selected NumPy 2.5.1 instead of 2.4.6. The test is a dependency scan against the exact lockfile; no application-code change or upstream exploit fixture is needed.

The audit target now runs pip-audit under `PYTHON_VERSION`, which also controls the dependency export. A regression first failed with scanner Python 3.15 when 3.11 was requested, then passed after the fix. This preserves each boundary's environment markers.

## Applied direct updates

Current release metadata was checked with the [PyPI JSON API](https://docs.pypi.org/api/json/). Every direct dependency remains pinned. Both optional extras and matching development groups use the same pins.

| Component | Before | After | Classification |
| --- | --- | --- | --- |
| basedpyright | 1.39.9 | 1.40.2 | Compatible manifest pin |
| coverage | 7.15.0 | 7.17.1 | Compatible manifest pin |
| hatchling | 1.31.0 | 1.33.0 | Compatible manifest pin |
| matplotlib | 3.11.0 | 3.11.2 | Compatible manifest pin |
| numpy | 2.5.1 | 2.5.4 | Compatible manifest pin |
| packaging | 26.2 | 26.3 | Compatible manifest pin |
| pillow | 12.3.0 | 13.0.0 | Compatible manifest pin |
| pytest-env | 1.6.0 | 1.8.4 | Compatible manifest pin |
| pytest-mock | 3.15.1 | 3.17.0 | Compatible manifest pin |
| pytest | 9.1.1 | 9.2.0 | Compatible manifest pin |
| ruff | 0.15.21 | 0.17.3 | Compatible manifest pin |
| torch | 2.13.0 | 2.15.0 | Compatible manifest pin |
| torchao | 0.17.0 | 0.18.0 | Compatible manifest pin |
| tqdm | 4.68.4 | 4.70.1 | Compatible manifest pin |
| zizmor | 1.28.0 | 1.32.0 | Compatible manifest pin |

NumPy 2.4.6 remains the latest 2.4 release for Python 3.11. NumPy 2.5.4 requires Python 3.12 or newer. Captum 0.9.0, PyYAML 6.0.3, hatch-vcs 0.5.0, pytest-cov 7.1.0, pdbpp 0.12.1, and pip-audit 2.10.1 were already current.

[PyTorch 2.15](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) changes the default Linux wheels to CUDA 13.2. [TorchAO 0.18](https://github.com/pytorch/ao/releases/tag/v0.18.0) removes its deprecated v1 tensor layouts; the repository uses v2 quantization configurations. [Pillow 13](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html) includes the reported security fixes. Its removal of Python 3.10 support does not affect the repository.

The lock refresh updates 57 existing package names, including CUDA components, Triton 3.9.0, and the vulnerable transitive packages below. The new python-dateutil 4.0.0 is compatible with the supported Python range; its [release notes](https://dateutil.readthedocs.io/en/stable/changelog.html) describe stricter parsing and a new tzdata dependency.

## Advisory results

The findings below include runtime dependencies (`filelock`, `setuptools`), optional image support (`pillow`), and security-tool dependencies (`certifi`, `msgpack`, `pip`, `urllib3`). Each row records a distinct ID from the before scan. The after scans confirm that none remain.

| Advisory | Package | Current before change | Updated | Status | Breaking change required | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| PYSEC-2026-4320 | certifi | 2026.7.22 | 2026.10.23 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-6q48-6c7w-9gw2); both boundary scans |
| CVE-2026-112940 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-v4x8-3wr5-3wq8); both boundary scans |
| CVE-2026-112913 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-9vhv-jxww-xfm5); both boundary scans |
| CVE-2026-112907 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-567q-5qvw-pf92); both boundary scans |
| CVE-2026-112936 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-j8f7-rjxc-mr56); both boundary scans |
| CVE-2026-112910 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-8mx5-7m5m-r77g); both boundary scans |
| CVE-2026-112909 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-69hj-f68j-2879); both boundary scans |
| CVE-2026-112939 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-rfpc-qcw5-3pc4); both boundary scans |
| CVE-2026-112915 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-cwqr-hh33-gh44); both boundary scans |
| CVE-2026-112935 | filelock | 3.29.7 | 4.2.7 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-fq56-3cx3-fjxp); both boundary scans |
| CVE-2026-114613 | msgpack | 1.2.1 | 1.2.4 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-j586-36cw-2gc2); both boundary scans |
| PYSEC-2026-4327 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-4558-3h8r-v8xp); both boundary scans |
| PYSEC-2026-4328 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-4rr5-89j8-mwwm); both boundary scans |
| PYSEC-2026-4326 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-27h8-gf7j-v5hc); both boundary scans |
| PYSEC-2026-4339 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-v9pv-5gc8-w2m6); both boundary scans |
| PYSEC-2026-4340 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-4rm8-3hmm-39fg); both boundary scans |
| PYSEC-2026-4343 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-9v9v-9qvf-h2hj); both boundary scans |
| PYSEC-2026-4334 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-pchr-8j38-3x4p); both boundary scans |
| PYSEC-2026-4337 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-p8hx-v2p6-rp84); both boundary scans |
| PYSEC-2026-4341 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-3j72-p6qg-6xp7); both boundary scans |
| PYSEC-2026-4330 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-5x8w-8jj5-q52h); both boundary scans |
| PYSEC-2026-4338 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-g842-9rc4-69pw); both boundary scans |
| PYSEC-2026-4335 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-xgh8-jp6q-p5x8); both boundary scans |
| PYSEC-2026-4336 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-654x-cwwv-5j9c); both boundary scans |
| PYSEC-2026-4331 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-7mwm-ghpf-9vrm); both boundary scans |
| PYSEC-2026-4332 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-8g5g-8vmr-9782); both boundary scans |
| PYSEC-2026-4329 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-58r4-6rv3-gfh8); both boundary scans |
| PYSEC-2026-4333 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-cp9v-662q-vvx9); both boundary scans |
| PYSEC-2026-4342 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-g4qv-2jxx-769h); both boundary scans |
| CVE-2026-112895 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-p4wg-p73j-c528); both boundary scans |
| CVE-2026-112400 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-65qm-jvmh-6pqg); both boundary scans |
| CVE-2026-112897 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-q45j-58j2-45rg); both boundary scans |
| CVE-2026-112395 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-462x-pgg6-55fp); both boundary scans |
| CVE-2026-112404 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-c6hw-5grf-ff96); both boundary scans |
| CVE-2026-112899 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-vx6w-vv9f-6gvg); both boundary scans |
| CVE-2026-112900 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-w889-685w-c8fh); both boundary scans |
| CVE-2026-112893 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-j4m9-6w36-pw47); both boundary scans |
| CVE-2026-112891 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-fc9g-5qg7-mf45); both boundary scans |
| CVE-2026-112390 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-22hg-r758-rrw9); both boundary scans |
| CVE-2026-112401 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-74r2-7ff9-mjj2); both boundary scans |
| CVE-2026-112391 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-254f-3936-288h); both boundary scans |
| CVE-2026-112894 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-jg86-v3jv-hrpm); both boundary scans |
| CVE-2026-112406 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-f2g6-542v-9j58); both boundary scans |
| CVE-2026-112399 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-5xm2-6676-qcp9); both boundary scans |
| CVE-2026-112892 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-gg2r-wvfm-q9rx); both boundary scans |
| CVE-2026-112393 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-332p-7hrc-92wf); both boundary scans |
| CVE-2026-96638 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-24rc-8hj2-v98c); both boundary scans |
| CVE-2026-96640 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-mrfp-p49p-3hf2); both boundary scans |
| CVE-2026-96778 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-g7cr-3p24-7wf7); both boundary scans |
| GHSA-78g4-xqwv-q3g8 | pillow | 12.3.0 | 13.0.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-78g4-xqwv-q3g8); both boundary scans |
| PYSEC-2026-3721 | pip | 26.1.2 | 26.3 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-qwm4-qh6w-59xr); both boundary scans |
| GHSA-xmj3-9gf7-h52w | setuptools | 83.0.0 | 84.2.1 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-xmj3-9gf7-h52w); both boundary scans |
| PYSEC-2026-4177 | urllib3 | 2.7.0 | 2.8.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-vxq7-64xx-v4gw); both boundary scans |
| PYSEC-2026-4176 | urllib3 | 2.7.0 | 2.8.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g); both boundary scans |
| PYSEC-2026-4175 | urllib3 | 2.7.0 | 2.8.0 | fixed | No public API change | [Advisory](https://github.com/advisories/GHSA-8988-9cw3-xx77); both boundary scans |

## Toolchain review

uv is updated from 0.11.28 to 0.13.7 because the old release only offered a Python 3.15 beta. The new toolchain installs stable Python 3.15.2. setup-uv is updated from v9.0.0 to v10.3.3 and upload-artifact from v7.0.1 to v7.0.2, with exact upstream commit SHA pins. The uv download checksum comes from its official x86-64 Linux release artifact. checkout v7.0.1 is current. Runtime assertions and cache keys match PyTorch 2.15.0 and TorchAO 0.18.0.

CI retains Python 3.11 and 3.14 and adds 3.15. Production compilation, typing, and audit-tool execution now use 3.15. Security and dependency reports check the 3.11 and 3.15 boundaries. The CI matrix contract failed before the change because 3.15 was absent, then passed after the update.

## Validation

All checks below passed on Linux. CUDA tests used an RTX 3090 with driver 580.82.07 and PyTorch 2.15.0+cu132.

| Check | Python | Result |
| --- | --- | --- |
| `make style quality types check-distribution` | 3.15.2 | Passed; no lint or type errors |
| `make test-ci` | 3.11.14 | 691 passed, 3 expected skips |
| `make test-ci` | 3.14.6 | 691 passed, 3 expected skips |
| `make test-ci` | 3.15.2 | 692 passed, 3 expected skips, including the added audit regression |
| `make test-compile-cpu` | 3.14.6 and 3.15.2 | 9 passed on each version |
| CUDA tests excluding compilation | 3.14.6 and 3.15.2 | 333 passed, 3 expected skips on each version |
| `make test-compile-cuda` | 3.14.6 and 3.15.2 | 9 passed on each version |
| Audit/workflow/report regressions | 3.11.14 and 3.15.2 | 19 passed on each version |
| `make report-deprecations` | 3.11/3.15 markers, runner 3.15.2 | 28 exact direct pins checked; no yanks, inactive classifiers, or Python conflicts |
| Source/wheel build, rebuild from sdist, console scripts, and wheel metadata | 3.15.2 | Passed; both wheels declare `>=3.11,<3.16` and the new runtime pins |

The complete test set on Python 3.15 passed in four selections: 1,043 passed and six expected skips. The skips cover CPU cross-attention quantization, four unsupported 3D RoPE cases across CPU/CUDA, and the optional FlashAttention package. AOTInductor packaging, packed dynamic shapes, checkpointing, and specialized attention backward paths passed.

The original CPU runs found two stale packaging-test version expectations. Those expectations were updated and the Python 3.14 suite then passed. The final Python 3.15 suite includes both that change and the audit-marker regression. The Python 3.11 audit regression suite was rerun after the marker fix.

Security command: `uvx --from uv==0.13.7 uv run --isolated --frozen --only-group ci-security --python 3.15 python tools/run_security_audit.py --report-dir /tmp/vit-127-final-audit`.

- Tools: uv 0.13.7, pip-audit 2.10.1, zizmor 1.32.0.
- Source: the PyPI advisory database, checked on 3 November 2026.
- The package scans for Python 3.11 and 3.15 and the strict offline workflow scan each exited 0 and wrote their report. No packages were skipped.
- Audited lockfile SHA-256: `5b0b8afaf79e4635df94174480cfad534bddd8b417a81ba943daf73e287c814d`.
- The separate build-backend scan resolved the pinned build requirements and their transitive dependencies with `uv pip compile --python-version 3.14 --generate-hashes`, then used the same strict pip-audit options. All ten packages passed, including hatchling 1.33.0 and hatch-vcs 0.5.0.

The current TorchAO release still logs an enum-registration deprecation message with PyTorch 2.15. Triton 3.9 warns that its default FP fusion setting will change in 3.10. These upstream messages do not fail the checks; no suppression was added.

## Audit scope

Inspected `pyproject.toml`, `uv.lock`, `Makefile`, `.github/workflows/`, and the repository audit tools. No container manifests, system-package installation steps, or submodules are present. The security scans include runtime, optional, development, and audit dependencies, with no skipped packages. The final Python 3.11 scan checked 84 packages and selected NumPy 2.4.6, contourpy 1.3.3, and networkx 3.6.1. The Python 3.15 scan checked 81 packages and selected NumPy 2.5.4, contourpy 1.4.0, and networkx 3.7. The separate build scan checked ten packages. OS packages, the host NVIDIA driver, and upstream action internals are outside the Python package scan.

The initial validation used local workflow commands. Remote validation of the published branch is recorded in the pull request. Historical performance results in README remain tied to their measured PyTorch 2.13.0 environment; these updates do not establish a new performance baseline.

## Publication validation: 5 November 2026

The publication branch starts at master commit `1d86bcef3a8a17d97f81a3685583a6310ce87769`. It retains the upstream dynamic-shape regressions and the direct pip security pin. That pin advances from 26.2.1 to the already locked and tested pip 26.3. No other package version changed when the work was applied to the current base.

Fresh security scans under Python 3.11 and 3.15 and the strict workflow scan passed on 5 November. The lockfile SHA-256 is `cce1306f29761cd46ee1cfd2b79e95a64651fb61c7dfa6bf6b77dace6ea8902d`. The package counts remain 84 and 81, with NumPy 2.4.6 and 2.5.4 selected respectively. The dependency report now checks 29 exact direct pins and reports no yanks, inactive classifiers, or Python conflicts.

The first full local gate encountered shared compiler-cache binaries that required an unavailable GLIBC_2.43 symbol. AOT tests passed after recompilation. Publication validation uses a fresh task-specific `TORCHINDUCTOR_CACHE_DIR` and preserves the shared cache.

The fresh-cache `make check` gate passed on Python 3.15.2: 707 tests passed and 348 skipped. Of the skips, 345 require CUDA; the remaining three are the documented CPU quantization and 3D RoPE cases. `make test-ci` also passed: 692 tests passed and three skipped. Formatting, linting, types, wheel validation, and the stronger upstream CPU compilation regressions passed.

CUDA could not initialize on 5 November. NVML reports a driver/library version mismatch with library version 580.178, and PyTorch reports CUDA error 804. The CUDA results from 3 November above are historical; fresh CUDA correctness and compilation checks were unavailable for publication.
