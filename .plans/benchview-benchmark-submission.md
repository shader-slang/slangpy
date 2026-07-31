# Submit SlangPy benchmark fixtures through the BenchView API

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, SlangPy's pytest benchmark fixtures still measure and display results locally, but an enabled benchmark run sends native BenchView version-one JSON to the configured BenchView HTTP service instead of importing `pymongo` and writing a legacy report directly to MongoDB. Tests whose original pytest function name contains `_cpu` identify their metric as `cpu_time`; every other test uses `gpu_time`, matching the existing imported history. Both use milliseconds. Pytest parameters become legacy-compatible string case dimensions, source file plus test function remains the stable test identity, and machine/VCS facts use BenchView's dedicated environment and run fields.

A developer can demonstrate the result without a GPU by running focused Python tests that build representative submissions and intercept the HTTP request. In production, `python tools/ci.py benchmark-python` receives a BenchView base URL, reads the write key only from the `BENCHVIEW_API_KEY` environment variable, runs each device shard, and submits batches to `<base>/api/v1/submissions` with a Bearer authorization header.

## Progress

- [x] (2026-07-16 13:47Z) Read the repository benchmark instructions, current fixture/report/plugin lifecycle, CI wrapper, and BenchView's frozen version-one contract and existing SlangPy legacy mapping.
- [x] (2026-07-16 13:47Z) Chose the native observation, run identity, batching, authentication, and compatibility design recorded below.
- [x] (2026-07-16 14:05Z) Added native BenchView observation construction, bounded batching, nested-base URL handling, safe Bearer-authenticated HTTP submission, and parallel accumulation that retains local report comparison.
- [x] (2026-07-16 14:05Z) Replaced MongoDB pytest and `tools/ci.py` options with API-oriented options and environment-only key handling; removed the `pymongo` development dependency.
- [x] (2026-07-16 14:05Z) Added nine focused tests and updated benchmark documentation, including compatibility corrections for legacy string dimensions and `_cpu` metric selection.
- [x] (2026-07-16 14:13Z) Completed the Windows debug build, nine focused tests, Pyright with zero findings, the full pre-commit suite, `git diff --check`, CI help inspection, and a generated-payload injection through BenchView's real in-memory API with HTTP 201.
- [x] (2026-07-16 14:35Z) Moved the native BenchView protocol, batching, and HTTP producer into `slangpy/testing/benchmark/benchview.py`, leaving `report.py` responsible only for legacy local reports; revalidated nine focused tests, targeted Pyright, pre-commit, and whitespace checks.

## Surprises and Discoveries

- Observation: the existing `run_id` is supplied to each device-specific pytest process and the plugin uses it only as a legacy report field and MongoDB document value.
  Evidence: `tools/ci.py` loops over device types and invokes `--benchmark-upload <run_id>` for every process, while `slangpy/testing/benchmark/plugin.py` assigns that value to `report["run_id"]` immediately before the direct insert.

- Observation: timing mechanism alone does not determine the logical metric. Several `BenchmarkPythonFunction` cases synchronize GPU work, while explicitly CPU-oriented tests consistently contain `_cpu` in the original pytest function name.
  Evidence: the existing importer used `_cpu` after product-owner clarification, and names in `test_benchmark_interop.py` and `test_benchmark_ppisp.py` distinguish synchronized GPU and CPU-overhead cases this way.

- Observation: all historical pytest parameters were stringified before legacy storage, so emitting native integers or booleans would create different BenchView observation identities from the imported history.
  Evidence: the old `ReportFixture` built `params` with `str(v)`, and the BenchView importer preserved those string types as case dimensions.

- Observation: the existing Windows CMake cache could not regenerate until Visual Studio's Ninja and developer environment were supplied, then compilation required `/EHsc` because the regenerated cache combined standard-library exception warnings with `/WX`.
  Evidence: the first build failed with a missing `CMAKE_MAKE_PROGRAM`, the second found MSVC but failed on C4530, and the same preset completed after configuring the local cache with Visual Studio Ninja and `/EHsc`. No repository build file changed.

- Observation: a full-repository Pyright rerun after the module split reports seven unrelated profiler/UI attribute errors in `examples/pathtracer/pathtracer.py`.
  Evidence: targeted Pyright over `benchview.py`, `report.py`, the fixture, plugin, and focused test reports zero errors and zero warnings; none of the full-run findings reference benchmark code.

- Observation: saved local reports and comparison tables depend on the legacy `BenchmarkReport` shape.
  Evidence: `slangpy/testing/benchmark/table.py` reads top-level `min`, `max`, `mean`, `median`, and `stddev`, and `.benchmarks` files are loaded back into that shape.

## Decision Log

- Decision: retain `BenchmarkReport` for local save/compare behavior and accumulate a parallel list of native BenchView observations using the same sample list.
  Rationale: this makes the server path native without breaking existing local benchmark workflows or requiring a migration of `.benchmarks` files.
  Date/Author: 2026-07-16, Codex.

- Decision: isolate native BenchView payload and transport code in `slangpy/testing/benchmark/benchview.py`, while `report.py` remains the legacy local-report module.
  Rationale: the two formats serve different persistence boundaries and keeping them separate makes it harder for future local-report changes to accidentally alter the API producer.
  Date/Author: 2026-07-16, Codex.

- Decision: use project `slangpy`, suite `python`, test identity `<normalized source file>:<original pytest function>`, and the existing stringified pytest parameters as case dimensions with `DeviceType.cuda`-style values normalized to `cuda`.
  Rationale: these values and scalar types match the accepted legacy importer mapping, so new submissions continue the same graph identities.
  Date/Author: 2026-07-16, Codex.

- Decision: preserve the established `_cpu` function-name marker for metric selection instead of inferring semantics from the wrapper class.
  Rationale: Python wrappers measure both synchronized GPU work and CPU dispatch overhead; changing every Python-wrapper result to `cpu_time` would split or mislabel existing histories.
  Date/Author: 2026-07-16, Codex.

- Decision: derive the logical run key from the full Git revision, suite, and the first 16 hexadecimal characters of a deterministic digest of `projectVersion` and `slangBuildTag` run dimensions.
  Rationale: the key is independent of host, process, target, timestamp, or random state and matches the run-key policy used for already imported SlangPy data.
  Date/Author: 2026-07-16, Codex.

- Decision: treat the caller-provided run ID as `execution.requestId`, generate a fresh process execution UUID, and derive each batch idempotency key from its complete pre-key JSON payload.
  Rationale: distributed processes share the logical run but retain traceable execution identity. Reposting the same in-memory request is an exact retry, while running the benchmarks again naturally creates new observations and therefore intentional replacements.
  Date/Author: 2026-07-16, Codex.

- Decision: use `urllib.request` from the Python standard library and obtain the write key only from `BENCHVIEW_API_KEY`.
  Rationale: SlangPy gains no runtime HTTP or MongoDB dependency, and the secret does not appear in command-line arguments or logs.
  Date/Author: 2026-07-16, Codex.

- Decision: retain `--benchmark-upload` as an alias for the clearer `--benchmark-submit`, but remove the MongoDB connection and database options.
  Rationale: existing direct pytest invocations keep working after replacing their destination option, while help and new CI commands describe the HTTP behavior accurately.
  Date/Author: 2026-07-16, Codex.

## Outcomes and Retrospective

SlangPy benchmark fixtures now retain their local report/comparison behavior while also accumulating native BenchView observations. The legacy report model stays in `report.py`, while the native API producer has its own `benchview.py` module. Enabled sessions submit bounded authenticated HTTP batches below root or nested deployment bases, and direct MongoDB code, arguments, and the `pymongo` development dependency are gone. The producer preserves imported test, case, run, and metric histories rather than creating parallel graph identities. The implementation build, focused tests, pre-commit hooks, whitespace checks, CI help, and an in-memory request accepted by the real BenchView API pass; targeted typechecking of every changed benchmark file is clean, while the current full-repository Pyright run has unrelated pathtracer errors recorded above. A live benchmark run and production submission remain an operator action because they require a selected GPU target and the real write key.

## Context and Orientation

Before this change, `slangpy/testing/benchmark/fixtures.py` appended only a legacy `BenchmarkReport` to a private pytest configuration context. `slangpy/testing/benchmark/plugin.py` owned local save/compare options and inserted the completed legacy report directly into MongoDB. `slangpy/testing/benchmark/report.py` imported `pymongo` inside `upload_report`, `tools/ci.py` forwarded a credential-bearing connection string and database name, and `requirements-dev.txt` carried `pymongo` only for that path. The implemented `benchview.py` now owns native payload construction and HTTP submission; `report.py` retains only local report serialization. The fixture and plugin connect both paths, and `.github/instructions/benchmarks.instructions.md` is the updated user-facing guide.

BenchView accepts `POST /api/v1/submissions` below either a root or nested deployment base. One submission contains a project, producer, logical run, concrete execution, and 1 through 500 complete observations. A logical run can be assembled by independent processes because its key excludes machine and process facts. An observation is replaced by a later submission with the same project, run key, test ID, and case dimensions. Exact HTTP retries reuse the same project-scoped idempotency key and body. Every passed observation has at least one explicitly named metric and unit. The request body limit is 8 MiB.

The checked-in BenchView contract is not copied into SlangPy. This producer emits the small, deliberately selected subset described here and tests exact representative JSON. BenchView remains the authoritative validator at the HTTP boundary.

## Plan of Work

Add `slangpy/testing/benchmark/benchview.py` with small `dict[str, Any]` aliases rather than a parallel exhaustive contract hierarchy. Add helpers to normalize RFC 3339 timestamps, paths and scalar dimensions; map the existing machine dictionary into stable environment identity plus volatile telemetry; construct deterministic run fields; greedily batch observations below both a configurable observation count and the 8 MiB request limit; derive printable idempotency keys; and submit JSON with `urllib.request.Request`. Keep legacy report generation, serialization, and comparison helpers in `report.py`, but delete `upload_report` and all `pymongo` use.

Extend `ReportFixture` in `slangpy/testing/benchmark/fixtures.py` so each measurement appends both its existing local report and a BenchView observation. Use the original pytest function's `_cpu` marker to select `cpu_time` and use `gpu_time` otherwise; both use unit `ms`, direction `lower`, and the raw samples distribution. Normalize the source path, preserve the legacy string parameter representation, and include the adapter name only as observation metadata because full stable GPU identity comes from the session machine record.

Change the context and lifecycle in `slangpy/testing/benchmark/plugin.py`. Cache a fresh execution UUID and native observations. Validate API URL and `BENCHVIEW_API_KEY` during pytest configuration whenever submission is enabled, so a long benchmark run cannot discover missing configuration only at session end. At session finish, retain local save behavior and then build and post every batch. Print only the destination and accepted/duplicate counts, never the key or authorization header.

Update `tools/ci.py` so `benchmark-python` accepts `--api-url` instead of MongoDB connection/database arguments and passes `--benchmark-submit` plus `--benchmark-api-url` to pytest. The environment, including `BENCHVIEW_API_KEY`, is inherited by each process. Remove `pymongo` from `requirements-dev.txt`. Update `.github/instructions/benchmarks.instructions.md` with direct pytest and CI examples, nested URL behavior, the environment key, native metric semantics, and retry/replacement behavior.

Add pure tests under `slangpy/tests/utils/test_benchmark_submission.py`. Cover GPU and CPU observation construction, legacy-compatible dimensions and stable test ID, dual local/native accumulation, deterministic distributed run keys, environment identity/telemetry separation, batch sizing and unique idempotency, root and nested API endpoint construction, Bearer headers, duplicate receipts, and safe HTTP errors that do not disclose the key. These tests must require no GPU, MongoDB, or live BenchView instance.

## Concrete Steps

Run all commands from `C:\sw\slangpy`. Per repository policy, build before tests:

    cmake --build --preset windows-msvc-debug
    pytest slangpy/tests/utils/test_benchmark_submission.py -v
    python tools/ci.py typing-check-python
    pre-commit run --all-files

If pre-commit modifies files, run it again. Inspect `git diff --check` and the focused diff before handoff. A live manual submission, when desired, uses:

    $env:BENCHVIEW_API_KEY = "<write key>"
    pytest slangpy/benchmarks -v --benchmark-submit <request-id> --benchmark-api-url http://localhost:3000

For the hosted nested deployment, replace the base URL with `http://rtrci.nvidia.com/benchview`; the producer appends `/api/v1/submissions` without discarding `/benchview`.

## Validation and Acceptance

The focused tests pass without importing `pymongo` or contacting external services. Their intercepted request has `Content-Type: application/json`, `Authorization: Bearer <configured key>`, and a URL ending in the correct nested `/api/v1/submissions`. The payload uses schema version 1, project `slangpy`, suite `python`, one stable commit/config run key across distinct execution IDs, normalized device dimensions, stable source/function test IDs, environment identity, and `gpu_time` or `cpu_time` millisecond samples as appropriate.

Starting a submission-enabled pytest process without `--benchmark-api-url`/`BENCHVIEW_API_URL` or without `BENCHVIEW_API_KEY` fails during configuration with a concise diagnostic. A non-2xx response fails submission with status and a bounded response message but never includes the key. A 200 duplicate receipt and a 201 accepted receipt are both successes. No production test depends on MongoDB or the BenchView repository.

The Windows debug build completes before tests. The focused pytest suite, Python typing command, pre-commit suite, and `git diff --check` exit zero. `rg pymongo slangpy/testing/benchmark tools/ci.py requirements-dev.txt` finds no remaining direct database upload implementation or dependency.

Final evidence on 2026-07-16: `cmake --build --preset windows-msvc-debug` completed after supplying Visual Studio's installed Ninja and `/EHsc` to the local regenerated cache; `python -m pytest slangpy/tests/utils/test_benchmark_submission.py -v` passed 9 tests; `python -m pyright` reported 0 errors and 0 warnings; and `python -m pre_commit run --all-files` passed every hook. A representative generated request sent through `apps/api/src/app.ts` with an in-memory BenchView transaction log returned HTTP 201, cursor 0, and observation count 1. That check used a synthetic key and did not connect to MongoDB.

Separation evidence on 2026-07-16: the focused suite again passed 9 tests, targeted Pyright over the changed benchmark modules and test reported 0 errors and 0 warnings, both the full tracked-file pre-commit run and the explicit new-file run passed, and `git diff --check` completed cleanly. A full Pyright rerun currently reports only the unrelated pathtracer findings recorded under Surprises and Discoveries.

## Idempotence and Recovery

Payload generation is side-effect free. The HTTP sender posts immutable serialized bodies; an in-process retry sends the same body and idempotency key. If a process is rerun, its fresh execution UUID and timestamps produce fresh keys, so BenchView treats results as intentional replacements. Batches are independent transactions, so a later batch can fail after earlier batches commit. Rerunning the benchmark safely resubmits a complete new set; retrying a captured failed batch safely reuses its original body.

Local `.benchmarks` save and comparison files retain their legacy shape. No migration or deletion is performed. The implementation removes only the obsolete direct database code and its unused development dependency.

## Artifacts and Notes

The intended native metric core is compact:

    "metrics": [{
        "id": "gpu_time",
        "name": "GPU time",
        "unit": "ms",
        "direction": "lower",
        "distribution": {"samples": [...]}
    }]

The logical run key has the form:

    git:<full revision>/suite:python/config:<16 lowercase digest characters>

The API key is read from `BENCHVIEW_API_KEY` only and is never persisted in a report or plan.

## Interfaces and Dependencies

`slangpy.testing.benchmark.benchview` exposes `BenchViewObservation = dict[str, Any]`, `build_benchview_observation(...)`, `build_benchview_submissions(...)`, and `submit_benchview_submissions(...)`. `slangpy.testing.benchmark.report` continues to expose the legacy `BenchmarkReport`, `Report`, and local serialization helpers. All arguments have type annotations. HTTP uses `urllib.request`; hashing and JSON use `hashlib` and `json`; execution IDs use `uuid`. No third-party dependency is added.

`ReportFixture.__call__` accepts optional `metric_id` and `metric_name` overrides and otherwise selects the historical `_cpu`/GPU mapping. The pytest context includes `benchmark_observations` and `execution_id`. `tools/ci.py benchmark-python` exposes `--api-url`; `BENCHVIEW_API_KEY` remains an inherited environment variable rather than an argparse option.

Revision note (2026-07-16, Codex): Created this ExecPlan before implementation because replacing a benchmark storage protocol affects fixture data modeling, pytest lifecycle, CI arguments, security, compatibility, and validation across several SlangPy modules.

Revision note (2026-07-16, Codex): Corrected the initial native-type and wrapper-based metric assumptions after comparing them with imported production history. Native submissions now preserve string parameter identities and the `_cpu` marker, preventing graph splits or mislabeled synchronized GPU benchmarks.

Revision note (2026-07-16, Codex): Completed implementation and validation. Added native observation accumulation, deterministic run/idempotency construction, environment mapping, size-bounded batching, nested-base authenticated HTTP, early configuration errors, API-oriented CI options, documentation, and nine focused tests; removed direct MongoDB integration and recorded the successful real BenchView in-memory acceptance check.

Revision note (2026-07-16, Codex): Separated the native BenchView producer into `benchview.py` and returned `report.py` to its legacy local-report responsibility; updated imports, tests, documentation, and this plan to describe the module boundary.
