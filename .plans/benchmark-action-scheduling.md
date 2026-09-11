# Separate ordinary, nightly, and historical benchmark workflows

This ExecPlan is a living document. The sections Progress, Surprises and Discoveries, Decision Log, and Outcomes and Retrospective must be kept up to date as work proceeds.

This plan follows `.agents/PLANS.md` from the repository root.

## Purpose / Big Picture

After this change, SlangPy's normal benchmark workflow continues to build the selected current revision and submit results to BenchView through the existing `tools/ci.py benchmark-python` command. It contains no historical-source patching and can be tested independently before the old four-hour schedule is removed. A nightly scheduler then finds `main` commits from the preceding 24 hours and starts that ordinary workflow once for every returned commit.

A separate `backfill-benchmark.yml` workflow handles the exceptional task of benchmarking older `main` revisions. It builds an unmodified historical checkout in a temporary clone, overlays the current BenchView reporting files only after compilation, and then runs the normal current `tools/ci.py` from inside that clone. Backfill has an explicit earliest supported commit. Revisions older than that boundary are out of scope; the implementation must advance the boundary if the first pilot proves that supporting it would require additional legacy compatibility code.

The nightly workflow uses GitHub's own authenticated `actions/github-script` client to dispatch ordinary workflows directly. It needs no repository checkout, Python process, GitHub CLI, extra token, or existing-run lookup. The resumable local backfill scheduler uses the official GitHub `gh` command-line client and no third-party Python package. A developer can observe success by manually running the ordinary workflow, seeing Windows and Linux submissions in BenchView with the expected commit identity, triggering the nightly workflow and previewing local backfill without dispatching, successfully running one boundary backfill, interrupting and resuming a small backfill, and finally allowing the backfill to work forward while never exceeding four active workflow runs.

## Progress

- [x] (2026-07-16 15:07Z) Built a working prototype that enumerates live `main` history, dispatches workflows through the REST API, persists atomic backfill state, and submits current benchmark observations to BenchView.
- [x] (2026-07-17 09:43Z) Rejected the prototype's combined ordinary/backfill workflow and separate `tools/run_benchmark_ci.py` runner in favor of the three-path design recorded here.
- [x] (2026-07-17 09:43Z) Selected the official `gh` CLI for GitHub access and selected commit `f3ad0fd91d8cf4eeb2be3b505765b43482aa952a` as the initial, provisional backfill boundary.
- [x] (2026-07-17 09:52Z) Milestone 1 implementation and local validation: restored the ordinary scheduled/manual workflow, retained only the BenchView transport changes in `tools/ci.py`, removed the rejected scheduler/runner prototype, added ordinary-path tests and documentation, built successfully, passed 12 focused tests, passed targeted and full Pyright, and passed pre-commit.
- [x] (2026-07-17 12:02Z) Milestone 1 live acceptance: [workflow run 29577592368](https://github.com/shader-slang/slangpy/actions/runs/29577592368) completed successfully at `7708f86d51ca35d52fc573f9972d9399f40718b6`. Both Windows and Linux jobs configured, built, installed the in-tree `slangpy-torch`, reached `tools/ci.py benchmark-python`, and had BenchView accept benchmark batches. The deployed graph API records passed observations for that exact revision from Windows host `CI-VM` and Linux host `kernelvm-9c2ad50b`.
- [x] (2026-07-17) Repaired and pushed the first live-run failures on `dev/ccummings/benchview`: commit `a022e2e7` installs PyTorch and the in-tree `slangpy-torch` bridge, and commit `c92535aa` resolves `ppisp/extensions.slang` and gives every CUDA-only PPISP benchmark an explicit device dimension. The Windows debug build passed in an MSVC developer environment, 16 focused tests passed, the `nodevice` PPISP run skipped all 24 cases for the explicit CUDA-selection reason, Pyright reported zero findings, pre-commit passed, and `git diff --check` passed.
- [x] (2026-07-17 13:40Z) Audited the Milestone 2/3 starting point at commit `8453b4e1`: the rejected scheduler prototype is absent from the current tree and reachable history, the ordinary workflow still has its four-hour schedule and no revision input, the provisional floor resolves locally at `2025-09-02T14:42:35Z`, and unrelated documentation, CMake, plan, and submodule changes remain preserved.
- [x] (2026-07-17) Implemented the local Milestone 2/3 structure: ordinary CI now selects an optional exact revision without historical logic; nightly scheduling uses `actions/github-script` directly; historical work has a manual temporary-clone workflow with a pre-build floor guard, post-build reporting overlay, exact source override, and guarded cleanup; and the local scheduler has a no-shell typed `gh` adapter, versioned atomic state, deterministic-title recovery, oldest-first dispatch, and a four-run cap.
- [x] (2026-07-17) Completed local Milestone 2/3 validation: the Visual Studio 18 debug build passed, all 38 focused submission and scheduling tests passed under the repository's Python 3.12 environment, targeted Pyright reported zero findings, full pre-commit passed including YAML validation, and `git diff --check` passed. A completion audit additionally bounded minute-by-minute workflow polling, kept uncertain-only state alive through its reconciliation grace period, verified historical source identity and Ctrl+C status, and selected GitHub API version `2026-03-10`. Live dispatch pilots remain pending authentication and publication of the workflows.
- [x] (2026-07-17) Simplified both hosted workflows after user review: nightly scheduling now dispatches every `main` commit from the fixed 24-hour interval without reading existing runs or exposing dry-run behavior, while historical cloning, boundary validation, submodule synchronization, and guarded cleanup use ordinary Git plus native PowerShell/Bash rather than inline Python programs.
- [ ] Milestone 2: disable the old schedule, add exact-revision selection for future commits, and validate direct GitHub Actions scheduling of ordinary workflows.
- [ ] Milestone 3: add the dedicated temporary-clone backfill workflow, prove or advance the compatibility boundary, redirect the resumable local scheduler to it, and validate interruption recovery and the four-run cap.
- [ ] Complete repository build, focused tests, Pyright, pre-commit, workflow syntax checks, documentation, live dispatch pilots, and the final operator handoff.

## Surprises and Discoveries

- Observation: a `workflow_dispatch` request selects the workflow by branch or tag, not by an arbitrary detached commit SHA.
  Evidence: GitHub's workflow dispatch request requires a `ref` that is a branch or tag. To benchmark an exact older commit while using the current workflow definition, the request must use `ref: main` and pass the desired SHA through a declared workflow input consumed by `actions/checkout`.

- Observation: current GitHub.com dispatch responses can include the new workflow run ID and URLs immediately.
  Evidence: the current API accepts `return_run_details: true` and responds with `workflow_run_id`, `run_url`, and `html_url`. The scheduler can persist that ID without waiting for the new run to appear in a later listing, while deterministic run titles still cover a crash between dispatch and state publication.

- Observation: commit `f3ad0fd91d8cf4eeb2be3b505765b43482aa952a`, dated 2 September 2025, introduced `--device-types` and the per-device benchmark loop used by current `tools/ci.py`.
  Evidence: `git log -S"--device-types" -- slangpy/testing/plugin.py tools/ci.py` identifies that commit. Earlier commits require a different execution shape, so this plan initially excludes them rather than maintaining an additional benchmark runner.

- Observation: historical revisions still contain the obsolete MongoDB reporter even after the provisional device-filter boundary.
  Evidence: the current BenchView files under `slangpy/testing/benchmark/` must be overlaid after historical setup, configure, and build. The current `tools/ci.py` and `tools/gpu_clock.py` must also be overlaid for the benchmark step, but they must not configure or compile the historical source.

- Observation: retrieving 1,000 workflow runs through the prototype required as many as ten paginated requests and took roughly 35 seconds.
  Evidence: live no-write previews on 16 July 2026 took 35 to 42 seconds. The final backfill should do a broad title reconciliation only on startup and periodic refresh, while minute-by-minute capacity polling reads only recent runs.

- Observation: `gh api --paginate --slurp` downloads every available workflow-run page before Python can apply its `maximum` slice.
  Evidence: the audited adapter used that form even for the minute-by-minute `maximum=100` capacity poll. It now requests explicit bounded pages and stops as soon as the requested maximum or the final short page is reached; `test_github_cli_fetches_only_the_requested_workflow_run_pages` locks this behavior.

- Observation: a state containing only a recent uncertain `dispatching` record is not complete even though it has no immediately pending record.
  Evidence: the original loop predicate exited on an empty pending list and incorrectly printed that all commits were requested. The scheduler now remains active while either pending or uncertain records exist, and an offline time-stepped test proves it waits through the grace period before retrying an unmatched request.

- Observation: the repository contained 355 `main` commits since 1 September 2025 when the prototype was tested.
  Evidence: local Git history and the live GitHub commits endpoint agreed. The final supported count will be slightly smaller because the provisional boundary excludes commits before the afternoon of 2 September and may move forward after the pilot.

- Observation: an unset GitHub repository variable is passed to a workflow command as an explicit empty string.
  Evidence: the ordinary workflow always supplies `--api-url "${{ env.BENCHVIEW_API_URL }}"`. `tools/ci.py` now distinguishes an omitted option from an explicitly empty option, forwarding the latter to pytest so `pytest_configure` reports the missing URL/key instead of silently disabling submission. `test_ci_wrapper_forwards_an_explicit_empty_api_url` covers this behavior.

- Observation: GitHub CLI authentication was unavailable during the first implementation pass but was renewed before final acceptance.
  Evidence: `gh auth status` initially reported the `ccummingsNV` token as invalid. During the final audit it identified `ccummingsNV` as the active authenticated account, allowing the successful workflow and job logs to be inspected without exposing the token.

- Observation: the local `gh` executable is available for the Milestone 3 backfill controller, but its saved authentication became invalid again before this implementation pass.
  Evidence: `gh --version` reports 2.94.0, while `gh auth status` reports the active `ccummingsNV` credential as invalid. Fake-runner unit coverage and local code work remain possible, but local backfill previews or dispatch pilots require `gh auth login -h github.com` first. Nightly scheduling is unaffected because `actions/github-script` receives GitHub's job token automatically.

- Observation: the existing Windows debug build directory was configured with Visual Studio 18, so entering a Visual Studio 2022 developer environment produced standard-library linker mismatches rather than a source failure.
  Evidence: the VS2022 attempt failed on unresolved `__std_*` symbols, while the same incremental build under `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat` linked all 23 remaining targets successfully.

- Observation: the benchmark workflow's copied PyTorch setup condition referenced the `unit-test` flag, while its matrix supplies only the `benchmark` flag, and the workflow did not install the compiled `slangpy-torch` bridge after building.
  Evidence: the live CUDA tensor benchmarks raised `PyTorch tensors detected but slangpy-torch is not installed`. The ordinary unit-test workflow already establishes the required order: install PyTorch, build SlangPy, then run `python tools/ci.py install-slangpy-torch`.

- Observation: `test_benchmark_bwd_diff.py` searched the repository root for `extensions.slang`, but the only supplied file is `slangpy/benchmarks/ppisp/extensions.slang`.
  Evidence: the live Slang compiler reported include error E15300, and the local repository inventory plus the corrected include-path regression test identify the PPISP directory as the valid source.

- Observation: device-isolation classification is driven by the test function's explicit `device_type` parameter, not by a device requested later inside the test body.
  Evidence: PPISP tests hard-coded `get_torch_device(DeviceType.cuda)` without declaring the parameter, so the plugin assigned them to the `nodevice` shard. After adding a CUDA parameter to all such tests, the local `--device-types nodevice -rs` run reports 24 skips from `slangpy/testing/plugin.py` with target device `cuda`.

## Decision Log

- Decision: keep `.github/workflows/ci-benchmark.yml` as the ordinary benchmark workflow and remove all historical harness overlay behavior from it.
  Rationale: scheduled and manually selected current revisions should exercise the same simple path developers already understand. Historical compatibility must not make routine benchmarks harder to test or maintain.
  Date/Author: 2026-07-17, Codex.

- Decision: implement and validate the ordinary BenchView migration before disabling its existing four-hour schedule.
  Rationale: this produces a small first change that can be run on the real performance workers and verified in BenchView before scheduling and historical behavior are introduced.
  Date/Author: 2026-07-17, Codex.

- Decision: after ordinary validation, add one optional `revision` input to `ci-benchmark.yml`.
  Rationale: dispatching the workflow on a branch naturally benchmarks the branch tip. GitHub does not support a detached SHA as the dispatch `ref`, so the optional input is the smallest generic mechanism for nightly scheduling or manually selecting an exact commit. It is not a legacy compatibility feature and may target only revisions that already contain the BenchView producer.
  Date/Author: 2026-07-17, Codex.

- Decision: use `.github/workflows/backfill-benchmark.yml` exclusively for pre-BenchView historical revisions.
  Rationale: the workflow can own temporary cloning, post-build overlays, source identity correction, and the supported-history boundary without leaking those concerns into ordinary CI.
  Date/Author: 2026-07-17, Codex.

- Decision: use `f3ad0fd91d8cf4eeb2be3b505765b43482aa952a` as a provisional inclusive backfill floor and do not support earlier commits.
  Rationale: this is the first commit with the device-selection interface required by the normal current `tools/ci.py`. The user explicitly accepts a finite history. If an end-to-end pilot at this SHA reveals another difficult compatibility boundary, move the floor forward and record the first passing SHA instead of adding a special legacy runner.
  Date/Author: 2026-07-17, Codex.

- Decision: build historical source with its own `tools/ci.py`, then overlay the current reporting harness and current `tools/ci.py` only for benchmark execution.
  Rationale: historical setup and native compilation must retain the historical build behavior, while the benchmark process needs the current HTTP submission interface. Executing the copied current `tools/ci.py` from inside the temporary clone gives it the correct project paths and removes the need for `tools/run_benchmark_ci.py`.
  Date/Author: 2026-07-17, Codex.

- Decision: use `actions/github-script` directly for nightly scheduling and retain the official `gh` CLI only for the local backfill controller.
  Rationale: the nightly scheduler already executes inside GitHub Actions, where the action provides an authenticated Octokit client that can enumerate commits and dispatch `workflow_dispatch` events. Python and `gh` would only add setup there. Backfill remains a long-running, resumable local process, where `gh` supplies authentication and API transport without a custom HTTP client or third-party Python dependency.
  Date/Author: 2026-07-17, Codex.

- Decision: interpret “commits pushed to main” as commits reachable from `main`, filtered by committer timestamp.
  Rationale: historical push events are not a durable commit index, whereas GitHub's commits endpoint can enumerate landed history. This creates one benchmark request per landed commit.
  Date/Author: 2026-07-17, Codex.

- Decision: use deterministic titles `ci-benchmark: <SHA>` for traceability and `backfill-benchmark: <SHA>` for local-state reconciliation, while persisting a write-ahead `dispatching` state for backfill.
  Rationale: nightly intentionally dispatches every commit in every invocation and does not deduplicate by title. Backfill titles still reconcile the narrow case where the local process exits after GitHub accepts the request but before its state file is replaced.
  Date/Author: 2026-07-17, Codex.

- Decision: count every non-completed `backfill-benchmark` workflow as active and dispatch at most one oldest pending commit per minute while fewer than four are active.
  Rationale: this implements the requested workflow-level pressure limit without depending on the two Windows/Linux jobs inside each matrix run.
  Date/Author: 2026-07-17, Codex.

- Decision: preserve local no-submission use when `--api-url` is omitted, but fail through the pytest plugin when the workflow explicitly passes an empty API URL.
  Rationale: local developers may still use `tools/ci.py benchmark-python` without a server, while a misconfigured production workflow must not appear successful after discarding all results.
  Date/Author: 2026-07-17, Codex.

- Decision: make the ordinary benchmark workflow install and remove the in-tree `slangpy-torch` extension using the same post-build command as ordinary unit CI.
  Rationale: installing the `torch` wheel alone does not provide SlangPy's tensor bridge. Building the bridge from the checked-out revision keeps it ABI-aligned with that revision, and cleanup prevents contamination of persistent performance runners.
  Date/Author: 2026-07-17, Codex.

- Decision: require CUDA-only benchmark cases to expose `device_type=cuda` through pytest parameterization.
  Rationale: this is the routing contract understood by the isolation plugin and also records CUDA as a normal BenchView case dimension. A hard-coded device request inside the test is too late for shard selection.
  Date/Author: 2026-07-17, Codex.

## Outcomes and Retrospective

The earlier prototype proved that live commit discovery, deterministic run reconciliation, atomic local recovery, and native BenchView submission are feasible. It did not dispatch a real benchmark workflow and its architecture is not the desired result: it places historical checkout and overlay logic in `ci-benchmark.yml`, adds a second benchmark runner, and contains a large custom REST client.

Milestone 1 is implemented locally. The resulting `ci-benchmark.yml` differs from its repository baseline only by the BenchView URL/key environment and the two API-based `tools/ci.py` invocations. The obsolete MongoDB dependency and uploader are removed, while local report save/compare data remains. The alternate runner, scheduling workflows/scripts, and custom REST client from the rejected prototype are absent. The Windows debug build completed, 12 focused tests passed, targeted and full Pyright reported zero findings, full and explicit-new-file pre-commit passed, and `git diff --check` found no errors.

Milestone 1 is complete. The first external run exposed stale suite assumptions: PyTorch setup was gated on the wrong matrix flag, `slangpy-torch` was not installed, one Slang include directory was wrong, and CUDA-only PPISP tests were routed to `nodevice`. Commits `a022e2e7` and `c92535aa` repaired those failures, and `7708f86d` restored restricted Linux GPU-clock control without running the Python process as root. Local evidence is a successful Windows debug build, 16 focused tests, 24 correctly skipped PPISP cases in a `nodevice` run, zero Pyright findings, a clean full pre-commit run, and a clean diff check.

The final external evidence is [GitHub Actions run 29577592368](https://github.com/shader-slang/slangpy/actions/runs/29577592368), completed at 12:02Z on 17 July 2026. Its Windows and Linux matrix jobs both completed successfully, installed the checked-out revision's `slangpy-torch` extension, invoked the ordinary `tools/ci.py benchmark-python` path, and received successful BenchView batch acknowledgements. BenchView's deployed read API shows revision `7708f86d51ca35d52fc573f9972d9399f40718b6` with passed Windows observations on `CI-VM` and passed Linux observations on `kernelvm-9c2ad50b`. The original missing-bridge, missing-include, and `nodevice` routing errors do not recur in the final logs. Per-device execution deliberately continues after a failed shard; both final CUDA shards abort in the autograd suite for an unrelated native failure, while other platform/target batches are durably accepted. The user confirmed the benchmark workflow is operational, and this does not block the Milestone 1 transport and ordinary-workflow acceptance gate.

The local Milestone 2 and 3 implementation is complete and validated. User review removed unnecessary hosted-workflow machinery: `schedule-benchmarks.yml` now performs recent-commit discovery and unconditionally dispatches every returned commit through `actions/github-script`, while `backfill-benchmark.yml` uses ordinary Git and native shell cleanup rather than embedded Python. Python and `gh` remain only where they add value—the stateful local historical controller. The Visual Studio 18 debug build, 38 focused offline tests, targeted Pyright, full pre-commit, and the diff check all pass. The remaining acceptance work is inherently live: publish the workflows, run the nightly and exact-revision pilots, renew local `gh` authentication, then run the two historical pilots and interruption/capacity trial.

## Context and Orientation

Run every command in this plan from `C:\sw\slangpy`. The repository is currently on `dev/ccummings/benchview` with Milestone 1 pushed through `7708f86d`; local documentation, `tests/CMakeLists.txt`, plan changes, and the pre-existing `external/slang-rhi` submodule pointer remain outside that commit. Preserve those unrelated changes. The BenchView producer is committed and remains the transport implementation that the later scheduling workflows need.

`.github/workflows/ci-benchmark.yml` is the existing Windows/Linux performance workflow. In the repository baseline it runs every four hours and on manual dispatch, checks out the workflow revision, sets up Python and CMake, invokes `python tools/ci.py setup`, `configure`, and `build`, and finally invokes `python tools/ci.py benchmark-python`. The prototype changed it to accept a historical SHA, overlay reporter files, and call `tools/run_benchmark_ci.py`; those historical changes must move out.

`tools/ci.py` is the normal cross-platform CI entry point. Its `benchmark_python` function selects D3D12, Vulkan, CUDA, Metal, and non-device test shards as appropriate; optionally locks GPU clocks; and invokes pytest. The accepted BenchView change replaces MongoDB arguments with `--api-url` and passes `--benchmark-submit` plus `--benchmark-api-url`. It must remain the only benchmark runner.

`slangpy/testing/benchmark/benchview.py`, `fixtures.py`, `plugin.py`, `report.py`, and `utils.py` form the current BenchView reporting harness. The pytest plugin collects observations, constructs submissions at session completion, and sends them with `BENCHVIEW_API_KEY`. `plugin.py` also contains `apply_benchmark_source_override`, which lets a historical overlaid checkout report the intended source SHA, branch, and clean state rather than the current harness commit and dirty working tree. Ordinary local runs without override environment variables retain their natural Git identity.

`tools/benchmark_actions.py` and `tools/backfill_benchmarks.py` implement the local historical scheduler. The first is a small typed adapter around `gh`; the second owns the resumable state machine. Nightly scheduling does not share this machinery because `.github/workflows/schedule-benchmarks.yml` can enumerate and dispatch workflows with GitHub's own authenticated API client. `tools/run_benchmark_ci.py` was a prototype-only duplicate runner and remains deleted.

A workflow dispatch is a request for GitHub to start a workflow declaring `workflow_dispatch`. The request's `ref` selects a branch or tag containing the workflow definition. An optional workflow input is a named string in that request. Ordinary exact-revision scheduling dispatches `ci-benchmark.yml` from `main` and passes `revision=<SHA>`; historical scheduling dispatches `backfill-benchmark.yml` from `main` and passes `target_sha=<SHA>`.

The local backfill state remains an ignored JSON file at `.temp/benchmark-backfill-state.json`. It records its schema version, repository, branch, workflow, supported lower bound, each commit, dispatch status, and any known run ID and URL. It must never contain GitHub credentials or the BenchView API key. The redesigned state must identify `backfill-benchmark.yml`; an incompatible state written by the prototype must fail with a clear instruction to archive it and start a new state file, never be silently reinterpreted.

The checked-in `.plans/benchview-benchmark-submission.md` describes the producer-side BenchView work. This plan preserves that producer and changes only normal CI invocation, scheduling, and the historical execution boundary.

## Plan of Work

### Milestone 1: prove the ordinary BenchView workflow

First reduce `.github/workflows/ci-benchmark.yml` to its ordinary purpose. Restore the baseline four-hour cron and input-free manual dispatch. Remove the historical `benchmark_ref`, historical checkout selection, overlay step, hard-coded `main` identity, and every invocation of `tools/run_benchmark_ci.py`. Keep the Windows/Linux matrix and existing setup, configure, and build steps. Gate PyTorch setup on the actual `benchmark` matrix flag, install the in-tree `slangpy-torch` bridge after the SlangPy build, and remove it in an always-run cleanup step because the performance workers are persistent. Add only the `BENCHVIEW_API_URL` repository variable and `BENCHVIEW_API_KEY` repository secret to the job environment, and change the two benchmark steps to invoke `python tools/ci.py benchmark-python --run-id "${{ github.run_id }}" --api-url "${{ env.BENCHVIEW_API_URL }}"`, retaining GPU clock locking where supported by the worker configuration.

Keep the accepted BenchView changes in `tools/ci.py`: remove the MongoDB command-line options, define `--api-url`, obtain a default from `BENCHVIEW_API_URL`, and append the current pytest submission options only when an API URL is present. Preserve the existing device selection, failure continuation, and clock restoration behavior. Remove `pymongo` from `requirements-dev.txt` only if no remaining current code or tests import it. Do not add a compatibility fallback for revisions before the chosen boundary.

Keep and finish the producer changes in `slangpy/testing/benchmark/`. Move any tests that only exist for `tools/run_benchmark_ci.py` to normal `tools/ci.py` command-construction coverage if they remain useful, then delete `tools/run_benchmark_ci.py`. Update `.github/instructions/benchmarks.instructions.md` so its ordinary instructions describe the API URL, secret, normal `ci.py` command, and current schedule without mentioning backfill.

Build before testing, run the focused producer and CI-wrapper tests, run the CUDA-only suites through a `nodevice` shard to prove they are excluded before execution, and run a manual ordinary workflow on a revision containing these changes. Acceptance for this milestone requires both matrix jobs to build the in-tree torch bridge, reach `tools/ci.py benchmark-python` without device-routing or include-path failures, have BenchView accept their submissions, and show the selected current commit and correct Windows/Linux hosts. Keep the four-hour schedule in place until that evidence exists. Record the GitHub run URL and BenchView verification here without recording secrets.

### Milestone 2: add exact future scheduling inside GitHub Actions

After Milestone 1 passes, remove the four-hour cron from `.github/workflows/ci-benchmark.yml`. Add a single optional string input named `revision`. Set the deterministic run name to `ci-benchmark: ${{ inputs.revision || github.sha }}` and set the checkout ref to the same expression. Export `BENCHVIEW_BENCHMARK_REF` as the selected revision and `BENCHVIEW_BENCHMARK_BRANCH` as `github.ref_name`, so dispatching from `main` with an older SHA still reports branch `main`. This input is for revisions that contain the current BenchView producer; it does not trigger overlays or historical compatibility.

Keep `.github/workflows/schedule-benchmarks.yml` as a small nightly and manually dispatched job with `contents: read` and `actions: write`. Its only step uses `actions/github-script`. The script fixes one UTC interval at startup, beginning 24 hours earlier, paginates `main` commits in that interval, and dispatches every commit oldest first from workflow ref `main` with input `revision=<SHA>`. It deliberately does not list existing workflow runs, suppress duplicate titles, or expose a dry-run mode. It needs no checkout, Python setup, `gh` executable, or extra secret because `actions/github-script` receives the repository's job token. GitHub explicitly allows `workflow_dispatch` events created with `GITHUB_TOKEN` to start new workflow runs. Concurrency prevents overlapping scheduler instances.

Acceptance requires one controlled scheduler invocation to dispatch exactly the commits returned for its 24-hour interval, plus an exact-revision `ci-benchmark.yml` run for a recent non-tip `main` SHA. The resulting BenchView observation must identify that exact SHA and `main`, proving the generic revision input works before it is used nightly.

### Milestone 3: add bounded historical backfill

Create `.github/workflows/backfill-benchmark.yml`. It is manual-only, declares required string input `target_sha`, uses deterministic run name `backfill-benchmark: ${{ inputs.target_sha }}`, and uses the same Windows/Linux performance-runner matrix and Python/CMake preparation as the ordinary workflow. The workflow definition is always dispatched from `main`, so `github.sha` identifies the current harness revision while `inputs.target_sha` identifies the historical source revision.

For each matrix job, use ordinary Git commands to make a unique normal recursive clone below `runner.temp`, check out `target_sha` detached, run `git submodule sync --recursive`, `git submodule update --init --recursive`, and Git LFS retrieval, and install the target revision's development and sample requirements. Do not embed a Python program for these repository operations. Validate that the target is the provisional floor commit or a descendant by running `git merge-base --is-ancestor f3ad0fd91d8cf4eeb2be3b505765b43482aa952a <target_sha>`. If not, stop before setup with a message naming the earliest supported SHA.

From the temporary clone, invoke the target revision's `python tools/ci.py setup`, `configure`, and `build`. Only after a successful build, overlay the current harness by running `git checkout "${{ github.sha }}" --` for `tools/ci.py`, `tools/gpu_clock.py`, and the required files under `slangpy/testing/benchmark/`. Do not overlay native source, CMake files, the global test plugin, or the build configuration. Then invoke the overlaid `python tools/ci.py benchmark-python` from the temporary clone with the same API arguments as the ordinary workflow. Set `BENCHVIEW_BENCHMARK_REF` to `target_sha` and `BENCHVIEW_BENCHMARK_BRANCH` to `main`; the reporter override must emit `dirty: false` while retaining timestamps and other source facts from historical HEAD.

Clean the unique temporary clone in OS-specific `if: always()` steps using native PowerShell on Windows and Bash on Linux. Before recursively deleting anything, resolve both the candidate and `runner.temp`, require the candidate's direct parent to equal `runner.temp`, and require its name to start with the workflow's fixed backfill prefix. A failed safety check must leave the directory in place and fail cleanup rather than delete an unexpected path. Do not embed Python for cleanup.

The first live backfill run must target `f3ad0fd91d8cf4eeb2be3b505765b43482aa952a`. If it builds, runs, and submits valid Windows and Linux observations, record it as the final inclusive floor. If it fails because the reporting harness or benchmark command cannot be overlaid without more compatibility code, inspect later history and advance the floor to the first commit that completes the same pilot. Update the workflow guard, scheduler default, tests, documentation, Decision Log, and this section together. Do not add a second benchmark runner or a pre-device-filter fallback merely to retain older commits.

Refactor `tools/backfill_benchmarks.py` so its default workflow is `backfill-benchmark.yml` and its default lower bound is the final supported boundary. Preserve the existing oldest-first, one-dispatch-per-minute behavior, four-active-run limit, atomic state replacement, `dispatching` write-ahead marker, deterministic-title reconciliation, transient retry, `--dry-run`, and `--once`. Count only non-completed runs of `backfill-benchmark.yml`; ordinary and nightly `ci-benchmark` runs do not consume backfill capacity.

Implement `tools/benchmark_actions.py` as a small typed adapter around `gh` for this local controller. It locates `gh` with `shutil.which`, fails with a concise installation/authentication message when unavailable, and executes commands with `subprocess.run` using argument arrays, captured UTF-8 output, `check=False`, and no shell. Authentication comes from `gh auth login`; do not accept or print a token argument. Commit listing uses `gh api --paginate --slurp`; workflow-run listing requests explicit bounded pages and flattens each page's `workflow_runs`; and dispatch sends exact JSON on standard input with `return_run_details: true`. Add current GitHub API version `2026-03-10` to every call, retain typed `Commit`, `WorkflowRun`, and `DispatchResult` records, and inject the command runner in tests.

Revise the state schema and compatibility validation to bind it to the repository, `main`, `backfill-benchmark.yml`, and supported lower bound. Before calling `gh`, atomically save a record as `dispatching`; after the POST succeeds, save its returned run ID and URL as `dispatched`. On restart, reconcile `dispatching` records against `backfill-benchmark: <SHA>` titles. Retry an unmatched uncertain dispatch only after the documented grace period. A prototype state with a different workflow or schema must produce a clear error explaining how to archive it; never mutate it silently.

Update `slangpy/tests/utils/test_benchmark_action_scheduling.py` and `.github/instructions/benchmarks.instructions.md`. Tests must cover the nightly workflow's direct commit listing and unconditional exact-input dispatch; `gh` JSON parsing and errors through a fake command runner; native historical workflow operations and cleanup guards; backfill-only active counts, oldest-first dispatch, four-run capacity, state mismatch, atomic reload, returned run details, uncertain-title reconciliation, and boundary rejection. Documentation must explain the nightly unconditional direct GitHub API path, then show `gh --version`, `gh auth status`, backfill dry-run and real commands, state location, Ctrl+C/restart behavior, the supported floor, and the prohibition on two simultaneous backfill schedulers sharing one state file.

Acceptance requires successful pilots at the final floor and at one recent pre-BenchView commit, an interrupted two-or-more-commit scheduler trial that resumes without duplicate dispatch, and observed capacity never exceeding four active backfill workflow runs.

## Concrete Steps

Work from the repository root and inspect the dirty tree before each milestone:

    cd C:\sw\slangpy
    git status --short
    git diff -- .github/workflows/ci-benchmark.yml tools/ci.py requirements-dev.txt slangpy/testing/benchmark tools

Verify the local backfill client's authentication before running it:

    gh --version
    gh auth status

The first command must print a GitHub CLI version. The second must identify an authenticated GitHub host without printing a token. Nightly scheduling does not use this client.

Per repository policy, build before running Python tests:

    cmake --build --preset windows-msvc-debug
    python -m pytest slangpy/tests/utils/test_benchmark_submission.py slangpy/tests/utils/test_benchmark_action_scheduling.py -v
    python -m pyright tools/ci.py tools/benchmark_actions.py tools/backfill_benchmarks.py slangpy/testing/benchmark slangpy/tests/utils/test_benchmark_action_scheduling.py
    pre-commit run --all-files
    git diff --check

If pre-commit edits files, repeat the affected tests and pre-commit command. New untracked files must also be passed explicitly to `pre-commit run --files` if the installed pre-commit version does not include them in `--all-files`.

Manually running `schedule-benchmarks.yml` is a real scheduler invocation. Expected output names the fixed UTC interval and reports one successful `ci-benchmark.yml` dispatch for every `main` commit returned in that interval, oldest first.

Preview the supported historical inventory:

    python tools/backfill_benchmarks.py --dry-run

Expected output names `backfill-benchmark.yml`, the final inclusive boundary SHA or timestamp, the number of supported commits, existing deterministic backfill titles, missing commits, and active backfill runs. It must not write `.temp/benchmark-backfill-state.json` and must not dispatch.

Run or resume the real backfill only after both live pilot workflows pass:

    python tools/backfill_benchmarks.py

Pressing Ctrl+C must return exit status 130 after the most recent state replacement. Repeating the same command must load that state, reconcile GitHub runs, and continue from the oldest undispatched supported commit.

## Validation and Acceptance

Milestone 1 is accepted only when `.github/workflows/ci-benchmark.yml` has no historical input, clone overlay, or alternate runner; both performance matrix jobs call `tools/ci.py`; and a real run produces readable BenchView observations for its selected current SHA. Merely passing unit tests is insufficient because this milestone exists to prove the real worker, GPU, authentication, and server path.

Milestone 2 is accepted when manual branch dispatch still benchmarks the branch tip with no revision input, exact-revision dispatch benchmarks a selected non-tip commit containing BenchView support, and one scheduler invocation dispatches every `main` commit from its preceding 24-hour interval. The scheduler workflow must contain no existing-run lookup, deduplication, dry-run behavior, checkout, Python setup, `gh` setup, or extra credential.

Milestone 3 is accepted when the workflow rejects a commit before the supported floor before building, the final floor pilot and a later historical pilot both submit the exact target SHA as branch `main`, and the historical native build occurs before any current files are overlaid. `tools/run_benchmark_ci.py` must no longer exist. Current `tools/ci.py` must not contain a fallback for revisions earlier than the supported floor.

Scheduler tests must run without a GPU, GitHub authentication, network, or real `gh` executable. With three active backfill runs, one oldest pending commit is dispatched; with four active runs, none is dispatched. A successful dispatch stores its returned run ID. A state left at `dispatching` becomes dispatched when a matching deterministic title appears, and becomes pending only after the grace period if no matching run appears. An incompatible prototype state is rejected without modification.

The final build and focused tests must pass, targeted Pyright must report zero errors, pre-commit must pass after any automatic edits, and `git diff --check` must report no whitespace errors. Record exact counts and live pilot URLs in Progress and Outcomes and Retrospective as they become available.

Local Milestone 1 evidence recorded on 17 July 2026 is:

    cmake --build --preset windows-msvc-debug
    [0/2] Re-checking globbed directories...

    python -m pytest slangpy/tests/utils/test_benchmark_submission.py -q
    12 passed in 0.05s

    python -m pyright
    0 errors, 0 warnings, 0 informations

    pre-commit run --all-files
    Passed

The first pre-commit pass reformatted `tools/ci.py`; the focused tests and Pyright were rerun afterward before the clean pre-commit result.

## Idempotence and Recovery

Ordinary and nightly workflow dispatches use traceable `ci-benchmark: <SHA>` titles. The nightly scheduler is stateless and intentionally dispatches every commit returned by each invocation, even when a matching title already exists. Because nightly commit volume is small, it dispatches the complete recent interval oldest first and does not share the backfill four-run throttle.

The backfill state is replace-only: write a complete sibling temporary JSON file, flush it, and atomically replace the final path. Commit discovery is additive and must not reset dispatched records. The local process saves `dispatching` before calling `gh`, then saves returned run details. If interrupted at any point, repeating the command is safe. Full title reconciliation occurs at startup and periodically; recent-run polling is sufficient between full scans for capacity and newly dispatched runs.

Do not run two backfill processes against the same state path. The CLI and documentation must warn about this. If the state file is corrupt or incompatible, preserve it, print its path and the reason, and stop. Starting over requires the operator to archive the old file explicitly; the scheduler can then reconstruct known workflow requests from deterministic GitHub titles before dispatching anything new.

The backfill workflow creates a unique temporary clone per matrix job. Setup, configure, and build may be rerun because the clone is disposable. Reporter overlay happens only after build. Cleanup is allowed only after the path safety checks described above. Failure at any earlier step leaves the authoritative repository and scheduler state unchanged; GitHub's ordinary workflow rerun can repeat the same target SHA.

## Artifacts and Notes

The initial provisional boundary is:

    f3ad0fd91d8cf4eeb2be3b505765b43482aa952a  2025-09-02  System to allow tests to be isolated to specific platform (#478)

Typical ordinary and backfill titles are:

    ci-benchmark: 0123456789abcdef0123456789abcdef01234567
    backfill-benchmark: f3ad0fd91d8cf4eeb2be3b505765b43482aa952a

A typical exact ordinary dispatch created by `actions/github-script` is conceptually:

    {"ref":"main","inputs":{"revision":"0123456789abcdef0123456789abcdef01234567"},"return_run_details":true}

A typical historical dispatch body is:

    {"ref":"main","inputs":{"target_sha":"f3ad0fd91d8cf4eeb2be3b505765b43482aa952a"},"return_run_details":true}

The default state path is:

    .temp/benchmark-backfill-state.json

The state contains commit identities, scheduling status, and workflow run references only. It contains no GitHub token and no BenchView key.

## Interfaces and Dependencies

No new Python package is added. Nightly scheduling uses `actions/github-script@v9` and the job's automatic GitHub token. The local backfill controller depends on GitHub CLI `gh`, authenticated with `gh auth login`, plus Python 3.10 or newer. BenchView configuration remains repository variable `BENCHVIEW_API_URL` and repository secret `BENCHVIEW_API_KEY`.

`tools.benchmark_actions.GitHubCli` must expose typed methods equivalent to `list_commits(repository: str, branch: str, since: datetime, until: datetime) -> list[Commit]`, `list_workflow_runs(repository: str, workflow: str, maximum: int) -> list[WorkflowRun]`, and `dispatch_workflow(repository: str, workflow: str, workflow_ref: str, inputs: dict[str, str]) -> DispatchResult`. Its constructor accepts an injectable command runner for tests. Every function and method has typed arguments and a documenting docstring, as required by `AGENTS.md`.

`tools.backfill_benchmarks.main(argv: Optional[list[str]] = None) -> int` is the only local scheduler entry point. `BackfillStateStore` atomically loads and saves the versioned state. Backfill scheduling helpers remain transport-independent so fake `GitHubCli` implementations can prove behavior without executing subprocesses.

`slangpy.testing.benchmark.plugin.apply_benchmark_source_override(report: Report) -> None` remains the single source-identity override. `tools.ci.benchmark_python(args: Any)` remains the single benchmark runner. There is no `tools/run_benchmark_ci.py` in the completed tree.

Revision note (2026-07-16, Codex): Created the first plan and prototype around one combined target-aware `ci-benchmark.yml`, a custom standard-library REST client, and a dedicated historical benchmark runner.

Revision note (2026-07-17, Codex): Replaced that design after user review. The ordinary workflow is now validated first and contains no backfill behavior; nightly scheduling uses an optional exact revision only for commits containing BenchView; historical work moves to `backfill-benchmark.yml`; the normal current `ci.py` is overlaid after historical build; support begins at a demonstrated compatibility boundary; and the initial revision used the official `gh` CLI for all GitHub access.

Revision note (2026-07-17, Codex): Simplified nightly scheduling after further user review. `schedule-benchmarks.yml` now uses `actions/github-script` and the automatic job token directly, so `tools/nightly_benchmarks.py`, checkout, Python setup, and `gh` setup are absent from the nightly path. The official `gh` adapter remains only for the resumable local backfill controller.

Revision note (2026-07-17, Codex): Removed the remaining hosted-workflow overengineering after another user review. Nightly now dispatches every `main` commit from the preceding 24 hours without existing-run lookup or dry-run behavior. Historical clone, checkout, boundary, synchronization, and cleanup operations now use Git and native PowerShell/Bash rather than embedded Python.

Revision note (2026-07-17, Codex): Implemented the local portion of Milestone 1. Restored the baseline scheduled/manual workflow shape, routed both performance jobs through the single current `tools/ci.py`, removed MongoDB and the rejected scheduling/alternate-runner prototype, added configuration and workflow regression tests, and completed all local gates. Left Milestone 1 live acceptance open until the uncommitted changes can run on GitHub's performance workers.
