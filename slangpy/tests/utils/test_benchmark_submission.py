# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, Optional, Type
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
import yaml

benchmark_api = import_module("slangpy.testing.benchmark.benchview")
_json_bytes = benchmark_api._json_bytes
benchmark_plugin = import_module("slangpy.testing.benchmark.plugin")
ci = import_module("tools.ci")
gpu_clock = import_module("slangpy.testing.benchmark.gpu_clock")

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FakeDeviceType:
    """Provide the enum-name surface used by real SlangPy device types."""

    name = "cuda"


class FakeResponse:
    """Act as the small urllib response surface consumed by the sender."""

    def __init__(self, status: int, body: bytes):
        super().__init__()
        self.status = status
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Any,
    ) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]


class FakePytestConfig:
    """Provide the option and context surface used during plugin configuration."""

    def __init__(self, submit: Any, api_url: Optional[str]):
        super().__init__()
        self.options = {"benchmark_submit": submit, "benchmark_api_url": api_url}

    def getoption(self, name: str) -> Any:
        return self.options[name]


def make_observation(metric_id: str = "gpu_time") -> dict[str, Any]:
    """Build one representative native observation for payload tests."""

    return benchmark_api.build_benchview_observation(
        filename="slangpy/benchmarks/test_benchmark_tensor.py",
        function_name="test_tensor_sum",
        display_name="test_tensor_sum[cuda-1024]",
        parameters={"device_type": FakeDeviceType(), "element_count": 1024, "contiguous": True},
        observed_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
        metrics=[
            benchmark_api.build_metric(
                metric_id,
                "GPU time" if metric_id == "gpu_time" else "CPU time",
                [1.25, 1.5, 1.0],
            )
        ],
        adapter_name="NVIDIA Test GPU",
        source_line=42,
    )


def submission_context() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return deterministic project, machine, and commit facts for batching tests."""

    project_info = {
        "name": "slangpy",
        "version": "0.41.0",
        "slang_build_tag": "v2026.1",
    }
    machine_info = {
        "node": "benchmark-host",
        "processor": "Test CPU",
        "machine": "AMD64",
        "system": "Windows",
        "release": "11",
        "version": "10.0.26100",
        "python_compiler": "MSC v.1944",
        "python_implementation": "CPython",
        "python_version": "3.12.8",
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-test",
                "name": "NVIDIA Test GPU",
                "driver_version": "600.00",
                "memory_total": 1024.0,
                "memory_used": 128.0,
                "temperature": 45.0,
            }
        ],
    }
    commit_info = {
        "id": "a" * 40,
        "time": datetime(2026, 7, 16, 12, tzinfo=timezone.utc),
        "branch": "main",
        "dirty": False,
    }
    return project_info, machine_info, commit_info


def test_historical_source_override_preserves_time_and_replaces_git_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the historical target as clean main source after overlaying current Python files."""

    historical_time = datetime(2025, 9, 2, 14, 42, 35, tzinfo=timezone.utc)
    report: dict[str, Any] = {
        "commit_info": {
            "id": "overlay-tree",
            "time": historical_time,
            "branch": "detached",
            "dirty": True,
        }
    }
    target = "1234567890abcdef1234567890abcdef12345678"
    monkeypatch.setenv("BENCHVIEW_BENCHMARK_REF", target)
    monkeypatch.setenv("BENCHVIEW_BENCHMARK_BRANCH", "main")

    benchmark_plugin.apply_benchmark_source_override(cast(Any, report))

    assert report["commit_info"] == {
        "id": target,
        "time": historical_time,
        "branch": "main",
        "dirty": False,
    }


@pytest.mark.parametrize("metric_id", ["gpu_time", "cpu_time"])
def test_build_observation_uses_native_identity_and_metric(metric_id: str) -> None:
    """Verify typed dimensions and explicit fixture timing semantics."""

    observation = make_observation(metric_id)

    assert observation["test"] == {
        "id": "slangpy/benchmarks/test_benchmark_tensor.py:test_tensor_sum",
        "name": "test_tensor_sum",
        "source": {
            "file": "slangpy/benchmarks/test_benchmark_tensor.py",
            "function": "test_tensor_sum",
            "line": 42,
        },
    }
    assert observation["case"]["dimensions"] == {
        "device_type": "cuda",
        "element_count": "1024",
        "contiguous": "True",
    }
    assert observation["metrics"][0] == {
        "id": metric_id,
        "name": "GPU time" if metric_id == "gpu_time" else "CPU time",
        "unit": "ms",
        "direction": "lower",
        "distribution": {"samples": [1.25, 1.5, 1.0]},
    }


def test_build_submissions_shares_one_run_key_across_batches() -> None:
    """Every process at one revision must land in the same logical run.

    Batches split by count or by body size, but a split must not fragment the run
    they belong to, and each batch needs its own idempotency key.
    """

    project_info, machine_info, commit_info = submission_context()
    first = make_observation()
    second = make_observation("cpu_time")
    second["test"] = {"id": "tests:second", "name": "second"}
    common = {
        "request_id": "request",
        "execution_id": "execution-a",
        "project_info": project_info,
        "machine_info": machine_info,
        "commit_info": commit_info,
        "project": benchmark_api.SLANGPY_PROJECT,
    }

    by_count = benchmark_api.build_benchview_submissions([first, second], batch_size=1, **common)
    single = len(_json_bytes(benchmark_api.build_benchview_submissions([first], **common)[0]))
    by_size = benchmark_api.build_benchview_submissions(
        [first, second], max_body_bytes=single, **common
    )

    assert len(by_count) == 2 and len(by_size) == 2
    assert by_count[0]["run"]["key"] == by_count[1]["run"]["key"]
    assert by_count[0]["idempotencyKey"] != by_count[1]["idempotencyKey"]
    # An observation that cannot fit on its own has nowhere to go.
    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match="exceeds"):
        benchmark_api.build_benchview_submissions([first], max_body_bytes=single - 1, **common)

    environment = by_count[0]["observations"][0]["environment"]
    assert environment["identity"]["gpus"][0]["memoryBytes"] == 1024 * 1024 * 1024
    assert "temperature" not in environment["identity"]["gpus"][0], "telemetry is not identity"
    assert environment["telemetry"]["gpus"][0]["temperature"] == 45.0


def test_the_ordinary_workflow_benchmarks_only_a_validated_revision() -> None:
    """The job carries the BenchView write key and then runs the revision's own code.

    Checking out the raw input would let a branch or tag move between validation and
    checkout, and validating only the explicit input would leave the dispatch default
    able to benchmark an unmerged commit.
    """

    text = (REPOSITORY_ROOT / ".github/workflows/ci-benchmark.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    # YAML 1.1 reads an unquoted `on` as the boolean true, which is how GitHub spells
    # the trigger block.
    workflow["on"] = workflow.pop(True)
    resolved = "${{ needs.validate-revision.outputs.revision }}"

    build = workflow["jobs"]["build"]
    assert build["needs"] == "validate-revision"
    checkout = next(s for s in build["steps"] if "checkout" in s.get("uses", ""))
    assert checkout["with"]["ref"] == resolved
    assert build["env"]["BENCHVIEW_BENCHMARK_REF"] == resolved

    resolve = next(
        s for s in workflow["jobs"]["validate-revision"]["steps"] if s.get("id") == "resolve"
    )
    assert resolve["env"]["DEFAULT_REVISION"] == "${{ github.sha }}"
    assert 'requested="${REVISION:-$DEFAULT_REVISION}"' in resolve["run"]
    # One resolution and one ancestry check, so neither path can bypass the other.
    assert resolve["run"].count("git rev-parse") == 1
    assert resolve["run"].count("merge-base --is-ancestor") == 1


def test_submission_url_preserves_arbitrary_nested_base() -> None:
    """Keep API traffic inside root and arbitrary-depth nginx mount paths."""

    assert (
        benchmark_api.benchview_submission_url("http://localhost:3000")
        == "http://localhost:3000/api/v1/submissions"
    )
    assert (
        benchmark_api.benchview_submission_url("http://host/foo/bar/hello/")
        == "http://host/foo/bar/hello/api/v1/submissions"
    )
    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match="credentials"):
        benchmark_api.benchview_submission_url("https://user:password@host/benchview")


def test_on_linux_only_the_nvidia_smi_mutation_is_elevated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Root is confined to the one command that needs it, not the whole helper."""

    ci_commands: list[list[str]] = []

    def capture_ci(
        command: list[str],
        shell: bool = False,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        del shell, env
        ci_commands.append(command)

    monkeypatch.setattr(ci, "get_os", lambda: "linux")
    monkeypatch.setattr(ci, "run_command", capture_ci)
    ci.benchmark_python(
        SimpleNamespace(
            device_type="cuda", lock_gpu_clocks=True, api_url=None, run_id="workflow-123"
        )
    )

    # ci.py invokes the helper as an ordinary user, around the benchmark run.
    helper = ["python", str(ci.PROJECT_DIR / "tools/gpu_clock.py")]
    assert ci_commands[0] == helper + ["lock", "--ratio", "0.7"]
    assert ci_commands[-1] == helper + ["unlock"]

    # Inside the helper, only the mutating nvidia-smi call is elevated; the query is not.
    queries: list[list[str]] = []
    monkeypatch.setattr(gpu_clock.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_clock, "NVIDIA_SMI", "nvidia-smi")
    monkeypatch.setattr(
        gpu_clock, "run_command", lambda command: (queries.append(command), "Test GPU")[1]
    )

    assert gpu_clock.nvidia_smi_mutation_command(["-i", "2", "--lock-gpu-clocks=1234"]) == [
        "sudo",
        "-n",
        "--",
        "nvidia-smi",
        "-i",
        "2",
        "--lock-gpu-clocks=1234",
    ]
    assert gpu_clock.get_gpu_name(2) == "Test GPU"
    assert queries[0][0] == "nvidia-smi", "a read-only query must not be elevated"


def test_submit_posts_bearer_authenticated_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the real urllib boundary without contacting a live service."""

    project_info, machine_info, commit_info = submission_context()
    submissions = benchmark_api.build_benchview_submissions(
        [make_observation()],
        request_id="request-1",
        execution_id="execution-1",
        project_info=project_info,
        machine_info=machine_info,
        commit_info=commit_info,
        project=benchmark_api.SLANGPY_PROJECT,
    )
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            201,
            json.dumps({"duplicate": False, "transactionId": "tx", "cursor": "0"}).encode(),
        )

    monkeypatch.setattr(benchmark_api, "urlopen", fake_urlopen)
    receipts = benchmark_api.submit_benchview_submissions(
        "http://host/benchview", "secret-write-key", submissions
    )

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "http://host/benchview/api/v1/submissions"
    assert request.get_header("Authorization") == "Bearer secret-write-key"
    assert request.get_header("Content-type") == "application/json"
    assert isinstance(request.data, bytes)
    assert json.loads(request.data)["idempotencyKey"].startswith("slangpy/")
    assert receipts == [{"duplicate": False, "transactionId": "tx", "cursor": "0"}]


def test_submit_retries_transient_failures_and_gives_up_at_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resets and gateway errors are both retried with the identical body, but bounded."""

    attempts: list[bytes] = []
    delays: list[float] = []
    failures_before_success = 2

    def flaky_urlopen(request: Request, timeout: float) -> FakeResponse:
        """Fail with a reset, then a retryable gateway error, then succeed."""

        assert isinstance(request.data, bytes)
        attempts.append(request.data)
        if len(attempts) == 1 <= failures_before_success:
            raise URLError(ConnectionResetError(104, "Connection reset by peer"))
        if len(attempts) <= failures_before_success:
            raise HTTPError(request.full_url, 503, "busy", hdrs=Message(), fp=BytesIO(b"temporary"))
        return FakeResponse(
            200, json.dumps({"duplicate": True, "transactionId": "tx", "cursor": "0"}).encode()
        )

    monkeypatch.setattr(benchmark_api, "urlopen", flaky_urlopen)
    monkeypatch.setattr(benchmark_api, "sleep", lambda delay: delays.append(delay))

    def submit() -> list[dict[str, Any]]:
        return benchmark_api.submit_benchview_submissions(
            "http://host/benchview",
            "secret-write-key",
            [{"schemaVersion": 1, "idempotencyKey": "stable-key"}],
            max_attempts=3,
            retry_delay_seconds=0.25,
        )

    receipts = submit()

    assert attempts[0] == attempts[1] == attempts[2], "the retried payload must be byte-identical"
    assert delays == [0.25, 0.5], "the delay must back off exponentially"
    assert receipts == [{"duplicate": True, "transactionId": "tx", "cursor": "0"}]

    # Persistent failure must terminate rather than retry a CI submission forever.
    attempts.clear()
    delays.clear()
    failures_before_success = 99
    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match=r"after 3 attempt\(s\)"):
        submit()
    assert len(attempts) == 3


def test_submit_redacts_key_from_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a malicious or reflected server diagnostic from disclosing credentials."""

    key = "never-print-this-key"

    attempts = 0

    def failing_urlopen(request: Request, timeout: float) -> FakeResponse:
        """Return a permanent authentication failure that must not be retried."""

        nonlocal attempts
        attempts += 1
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=BytesIO(f"invalid {key}".encode()),
        )

    monkeypatch.setattr(benchmark_api, "urlopen", failing_urlopen)
    with pytest.raises(benchmark_api.BenchmarkSubmissionError) as error:
        benchmark_api.submit_benchview_submissions(
            "http://host/benchview",
            key,
            [{"schemaVersion": 1}],
        )
    assert key not in str(error.value)
    assert "<redacted>" in str(error.value)
    assert attempts == 1


def test_the_write_key_does_not_follow_a_redirect_to_another_origin() -> None:
    """urllib copies headers onto redirects, which would hand the key to any host."""

    handler = benchmark_api._SameOriginRedirectHandler()
    request = Request(
        "http://benchview.test/benchview/api/v1/submissions",
        data=b"{}",
        method="POST",
        headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
    )

    # urllib refuses 307 and 308 on a POST outright, so the reachable leak is a
    # 301/302/303 that it rewrites into a GET while carrying the headers over.
    elsewhere = handler.redirect_request(
        request, BytesIO(b""), 302, "Found", Message(), "http://attacker.test/collect"
    )
    assert elsewhere is not None
    assert "Authorization" not in elsewhere.headers

    same = handler.redirect_request(
        request, BytesIO(b""), 302, "Found", Message(), "http://benchview.test/moved"
    )
    assert same is not None
    assert same.headers["Authorization"] == "Bearer secret"


def test_custom_project_replaces_every_slangpy_identity() -> None:
    """A reusing project must not leave SlangPy's identity anywhere in the payload.

    BenchView keys history by these, so a single missed field files another
    project's results under SlangPy. The idempotency prefix is the easiest to
    overlook because it is not part of the visible run identity.
    """

    project_info, machine_info, commit_info = submission_context()
    project = benchmark_api.BenchViewProject(
        id="falcor2",
        name="Falcor2",
        suite_id="imagetests",
        suite_name="Image Tests",
        repository="https://example.test/falcor2",
        producer_name="falcor2-pytest-benchmark",
        producer_version="1.0.0",
    )

    submission = benchmark_api.build_benchview_submissions(
        [make_observation()],
        request_id="request",
        execution_id="execution",
        project_info=project_info,
        machine_info=machine_info,
        commit_info=commit_info,
        project=project,
    )[0]

    assert submission["project"] == {"id": "falcor2", "name": "Falcor2"}
    assert submission["run"]["suite"] == {"id": "imagetests", "name": "Image Tests"}
    assert submission["run"]["vcs"]["repository"] == "https://example.test/falcor2"
    assert submission["run"]["key"].startswith("git:" + "a" * 40 + "/suite:imagetests/")
    assert submission["producer"]["name"] == "falcor2-pytest-benchmark"
    assert submission["idempotencyKey"].startswith("falcor2/")
    assert "slangpy" not in _json_bytes(submission).decode().replace(
        "slangpy/benchmarks/test_benchmark_tensor.py", ""
    )


def test_observation_carries_several_metrics_and_extra_metadata() -> None:
    """One case may report several measurements, including a metric with a breakdown."""

    compile_time = benchmark_api.build_metric("compile_time", "Compile time", [30.0])
    compile_time["breakdown"] = {
        "mode": "complete",
        "components": [{"id": "slang", "name": "Slang", "distribution": {"samples": [30.0]}}],
    }

    observation = benchmark_api.build_benchview_observation(
        filename="slangpy/benchmarks/test_benchmark_tensor.py",
        function_name="test_tensor_sum",
        display_name="test_tensor_sum",
        parameters={},
        observed_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
        metrics=[
            benchmark_api.build_metric("gpu_time", "GPU time", [1.0, 2.0]),
            compile_time,
        ],
        metadata={"deviceType": "vulkan"},
    )

    assert [metric["id"] for metric in observation["metrics"]] == ["gpu_time", "compile_time"]
    assert observation["metrics"][1]["breakdown"]["components"][0]["id"] == "slang"
    assert observation["metadata"]["deviceType"] == "vulkan"


def test_source_root_accepts_a_relative_filename_from_any_directory() -> None:
    """A relative filename is relative to the root, not to the working directory.

    Otherwise running pytest from somewhere other than the root would reject an
    in-root path and fail the whole submission.
    """

    observation = benchmark_api.build_benchview_observation(
        filename="slangpy/benchmarks/test_benchmark_tensor.py",
        function_name="test_tensor_sum",
        display_name="test_tensor_sum",
        parameters={},
        observed_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
        metrics=[benchmark_api.build_metric("gpu_time", "GPU time", [1.0])],
        source_root=REPOSITORY_ROOT,
    )

    assert observation["test"]["source"]["file"] == "slangpy/benchmarks/test_benchmark_tensor.py"


def test_source_root_rejects_a_path_outside_the_repository() -> None:
    """An unrelated source path must fail rather than become a machine-specific test ID."""

    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match="outside"):
        benchmark_api.build_benchview_observation(
            filename=str(Path(Path(__file__).resolve().anchor) / "stray_benchmark.py"),
            function_name="test_stray",
            display_name="test_stray",
            parameters={},
            observed_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
            metrics=[benchmark_api.build_metric("gpu_time", "GPU time", [1.0])],
            source_root=Path(__file__).parent,
        )
