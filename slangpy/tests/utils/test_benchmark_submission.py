# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
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

benchmark_api = import_module("slangpy.testing.benchmark.benchview")
benchmark_plugin = import_module("slangpy.testing.benchmark.plugin")
ci = import_module("tools.ci")
gpu_clock = import_module("tools.gpu_clock")

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
        samples=[1.25, 1.5, 1.0],
        observed_at=datetime(2026, 7, 16, 12, 30, tzinfo=timezone.utc),
        metric_id=metric_id,
        metric_name="GPU time" if metric_id == "gpu_time" else "CPU time",
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
    target = "f3ad0fd91d8cf4eeb2be3b505765b43482aa952a"
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


def test_build_submissions_shares_run_and_separates_environment_telemetry() -> None:
    """Prove distributed run identity, batching, and stable/volatile environment mapping."""

    project_info, machine_info, commit_info = submission_context()
    first = make_observation()
    second = make_observation("cpu_time")
    second["test"] = {
        "id": "slangpy/benchmarks/test_benchmark_tensor.py:test_tensor_sum_cpu",
        "name": "test_tensor_sum_cpu",
    }

    submissions = benchmark_api.build_benchview_submissions(
        [first, second],
        request_id="gitlab-pipeline-123",
        execution_id="execution-a",
        project_info=project_info,
        machine_info=machine_info,
        commit_info=commit_info,
        batch_size=1,
    )
    repeated = benchmark_api.build_benchview_submissions(
        [first, second],
        request_id="gitlab-pipeline-123",
        execution_id="execution-a",
        project_info=project_info,
        machine_info=machine_info,
        commit_info=commit_info,
        batch_size=1,
    )
    other_execution = benchmark_api.build_benchview_submissions(
        [first],
        request_id="gitlab-pipeline-123",
        execution_id="execution-b",
        project_info=project_info,
        machine_info=machine_info,
        commit_info=commit_info,
    )

    assert submissions == repeated
    assert len(submissions) == 2
    assert submissions[0]["run"]["key"] == submissions[1]["run"]["key"]
    assert submissions[0]["run"]["key"] == other_execution[0]["run"]["key"]
    assert submissions[0]["idempotencyKey"] != submissions[1]["idempotencyKey"]
    assert submissions[0]["idempotencyKey"] != other_execution[0]["idempotencyKey"]
    environment = submissions[0]["observations"][0]["environment"]
    assert environment["identity"]["machine"] == "benchmark-host"
    assert environment["identity"]["gpus"][0]["memoryBytes"] == 1024 * 1024 * 1024
    assert "temperature" not in environment["identity"]["gpus"][0]
    assert environment["telemetry"]["gpus"][0]["temperature"] == 45.0


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


def test_build_submissions_splits_at_the_body_limit() -> None:
    """Split large sessions without allowing one oversize observation through."""

    project_info, machine_info, commit_info = submission_context()
    first = make_observation()
    second = make_observation("cpu_time")
    second["test"] = {"id": "tests:second", "name": "second"}
    common = {
        "request_id": "request",
        "execution_id": "execution",
        "project_info": project_info,
        "machine_info": machine_info,
        "commit_info": commit_info,
    }
    single_sizes = [
        len(
            json.dumps(
                benchmark_api.build_benchview_submissions([observation], **common)[0],
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        for observation in (first, second)
    ]
    body_limit = max(single_sizes)

    submissions = benchmark_api.build_benchview_submissions(
        [first, second], max_body_bytes=body_limit, **common
    )
    assert len(submissions) == 2
    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match="exceeds"):
        benchmark_api.build_benchview_submissions(
            [first], max_body_bytes=single_sizes[0] - 1, **common
        )


def test_ci_wrapper_passes_benchview_options_to_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the ordinary CI entry point as the single benchmark runner."""

    commands: list[list[str]] = []

    def capture_command(
        command: list[str],
        shell: bool = False,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """Capture the generated command without starting benchmark subprocesses."""

        assert shell is False
        assert env is not None
        commands.append(command)

    monkeypatch.setattr(ci, "get_os", lambda: "linux")
    monkeypatch.setattr(ci, "run_command", capture_command)
    ci.benchmark_python(
        SimpleNamespace(
            device_type="cuda",
            lock_gpu_clocks=False,
            api_url="http://host/benchview",
            run_id="workflow-123",
        )
    )

    assert len(commands) == 1
    assert commands[0][:6] == [
        sys.executable,
        "-m",
        "pytest",
        "slangpy/benchmarks",
        "-ra",
        "--device-types",
    ]
    assert commands[0][-4:] == [
        "--benchmark-submit",
        "workflow-123",
        "--benchmark-api-url",
        "http://host/benchview",
    ]


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


def test_submit_retries_connection_resets_and_gives_up_after_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reset is ambiguous, so the identical idempotent batch is repeated, but bounded."""

    attempts: list[bytes] = []
    delays: list[float] = []
    resets_before_success = 2

    def flaky_urlopen(request: Request, timeout: float) -> FakeResponse:
        """Reset connections until ``resets_before_success`` have been consumed."""

        assert timeout == 30.0
        assert isinstance(request.data, bytes)
        attempts.append(request.data)
        if len(attempts) <= resets_before_success:
            raise URLError(ConnectionResetError(104, "Connection reset by peer"))
        return FakeResponse(
            200,
            json.dumps({"duplicate": True, "transactionId": "tx", "cursor": "0"}).encode(),
        )

    monkeypatch.setattr(benchmark_api, "urlopen", flaky_urlopen)
    monkeypatch.setattr(benchmark_api, "sleep", lambda delay: delays.append(delay))
    submit = lambda: benchmark_api.submit_benchview_submissions(
        "http://host/benchview",
        "secret-write-key",
        [{"schemaVersion": 1, "idempotencyKey": "stable-key"}],
        max_attempts=3,
        retry_delay_seconds=0.25,
    )

    receipts = submit()

    assert len(attempts) == 3
    assert attempts[0] == attempts[1] == attempts[2], "the retried payload must be byte-identical"
    assert delays == [0.25, 0.5], "the delay must back off exponentially"
    assert receipts == [{"duplicate": True, "transactionId": "tx", "cursor": "0"}]

    # Persistent failure must terminate rather than retry a CI submission forever.
    attempts.clear()
    delays.clear()
    resets_before_success = 99
    with pytest.raises(benchmark_api.BenchmarkSubmissionError, match=r"after 3 attempt\(s\)"):
        submit()
    assert len(attempts) == 3


def test_submit_retries_a_transient_gateway_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry temporary proxy failures while leaving permanent HTTP failures terminal."""

    attempts = 0
    delays: list[float] = []

    def flaky_urlopen(request: Request, timeout: float) -> FakeResponse:
        """Return one retryable gateway error followed by a normal receipt."""

        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(
                request.full_url,
                503,
                "Service unavailable",
                hdrs=Message(),
                fp=BytesIO(b"temporary"),
            )
        return FakeResponse(
            201,
            json.dumps({"duplicate": False, "transactionId": "tx", "cursor": "0"}).encode(),
        )

    monkeypatch.setattr(benchmark_api, "urlopen", flaky_urlopen)
    monkeypatch.setattr(
        benchmark_api,
        "sleep",
        lambda delay: delays.append(delay),
    )
    receipts = benchmark_api.submit_benchview_submissions(
        "http://host/benchview",
        "secret-write-key",
        [{"schemaVersion": 1, "idempotencyKey": "stable-key"}],
        retry_delay_seconds=0.5,
    )

    assert attempts == 2
    assert delays == [0.5]
    assert receipts == [{"duplicate": False, "transactionId": "tx", "cursor": "0"}]


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
