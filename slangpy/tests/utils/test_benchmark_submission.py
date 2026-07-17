# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, Optional, Type
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from slangpy.testing.benchmark.fixtures import ReportFixture

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


def test_report_fixture_accumulates_local_and_native_results() -> None:
    """Keep local comparisons while fixtures build the API-ready observation."""

    context: dict[str, Any] = {"benchmark_reports": [], "benchmark_observations": []}
    config = SimpleNamespace(_benchmark_context=context)
    node = SimpleNamespace(
        name="test_cpu_case[cuda]",
        originalname="test_cpu_case",
        location=("slangpy/benchmarks/test_cpu.py", 9, "test_cpu_case"),
        callspec=SimpleNamespace(params={"device_type": FakeDeviceType()}),
    )

    ReportFixture(cast(pytest.Config, config), node)(
        None,
        [2.0, 1.0, 3.0],
        0.25,
    )

    assert len(context["benchmark_reports"]) == 1
    assert context["benchmark_reports"][0]["data"] == [2.0, 1.0, 3.0]
    assert len(context["benchmark_observations"]) == 1
    assert context["benchmark_observations"][0]["metrics"][0]["id"] == "cpu_time"


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


def test_plugin_rejects_missing_api_configuration_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before an expensive benchmark session when URL or key configuration is absent."""

    monkeypatch.delenv("BENCHVIEW_API_URL", raising=False)
    monkeypatch.setenv("BENCHVIEW_API_KEY", "key")
    with pytest.raises(pytest.UsageError, match="benchmark-api-url"):
        benchmark_plugin.pytest_configure(cast(pytest.Config, FakePytestConfig("request", None)))

    monkeypatch.delenv("BENCHVIEW_API_KEY", raising=False)
    with pytest.raises(pytest.UsageError, match="BENCHVIEW_API_KEY"):
        benchmark_plugin.pytest_configure(
            cast(pytest.Config, FakePytestConfig("request", "http://localhost:3000"))
        )

    monkeypatch.setenv("BENCHVIEW_API_KEY", "key")
    with pytest.raises(pytest.UsageError, match="absolute HTTP URL"):
        benchmark_plugin.pytest_configure(
            cast(pytest.Config, FakePytestConfig("request", "not-a-url"))
        )


def test_ci_wrapper_passes_benchview_options_to_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the ordinary CI entry point as the single benchmark runner."""

    commands: list[list[str]] = []

    def capture_command(
        command: list[str],
        shell: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """Capture the generated command without starting benchmark subprocesses."""

        assert shell is True
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
        "pytest",
        "slangpy/benchmarks",
        "-ra",
        "--device-types",
        "cuda",
        f"--basetemp={ci.PYTEST_BASE_TEMP_DIR}",
    ]
    assert commands[0][-4:] == [
        "--benchmark-submit",
        "workflow-123",
        "--benchmark-api-url",
        "http://host/benchview",
    ]


def test_ci_wrapper_forwards_an_explicit_empty_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let pytest reject a missing workflow variable instead of skipping submission."""

    commands: list[list[str]] = []

    def capture_command(
        command: list[str],
        shell: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        """Capture the command generated for an explicitly configured API URL."""

        commands.append(command)

    monkeypatch.setattr(ci, "get_os", lambda: "linux")
    monkeypatch.setattr(ci, "run_command", capture_command)
    ci.benchmark_python(
        SimpleNamespace(
            device_type="cuda",
            lock_gpu_clocks=False,
            api_url="",
            run_id="workflow-123",
        )
    )

    assert commands[0][-4:] == [
        "--benchmark-submit",
        "workflow-123",
        "--benchmark-api-url",
        "",
    ]


def test_linux_ci_wrapper_does_not_run_gpu_clock_python_as_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elevate only nvidia-smi mutations inside the clock helper."""

    commands: list[list[str]] = []

    def capture_command(
        command: list[str],
        shell: bool = True,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        del shell, env
        commands.append(command)

    monkeypatch.setattr(ci, "get_os", lambda: "linux")
    monkeypatch.setattr(ci, "run_command", capture_command)
    ci.benchmark_python(
        SimpleNamespace(
            device_type="cuda",
            lock_gpu_clocks=True,
            api_url=None,
            run_id="workflow-123",
        )
    )

    assert commands[0][:2] == ["python", str(ci.PROJECT_DIR / "tools/gpu_clock.py")]
    assert commands[0][2:] == ["lock", "--ratio", "0.7"]
    assert commands[-1][:2] == ["python", str(ci.PROJECT_DIR / "tools/gpu_clock.py")]
    assert commands[-1][2:] == ["unlock"]
    assert all(command[0] != "sudo" for command in (commands[0], commands[-1]))


def test_linux_gpu_clock_elevates_only_nvidia_smi_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def capture_command(command: list[str]) -> str:
        commands.append(command)
        return "Test GPU"

    monkeypatch.setattr(gpu_clock.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu_clock, "NVIDIA_SMI", "nvidia-smi")
    monkeypatch.setattr(gpu_clock, "run_command", capture_command)

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
    assert commands == [
        [
            "nvidia-smi",
            "-i",
            "2",
            "--query-gpu=name",
            "--format=csv,noheader,nounits",
        ]
    ]


def test_ordinary_workflow_uses_ci_wrapper_without_historical_logic() -> None:
    """Keep scheduled and manual benchmarks on the normal current-revision path."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/ci-benchmark.yml").read_text(encoding="utf-8")

    assert "cron: '0 */4 * * *'" in workflow
    assert "workflow_dispatch:" in workflow
    assert workflow.count("python tools/ci.py benchmark-python") == 2
    assert workflow.count("--lock-gpu-clocks") == 2
    assert "Benchmark (Python, Linux, GPU Clock Locked)" in workflow
    assert "BENCHVIEW_API_URL" in workflow
    assert "BENCHVIEW_API_KEY" in workflow
    assert workflow.count("contains(matrix.flags, 'benchmark')") >= 4
    assert "contains(matrix.flags, 'unit-test')" not in workflow
    assert "python tools/ci.py install-slangpy-torch" in workflow
    assert "python -m pip uninstall slangpy-torch -y" in workflow
    assert "benchmark_ref" not in workflow
    assert "Overlay current BenchView benchmark harness" not in workflow
    assert "run_benchmark_ci.py" not in workflow
    assert "mongodb" not in workflow.lower()


def test_cuda_only_ppisp_benchmarks_declare_the_device_dimension() -> None:
    """Keep CUDA-only PPISP tests out of the non-device benchmark shard."""

    source_path = REPOSITORY_ROOT / "slangpy/benchmarks/test_benchmark_ppisp.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    device_tests: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get_torch_device"
            for child in ast.walk(node)
        ):
            device_tests.append(node)

    assert device_tests
    for test_function in device_tests:
        parameter_names = [parameter.arg for parameter in test_function.args.args]
        assert "device_type" in parameter_names, test_function.name
        get_device_calls = [
            child
            for child in ast.walk(test_function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get_torch_device"
        ]
        assert all(
            call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "device_type"
            for call in get_device_calls
        ), test_function.name


def test_backward_diff_benchmark_uses_the_available_extensions_include() -> None:
    """Keep extensions.slang and the benchmark include directory aligned."""

    benchmark_directory = REPOSITORY_ROOT / "slangpy/benchmarks"
    benchmark_source = (benchmark_directory / "test_benchmark_bwd_diff.py").read_text(
        encoding="utf-8"
    )

    assert (benchmark_directory / "ppisp/extensions.slang").is_file()
    assert 'os.path.join(BENCH_DIR, "ppisp")' in benchmark_source


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


def test_submit_redacts_key_from_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a malicious or reflected server diagnostic from disclosing credentials."""

    key = "never-print-this-key"

    def failing_urlopen(request: Request, timeout: float) -> FakeResponse:
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
