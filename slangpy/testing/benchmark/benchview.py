# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

BENCHVIEW_MAX_BODY_BYTES = 8 * 1024 * 1024
BENCHVIEW_DEFAULT_BATCH_SIZE = 100
BENCHVIEW_PROJECT_ID = "slangpy"
BENCHVIEW_REPOSITORY = "https://github.com/shader-slang/slangpy"
BENCHVIEW_SUITE_ID = "python"

BenchViewObservation = dict[str, Any]
BenchViewSubmission = dict[str, Any]


class BenchmarkSubmissionError(RuntimeError):
    """Report a safe BenchView payload or HTTP submission failure."""


def _rfc3339(value: datetime) -> str:
    """Convert a datetime to the UTC spelling emitted by native submissions."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    """Serialize finite JSON deterministically for sizing and producer digests."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _nonempty_string(value: Any) -> Optional[str]:
    """Return a non-empty string representation or omit the value."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_source_path(filename: str) -> str:
    """Prefer a repository-relative slash-separated source path for test identity."""

    source = Path(filename)
    if source.is_absolute():
        try:
            source = source.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
    return source.as_posix().removeprefix("./")


def _normalize_dimension(name: str, value: Any) -> str:
    """Preserve legacy string dimensions while shortening device enum text."""

    normalized = str(value)
    if name == "device_type":
        enum_name = getattr(value, "name", None)
        if normalized.startswith("DeviceType."):
            normalized = normalized.removeprefix("DeviceType.")
        elif isinstance(enum_name, str) and enum_name:
            normalized = enum_name
    return normalized


def build_benchview_observation(
    filename: str,
    function_name: str,
    display_name: str,
    parameters: dict[str, Any],
    samples: list[float],
    observed_at: datetime,
    metric_id: str,
    metric_name: str,
    adapter_name: Optional[str] = None,
    source_line: Optional[int] = None,
) -> BenchViewObservation:
    """Build one passed native BenchView observation from a measured pytest case."""

    if not samples or any(not math.isfinite(sample) for sample in samples):
        raise BenchmarkSubmissionError("Benchmark samples must be a non-empty finite list.")
    source_file = _normalize_source_path(filename)
    dimensions = {name: _normalize_dimension(name, value) for name, value in parameters.items()}
    source: dict[str, Any] = {"file": source_file, "function": function_name}
    if source_line is not None and source_line >= 1:
        source["line"] = source_line
    observation: BenchViewObservation = {
        "test": {
            "id": f"{source_file}:{function_name}",
            "name": function_name,
            "source": source,
        },
        "observedAt": _rfc3339(observed_at),
        "status": "passed",
        "metrics": [
            {
                "id": metric_id,
                "name": metric_name,
                "unit": "ms",
                "direction": "lower",
                "distribution": {"samples": samples},
            }
        ],
    }
    if dimensions:
        observation["case"] = {"dimensions": dimensions}
    if display_name != function_name or adapter_name:
        metadata: dict[str, Any] = {}
        if display_name != function_name:
            metadata["pytestName"] = display_name
        if adapter_name:
            metadata["adapterName"] = adapter_name
        observation["metadata"] = metadata
    return observation


def _build_benchview_environment(machine_info: dict[str, Any]) -> dict[str, Any]:
    """Separate stable machine identity from volatile GPU telemetry."""

    identity: dict[str, Any] = {}
    machine = _nonempty_string(machine_info.get("node"))
    if machine:
        identity["machine"] = machine

    os_identity: dict[str, Any] = {}
    for target, source in (
        ("name", "system"),
        ("version", "version"),
        ("architecture", "machine"),
    ):
        value = _nonempty_string(machine_info.get(source))
        if value:
            os_identity[target] = value
    if os_identity:
        identity["os"] = os_identity

    cpu: dict[str, Any] = {}
    processor = _nonempty_string(machine_info.get("processor"))
    architecture = _nonempty_string(machine_info.get("machine"))
    if processor:
        cpu["model"] = processor
    if architecture:
        cpu["architecture"] = architecture
    logical_cores = os.cpu_count()
    if logical_cores is not None and logical_cores > 0:
        cpu["logicalCores"] = logical_cores
    if cpu:
        identity["cpu"] = cpu

    gpu_values = machine_info.get("gpus")
    if not isinstance(gpu_values, list):
        gpu_values = []
    gpu_identities: list[dict[str, Any]] = []
    gpu_telemetry: list[dict[str, Any]] = []
    for fallback_index, gpu_value in enumerate(gpu_values):
        if not isinstance(gpu_value, dict):
            continue
        name = _nonempty_string(gpu_value.get("name"))
        if not name:
            continue
        index_value = gpu_value.get("index", fallback_index)
        index = index_value if isinstance(index_value, int) and index_value >= 0 else fallback_index
        gpu: dict[str, Any] = {"index": index, "name": name}
        uuid = _nonempty_string(gpu_value.get("uuid"))
        driver_version = _nonempty_string(gpu_value.get("driver_version"))
        if uuid:
            gpu["uuid"] = uuid
        if driver_version:
            gpu["driverVersion"] = driver_version
        memory_total = gpu_value.get("memory_total")
        if isinstance(memory_total, (int, float)) and math.isfinite(memory_total):
            gpu["memoryBytes"] = max(0, round(memory_total * 1024 * 1024))
        gpu_identities.append(gpu)

        telemetry: dict[str, Any] = {"index": index}
        for field in (
            "utilization",
            "memory_used",
            "clock_current_graphics",
            "clock_current_memory",
            "clock_max_graphics",
            "clock_max_memory",
            "temperature",
        ):
            value = gpu_value.get(field)
            if isinstance(value, (str, bool, int)) or (
                isinstance(value, float) and math.isfinite(value)
            ):
                telemetry[field] = value
        if len(telemetry) > 1:
            gpu_telemetry.append(telemetry)
    if gpu_identities:
        identity["gpus"] = gpu_identities

    attributes: dict[str, Any] = {}
    for target, source in (
        ("pythonImplementation", "python_implementation"),
        ("pythonVersion", "python_version"),
        ("pythonCompiler", "python_compiler"),
        ("osRelease", "release"),
    ):
        value = _nonempty_string(machine_info.get(source))
        if value:
            attributes[target] = value
    for index, gpu_value in enumerate(gpu_values):
        if isinstance(gpu_value, dict):
            serial = _nonempty_string(gpu_value.get("serial_number"))
            if serial:
                attributes[f"gpu{index}SerialNumber"] = serial
    if attributes:
        identity["attributes"] = attributes

    environment: dict[str, Any] = {}
    if identity:
        environment["identity"] = identity
    if gpu_telemetry:
        environment["telemetry"] = {"gpus": gpu_telemetry}
    return environment


def _build_submission_base(
    request_id: str,
    execution_id: str,
    project_info: dict[str, Any],
    commit_info: dict[str, Any],
) -> BenchViewSubmission:
    """Build fields shared by every batch from one pytest benchmark process."""

    revision = _nonempty_string(commit_info.get("id")) or "unknown"
    dimensions: dict[str, Any] = {}
    project_version = _nonempty_string(project_info.get("version"))
    slang_build_tag = _nonempty_string(project_info.get("slang_build_tag"))
    if project_version:
        dimensions["projectVersion"] = project_version
    if slang_build_tag:
        dimensions["slangBuildTag"] = slang_build_tag
    config_digest = hashlib.sha256(_json_bytes(dimensions)).hexdigest()[:16]

    vcs: dict[str, Any] = {"repository": BENCHVIEW_REPOSITORY, "revision": revision}
    commit_time = commit_info.get("time") or commit_info.get("author_time")
    if isinstance(commit_time, datetime):
        vcs["commitTime"] = _rfc3339(commit_time)
    elif _nonempty_string(commit_time):
        vcs["commitTime"] = str(commit_time)
    branch = _nonempty_string(commit_info.get("branch"))
    if branch:
        vcs["branch"] = branch
    if isinstance(commit_info.get("dirty"), bool):
        vcs["dirty"] = commit_info["dirty"]

    run: dict[str, Any] = {
        "key": f"git:{revision}/suite:{BENCHVIEW_SUITE_ID}/config:{config_digest}",
        "suite": {"id": BENCHVIEW_SUITE_ID, "name": "Python benchmarks"},
        "vcs": vcs,
    }
    if dimensions:
        run["dimensions"] = dimensions
    return {
        "schemaVersion": 1,
        "project": {"id": BENCHVIEW_PROJECT_ID, "name": "SlangPy"},
        "producer": {"name": "slangpy-benchmark-plugin", "version": "1.0.0"},
        "run": run,
        "execution": {"id": execution_id, "requestId": request_id},
    }


def _finalize_submission(
    base: BenchViewSubmission, observations: list[BenchViewObservation]
) -> BenchViewSubmission:
    """Add deterministic idempotency to one complete native request body."""

    content = {**base, "observations": observations}
    digest = hashlib.sha256(_json_bytes(content)).hexdigest()
    return {"schemaVersion": 1, "idempotencyKey": f"slangpy/{digest}", **content}


def build_benchview_submissions(
    observations: list[BenchViewObservation],
    request_id: str,
    execution_id: str,
    project_info: dict[str, Any],
    machine_info: dict[str, Any],
    commit_info: dict[str, Any],
    batch_size: int = BENCHVIEW_DEFAULT_BATCH_SIZE,
    max_body_bytes: int = BENCHVIEW_MAX_BODY_BYTES,
) -> list[BenchViewSubmission]:
    """Attach shared run/environment facts and greedily form valid API-sized batches."""

    if not request_id or not execution_id:
        raise BenchmarkSubmissionError("Benchmark request and execution IDs must be non-empty.")
    if batch_size < 1 or batch_size > 500:
        raise BenchmarkSubmissionError("BenchView batch size must be between 1 and 500.")
    if max_body_bytes < 1:
        raise BenchmarkSubmissionError("BenchView body limit must be positive.")

    base = _build_submission_base(request_id, execution_id, project_info, commit_info)
    environment = _build_benchview_environment(machine_info)
    prepared: list[BenchViewObservation] = []
    for observation in observations:
        item = dict(observation)
        if environment:
            item["environment"] = environment
        prepared.append(item)

    submissions: list[BenchViewSubmission] = []
    batch: list[BenchViewObservation] = []
    for observation in prepared:
        candidate = [*batch, observation]
        candidate_submission = _finalize_submission(base, candidate)
        if (
            len(candidate) <= batch_size
            and len(_json_bytes(candidate_submission)) <= max_body_bytes
        ):
            batch = candidate
            continue
        if not batch:
            raise BenchmarkSubmissionError("One benchmark observation exceeds the API body limit.")
        submissions.append(_finalize_submission(base, batch))
        batch = [observation]
        if len(_json_bytes(_finalize_submission(base, batch))) > max_body_bytes:
            raise BenchmarkSubmissionError("One benchmark observation exceeds the API body limit.")
    if batch:
        submissions.append(_finalize_submission(base, batch))
    return submissions


def benchview_submission_url(api_base_url: str) -> str:
    """Resolve the submission endpoint without discarding a nested deployment path."""

    try:
        parsed = urlsplit(api_base_url)
    except ValueError as error:
        raise BenchmarkSubmissionError(
            "BenchView API base URL must be an absolute HTTP URL."
        ) from error
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise BenchmarkSubmissionError("BenchView API base URL must be an absolute HTTP URL.")
    if parsed.username is not None or parsed.password is not None:
        raise BenchmarkSubmissionError("BenchView API base URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise BenchmarkSubmissionError(
            "BenchView API base URL must not contain query or fragment text."
        )
    return api_base_url.rstrip("/") + "/api/v1/submissions"


def _safe_response_text(data: bytes, write_key: str) -> str:
    """Decode a bounded server diagnostic while redacting the configured credential."""

    text = data[:2048].decode("utf-8", errors="replace")
    return text.replace(write_key, "<redacted>")


def submit_benchview_submissions(
    api_base_url: str,
    write_key: str,
    submissions: list[BenchViewSubmission],
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """POST immutable batches with Bearer authentication and return validated receipts."""

    if not write_key:
        raise BenchmarkSubmissionError("BenchView API write key must be non-empty.")
    endpoint = benchview_submission_url(api_base_url)
    receipts: list[dict[str, Any]] = []
    for submission in submissions:
        body = _json_bytes(submission)
        request = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {write_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = response.getcode()
                response_body = response.read(BENCHVIEW_MAX_BODY_BYTES + 1)
        except HTTPError as error:
            diagnostic = _safe_response_text(error.read(2049), write_key)
            raise BenchmarkSubmissionError(
                f"BenchView rejected a benchmark submission with HTTP {error.code}: {diagnostic}"
            ) from error
        except URLError as error:
            raise BenchmarkSubmissionError(
                f"Could not reach the BenchView submission endpoint: {error.reason}"
            ) from error
        if status < 200 or status >= 300:
            diagnostic = _safe_response_text(response_body, write_key)
            raise BenchmarkSubmissionError(
                f"BenchView rejected a benchmark submission with HTTP {status}: {diagnostic}"
            )
        try:
            receipt = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BenchmarkSubmissionError("BenchView returned an invalid JSON receipt.") from error
        if not isinstance(receipt, dict) or not isinstance(receipt.get("duplicate"), bool):
            raise BenchmarkSubmissionError("BenchView returned an invalid submission receipt.")
        receipts.append(receipt)
    return receipts
