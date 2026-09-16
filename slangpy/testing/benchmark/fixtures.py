# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy
from slangpy.core.function import FunctionNodeBwds
import numpy as np
from typing import Any, Callable, Optional, Union
from time import time, sleep
from datetime import datetime, timezone

from .benchview import build_benchview_observation
from .report import BenchmarkReport

DEFAULT_ITERATIONS = 2000
INITIAL_WARMUP_ITERATIONS = 100


class ReportFixture:

    def __init__(self, config: pytest.Config, node: Any):
        super().__init__()
        self.config = config
        self.node = node

    def __call__(
        self,
        device: Optional[spy.Device],
        data: list[float],
        cpu_time: float,
        metric_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Generate and store a benchmark report with the given data."""

        params = (
            {k: str(v) for k, v in self.node.callspec.params.items()}
            if hasattr(self.node, "callspec") and self.node.callspec
            else {}
        )

        meta = {}

        if device:
            meta["adapter_name"] = device.info.adapter_name

        # Calculate trimmed mean (remove 10% outliers - 5% from each end)
        sorted_data = np.sort(data)
        trim_count = int(len(sorted_data) * 0.05)  # 5% from each end
        if trim_count > 0:
            trimmed_data = sorted_data[trim_count:-trim_count]
        else:
            trimmed_data = sorted_data
        trimmed_mean = float(np.mean(trimmed_data))

        observed_at = datetime.now(timezone.utc)
        samples = [float(d) for d in data]
        filename = str(self.node.location[0]).replace("\\", "/")
        function_name = self.node.originalname
        if metric_id is None:
            metric_id = "cpu_time" if "_cpu" in function_name else "gpu_time"
        if metric_name is None:
            metric_name = "CPU time" if metric_id == "cpu_time" else "GPU time"
        report: BenchmarkReport = {
            "name": self.node.name,
            "filename": filename,
            "function": function_name,
            "params": params,
            "meta": meta,
            "timestamp": observed_at,
            "cpu_time": cpu_time,
            "data": samples,
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": trimmed_mean,
            "median": float(np.median(data)),
            "stddev": float(np.std(data)),
        }

        self.config._benchmark_context["benchmark_reports"].append(report)  # type: ignore
        self.config._benchmark_context["benchmark_observations"].append(  # type: ignore
            build_benchview_observation(
                filename=filename,
                function_name=function_name,
                display_name=self.node.name,
                parameters=params,
                samples=samples,
                observed_at=observed_at,
                metric_id=metric_id,
                metric_name=metric_name,
                adapter_name=meta.get("adapter_name"),
                source_line=int(self.node.location[1]) + 1,
            )
        )


class BenchmarkSlangFunction:

    def __init__(self, report_fixture: ReportFixture):
        super().__init__()
        self.report_fixture = report_fixture

    def __call__(
        self,
        device: spy.Device,
        function: Union[spy.Function, FunctionNodeBwds],
        iterations: int = DEFAULT_ITERATIONS,
        warmup_iterations: int = INITIAL_WARMUP_ITERATIONS,
        **kwargs: Any,
    ) -> None:
        """Run the benchmark with the given parameters."""

        start_time = time()

        for _ in range(warmup_iterations):
            function(**kwargs)

        query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=iterations * 2)
        for i in range(iterations):
            command_encoder = device.create_command_encoder()
            command_encoder.write_timestamp(query_pool, i * 2)
            function(**kwargs, _append_to=command_encoder)
            command_encoder.write_timestamp(query_pool, i * 2 + 1)
            device.submit_command_buffer(command_encoder.finish())
        device.wait()
        queries = np.array(query_pool.get_results(0, iterations * 2))
        frequency = float(device.info.timestamp_frequency)
        deltas = (queries[1::2] - queries[0::2]) / frequency * 1000.0

        end_time = time()
        cpu_time = end_time - start_time

        # Use the report fixture to generate and store the report
        self.report_fixture(
            device,
            deltas,
            cpu_time,
            metric_id="gpu_time",
            metric_name="GPU time",
            **kwargs,
        )


class BenchmarkComputeKernel:

    def __init__(self, report_fixture: ReportFixture):
        super().__init__()
        self.report_fixture = report_fixture

    def __call__(
        self,
        device: spy.Device,
        kernel: spy.ComputeKernel,
        thread_count: spy.uint3,
        iterations: int = DEFAULT_ITERATIONS,
        warmup_iterations: int = INITIAL_WARMUP_ITERATIONS,
        **kwargs: Any,
    ) -> None:
        """Run the benchmark with the given parameters."""

        start_time = time()

        for _ in range(warmup_iterations):
            kernel.dispatch(thread_count, **kwargs)

        query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=iterations * 2)
        for i in range(iterations):
            command_encoder = device.create_command_encoder()
            command_encoder.write_timestamp(query_pool, i * 2)
            kernel.dispatch(thread_count, command_encoder=command_encoder, **kwargs)
            command_encoder.write_timestamp(query_pool, i * 2 + 1)
            device.submit_command_buffer(command_encoder.finish())
        device.wait()
        queries = np.array(query_pool.get_results(0, iterations * 2))
        frequency = float(device.info.timestamp_frequency)
        deltas = (queries[1::2] - queries[0::2]) / frequency * 1000.0

        end_time = time()
        cpu_time = end_time - start_time

        # Use the report fixture to generate and store the report
        self.report_fixture(
            device,
            deltas,
            cpu_time,
            metric_id="gpu_time",
            metric_name="GPU time",
            **kwargs,
        )


class BenchmarkPythonFunction:

    def __init__(self, report_fixture: ReportFixture):
        super().__init__()
        self.report_fixture = report_fixture

    def __call__(
        self,
        device: Optional[spy.Device],
        function: Callable[..., None],
        iterations: int = 10,
        sub_iterations: int = DEFAULT_ITERATIONS // 10,
        warmup_iterations: int = INITIAL_WARMUP_ITERATIONS,
        sleeps: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run the benchmark with the given parameters."""

        start_time = time()

        for _ in range(warmup_iterations):
            function(**kwargs)

        if sleeps:
            sleep(1)
        deltas = []

        for _ in range(iterations):
            main_start_time = time()
            for _ in range(sub_iterations):
                function(**kwargs)
            main_end_time = time()
            deltas.append(1000 * (main_end_time - main_start_time) / sub_iterations)
        if sleeps:
            sleep(1)

        end_time = time()

        cpu_time = end_time - start_time

        # Use the report fixture to generate and store the report
        self.report_fixture(device, deltas, cpu_time, **kwargs)


@pytest.fixture
def report(request: pytest.FixtureRequest, pytestconfig: pytest.Config):
    yield ReportFixture(pytestconfig, request.node)


@pytest.fixture
def benchmark_slang_function(report: ReportFixture):
    yield BenchmarkSlangFunction(report)


@pytest.fixture
def benchmark_compute_kernel(report: ReportFixture):
    yield BenchmarkComputeKernel(report)


@pytest.fixture
def benchmark_python_function(report: ReportFixture):
    yield BenchmarkPythonFunction(report)
