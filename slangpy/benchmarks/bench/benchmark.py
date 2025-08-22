# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import slangpy as spy
from slangpy.core.function import FunctionNodeBwds
import numpy as np
from typing import Any, Union
from time import time

from bench.report import BenchmarkReport


class BenchmarkFixture:

    def __init__(self, config: pytest.Config, node: Any):
        super().__init__()
        self.config = config
        self.node = node

    def __call__(
        self,
        device: spy.Device,
        function: Union[spy.Function, FunctionNodeBwds],
        iterations: int = 200,
        warmup_iterations: int = 10,
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

        report: BenchmarkReport = {
            "name": self.node.name,
            "cpu_time": cpu_time,
            "data": [float(d) for d in deltas],
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "stddev": float(np.std(deltas)),
        }

        self.config._benchmark_reports.append(report)  # type: ignore


@pytest.fixture
def benchmark(request: pytest.FixtureRequest, pytestconfig: pytest.Config):
    yield BenchmarkFixture(pytestconfig, request.node)
