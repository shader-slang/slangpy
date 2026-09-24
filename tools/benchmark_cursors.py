# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Measure CPU cursor writes and warmed functional command recording.

Build Release first. Run from the repository root with, for example:
python -m tools.benchmark_cursors --device d3d12 --output build/cursors-before.json
GPU submission and synchronization are outside the timed regions.
Use --functional-only in fresh processes to assess cached calls independently.
Compare multiple process pairs with matched PYTHONHASHSEED values; rounds within
one process do not capture variation between processes.
"""

import argparse
import gc
import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Callable

import numpy as np
import slangpy as spy


def measure(action: Callable[[], None], iterations: int, rounds: int) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds + 2):
        gc.collect()
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            start = perf_counter_ns()
            for _ in range(iterations):
                action()
            samples.append((perf_counter_ns() - start) / iterations)
        finally:
            if gc_enabled:
                gc.enable()
    return samples[2:]


def benchmark_cursors(device: spy.Device, rounds: int) -> dict[str, list[float]]:
    module = device.load_module_from_source(
        "benchmark_cursors",
        """
struct Values { float scalar; float4 vector; float2x2 matrix; };
uniform Values values;
StructuredBuffer<Values> buffer;
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {}
""",
    )
    program = device.link_program([module], [module.entry_point("main")])
    owner = device.create_root_shader_object(program)
    root = spy.ShaderCursor(owner)
    cursor = root["values"]
    vector = spy.float4(1, 2, 3, 4)
    sequence_vector = [1.0, 2.0, 3.0, 4.0]
    numpy_vector = np.arange(4, dtype=np.float32)
    numpy_matrix = np.arange(4, dtype=np.float32).reshape(2, 2)
    fields = {"scalar": 1.0, "vector": vector}
    scalar_cursor = cursor.scalar
    samples = {
        "cursor_construction": measure(lambda: spy.ShaderCursor(owner), 20000, rounds),
        "scalar": measure(lambda: cursor.scalar.write(1.0), 20000, rounds),
        "scalar_cached_cursor": measure(lambda: scalar_cursor.write(1.0), 20000, rounds),
        "scalar_assignment": measure(lambda: setattr(cursor, "scalar", 1.0), 20000, rounds),
        "nested_assignment": measure(
            lambda: root["values"].__setitem__("scalar", 1.0), 20000, rounds
        ),
        "nested_attribute_assignment": measure(
            lambda: setattr(root.values, "scalar", 1.0), 20000, rounds
        ),
        "vector": measure(lambda: cursor.vector.write(vector), 20000, rounds),
        "sequence_vector": measure(lambda: cursor.vector.write(sequence_vector), 20000, rounds),
        "numpy_vector": measure(lambda: cursor.vector.write(numpy_vector), 10000, rounds),
        "numpy_matrix": measure(lambda: cursor.matrix.write(numpy_matrix), 10000, rounds),
        "dict_update": measure(lambda: cursor.write(fields), 10000, rounds),
    }
    layout = module.layout.get_type_layout(module.layout.find_type_by_name("Values"))
    bulk = spy.BufferCursor(device.info.type, layout, 1024)
    data = {"vector": np.arange(4096, dtype=np.float32).reshape(1024, 4)}
    samples["bulk_checked_1024"] = measure(
        lambda: bulk.write_from_numpy(data, unchecked_copy=False), 100, rounds
    )
    samples["bulk_raw_1024"] = measure(lambda: bulk.write_from_numpy(data), 100, rounds)

    if device.has_feature(spy.Feature.parameter_block):
        blocks = device.load_module_from_source(
            "benchmark_cursor_blocks",
            """
struct Inner { float value; };
struct Outer { ParameterBlock<Inner> inner; };
ParameterBlock<Outer> outer;
[shader("compute")]
[numthreads(1, 1, 1)]
void main() {}
""",
        )
        program = device.link_program([blocks], [blocks.entry_point("main")])
        owner = device.create_root_shader_object(program)
        root = spy.ShaderCursor(owner)
        samples["nested_parameter_block"] = measure(
            lambda: root.outer.inner.value.write(1.0), 10000, rounds
        )
    return samples


def benchmark_functional(device: spy.Device, rounds: int) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    for count in (1, 6):
        params = ", ".join(f"float a{i}" for i in range(count))
        expression = " + ".join(f"a{i}" for i in range(count))
        functional = spy.Module.load_from_source(
            device,
            f"benchmark_cursor_function_{count}",
            f"float run({params}) {{ return {expression}; }}",
        )
        function = functional.run
        for argument_kind in ("scalar", "tensor"):
            shape = (1,) if argument_kind == "scalar" else (1024,)
            if argument_kind == "scalar":
                args = tuple(float(i) for i in range(count))
            else:
                args = tuple(
                    spy.Tensor.from_numpy(device, np.full(shape, i, dtype=np.float32))
                    for i in range(count)
                )
            result = spy.Tensor.empty(device, shape=shape, dtype=float)
            timings: list[float] = []
            for _ in range(rounds + 2):
                encoder = device.create_command_encoder()
                # Warm the signature and compilation before timing each batch.
                function.append_to(encoder, *args, _result=result)
                gc.collect()
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    start = perf_counter_ns()
                    for _ in range(2000):
                        function.append_to(encoder, *args, _result=result)
                    timings.append((perf_counter_ns() - start) / 2000)
                finally:
                    if gc_enabled:
                        gc.enable()
                device.submit_command_buffer(encoder.finish())
                device.wait()
            # Check the work outside the timed region.
            np.testing.assert_allclose(result.to_numpy(), sum(range(count)))
            name = (
                f"functional_{count}_args"
                if argument_kind == "scalar"
                else f"functional_tensor_{count}_args"
            )
            samples[name] = timings[2:]
    return samples


def benchmark(
    device_type: spy.DeviceType, rounds: int, functional_only: bool = False
) -> dict[str, list[float]]:
    device = spy.Device(
        type=device_type,
        enable_debug_layers=False,
        compiler_options={"include_paths": [spy.SHADER_PATH]},
    )
    samples = {} if functional_only else benchmark_cursors(device, rounds)
    samples.update(benchmark_functional(device, rounds))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("d3d12", "vulkan", "cuda", "metal"), default="d3d12")
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--functional-only",
        action="store_true",
        help="Measure cached calls without preceding cursor workloads",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    samples = benchmark(getattr(spy.DeviceType, args.device), args.rounds, args.functional_only)
    result = {
        "device": args.device,
        "package_path": spy.__file__,
        "python_version": sys.version,
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        "functional_only": args.functional_only,
        "rounds": args.rounds,
        "unit": "nanoseconds per operation",
        "median": {name: median(values) for name, values in samples.items()},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["median"], indent=2))


if __name__ == "__main__":
    main()
