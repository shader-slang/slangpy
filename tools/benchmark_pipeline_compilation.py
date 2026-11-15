# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare serial calls, encoder batching, and explicit pending compilation.

Every sample uses a fresh process and defaults to disabled SlangPy caches. Device,
module and tensor setup is reported separately. GPU driver caches are not cleared.
The pending mode times preparation and compilation only, then dispatches outside
the measured interval to verify results. pending_dispatch includes individual
dispatches and a final wait, matching serial mode's command submission pattern.
For NVRTC 12.9+, use --nvrtc-no-cache to measure compilation rather than its
persistent compiler cache. --cuda-path selects an isolated toolkit in each child;
the loaded NVRTC DLL and version are verified and recorded on Windows.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODES = ("serial", "command_list", "pending", "pending_dispatch")


def get_nvrtc_info() -> dict[str, Any] | None:
    """Identify the NVRTC DLL actually loaded by Slang on Windows."""
    if sys.platform != "win32":
        return None
    import ctypes as ct

    kernel32 = ct.WinDLL("kernel32", use_last_error=True)
    psapi = ct.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ct.c_void_p
    psapi.EnumProcessModules.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_void_p),
        ct.c_ulong,
        ct.POINTER(ct.c_ulong),
    ]
    psapi.GetModuleFileNameExW.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_wchar_p, ct.c_ulong]
    process = kernel32.GetCurrentProcess()
    modules = (ct.c_void_p * 2048)()
    needed = ct.c_ulong()
    if not psapi.EnumProcessModules(process, modules, ct.sizeof(modules), ct.byref(needed)):
        raise ct.WinError(ct.get_last_error())
    assert needed.value <= ct.sizeof(modules), "Module enumeration buffer too small"
    matches: list[dict[str, Any]] = []
    for module in modules[: needed.value // ct.sizeof(ct.c_void_p)]:
        path = ct.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, module, path, len(path)):
            raise ct.WinError(ct.get_last_error())
        if Path(path.value).name.lower().startswith("nvrtc64_"):
            library = ct.CDLL(path.value)
            library.nvrtcVersion.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
            major, minor = ct.c_int(), ct.c_int()
            assert library.nvrtcVersion(ct.byref(major), ct.byref(minor)) == 0
            matches.append(
                {
                    "path": path.value,
                    "version": [major.value, minor.value],
                    "disable_concurrent_nvvm": os.environ.get("NVRTC_DISABLE_CONCURRENT_NVVM"),
                }
            )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one loaded NVRTC DLL, found: {matches}")
    return matches[0]


def run_sample(args: argparse.Namespace) -> dict[str, Any]:
    if args.cuda_path:
        os.environ["CUDA_PATH"] = str(args.cuda_path.resolve())
        os.environ["PATH"] = str(args.cuda_path.resolve() / "bin") + os.pathsep + os.environ["PATH"]
    if args.disable_concurrent_nvvm:
        os.environ["NVRTC_DISABLE_CONCURRENT_NVVM"] = "1"
    os.environ["SLANG_RHI_CUDA_CUBIN"] = "1" if args.cubin else "0"
    os.environ["SLANG_RHI_CUDA_CUBIN_TRACE"] = "1" if args.cubin else "0"
    os.environ["SLANG_RHI_CUDA_CUBIN_OPT_LEVEL"] = str(args.cubin_opt_level)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import slangpy as spy

    start = time.perf_counter()
    device = spy.Device(
        type=getattr(spy.DeviceType, args.backend),
        enable_debug_layers=False,
        enable_rhi_validation=False,
        enable_hot_reload=False,
        enable_ray_tracing=False,
        enable_compilation_reports=True,
        module_cache_path=None,
        shader_cache_path=args.cache_path,
        compiler_options={"downstream_args": ["--no-cache"] if args.nvrtc_no_cache else []},
        pipeline_compilation_mode=(
            spy.PipelineCompilationMode.serial
            if args.worker == "serial"
            else spy.PipelineCompilationMode.parallel
        ),
    )
    nvrtc = get_nvrtc_info() if args.backend == "cuda" else None
    if args.cuda_path and nvrtc:
        assert Path(nvrtc["path"]).is_relative_to(args.cuda_path.resolve()), nvrtc
    source = Path(__file__).with_suffix(".slang").read_text()
    module = spy.Module.load_from_source(device, "pipeline_compilation_workload", source)
    x = spy.Tensor.from_numpy(device, np.linspace(0.1, 0.9, args.elements, dtype=np.float32))
    functions = [
        getattr(module, name).constants({"VARIANT": variant, "ROUNDS": args.rounds})
        for name in ("signal", "network", "orbit")
        for variant in range(args.variants)
    ]
    outputs = [
        spy.Tensor.from_numpy(device, np.full(args.elements, -100, dtype=np.float32))
        for _ in functions
    ]
    device.wait_for_idle()
    setup = time.perf_counter() - start
    phases: dict[str, float] = {}
    serial_calls: list[dict[str, Any]] = []
    start = time.perf_counter()

    if args.worker == "serial":
        for index, (function, output) in enumerate(zip(functions, outputs)):
            call_start = time.perf_counter() if args.wall_trace else 0.0
            function(x, _result=output)
            if args.wall_trace:
                serial_calls.append(
                    {"kernel_index": index, "elapsed_s": time.perf_counter() - call_start}
                )
        device.wait_for_idle()
    elif args.worker == "command_list":
        encoder = device.create_command_encoder()
        for function, output in zip(functions, outputs):
            function.append_to(encoder, x, _result=output)
        recorded = time.perf_counter()
        phases["record_and_prepare_s"] = recorded - start
        command_buffer = encoder.finish()
        finished = time.perf_counter()
        phases["finish_compile_s"] = finished - recorded
        device.submit_command_buffer(command_buffer)
        device.wait_for_idle()
        phases["submit_and_wait_s"] = time.perf_counter() - finished
    else:
        for function, output in zip(functions, outputs):
            function.prepare(x, _result=output)
        prepared = time.perf_counter()
        phases["prepare_s"] = prepared - start
        device.create_pending_pipelines()
        compiled = time.perf_counter()
        phases["compile_s"] = compiled - prepared
        if args.worker == "pending_dispatch":
            for function, output in zip(functions, outputs):
                function(x, _result=output)
            device.wait_for_idle()
            phases["dispatch_and_wait_s"] = time.perf_counter() - compiled

    elapsed = time.perf_counter() - start
    reports_before_validation = device.get_compilation_reports()
    if args.worker == "pending":
        for output in outputs:
            np.testing.assert_array_equal(output.to_numpy(), -100)
        for function, output in zip(functions, outputs):
            function(x, _result=output)
        device.wait_for_idle()

    values = np.stack([output.to_numpy() for output in outputs])
    assert np.isfinite(values).all()
    reports = device.get_compilation_reports()
    assert reports == reports_before_validation, "Validation unexpectedly compiled another kernel"
    entries = [entry for report in reports for entry in report["entry_point_reports"]]
    pipelines = [entry for report in reports for entry in report["pipeline_reports"]]
    assert len(entries) == len(functions), (len(entries), len(functions))
    assert len(pipelines) == len(functions), (len(pipelines), len(functions))
    if args.cache_path is None:
        assert not any(entry["is_cached"] for entry in entries)
    wall_trace: list[dict[str, Any]] = []
    if args.wall_trace:
        # Reports are in program creation order, matching this workload's function order.
        # Only use the independent RHI Timer::now() timestamps. Slang's cumulative
        # compile_time/compile_slang_time/compile_downstream_time are NOT reliable in parallel.
        active_reports = [report for report in reports if report["entry_point_reports"]]
        assert len(active_reports) == len(functions)
        for index, report in enumerate(active_reports):
            for stage, field in (
                ("target", "entry_point_reports"),
                ("pipeline", "pipeline_reports"),
            ):
                assert len(report[field]) == 1
                entry = report[field][0]
                wall_trace.append(
                    {
                        "kernel_index": index,
                        "function": ("signal", "network", "orbit")[index // args.variants],
                        "variant": index % args.variants,
                        "stage": stage,
                        "start_ns": entry["start_time"],
                        "end_ns": entry["end_time"],
                    }
                )
    result = {
        "mode": args.worker,
        "backend": args.backend,
        "nvrtc": nvrtc,
        "nvrtc_no_cache": args.nvrtc_no_cache,
        "cubin": args.cubin,
        "cubin_opt_level": args.cubin_opt_level if args.cubin else None,
        "cache_path": str(args.cache_path.resolve()) if args.cache_path else None,
        "cached_entry_points": sum(entry["is_cached"] for entry in entries),
        "cuda_cache_disable": os.environ.get("CUDA_CACHE_DISABLE"),
        "adapter": str(device.info.adapter_name),
        "python": platform.python_version(),
        "slangpy_version": spy.SGL_VERSION,
        "slangpy_git_version": spy.SGL_GIT_VERSION,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "build_dir": (ROOT / "slangpy/.build_dir").read_text().strip(),
        "kernels": len(functions),
        "variants_per_function": args.variants,
        "rounds": args.rounds,
        "elements": args.elements,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "setup_s": setup,
        "elapsed_s": elapsed,
        **phases,
        # Reports are used only for counts/cache checks, never for parallel timings.
        "compiled_entry_points": len(entries),
        "created_pipelines": len(pipelines),
        "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "outputs": values.tolist(),
    }
    if args.wall_trace:
        result["wall_trace"] = wall_trace
        result["serial_calls"] = serial_calls
    device.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("d3d12", "vulkan", "cuda"), default="vulkan")
    parser.add_argument(
        "--variants", type=int, default=8, help="Variants per each of three functions"
    )
    parser.add_argument("--rounds", type=int, default=32, help="Unrolled iterations per kernel")
    parser.add_argument("--elements", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--wall-trace",
        action="store_true",
        help="Record raw RHI wall timestamps, not compiler timing counters",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--cuda-path", type=Path, help="Process-local CUDA toolkit root for NVRTC selection"
    )
    parser.add_argument("--disable-concurrent-nvvm", action="store_true")
    parser.add_argument(
        "--cubin", action="store_true", help="Use the optional standalone PTX compiler"
    )
    parser.add_argument("--cubin-opt-level", type=int, choices=range(4), default=3)
    parser.add_argument(
        "--cache-path", type=Path, help="Enable the SGL persistent shader/pipeline cache"
    )
    parser.add_argument(
        "--expect-cubin-cache",
        choices=("cold", "warm"),
        help="Assert CUBIN cache state in every sample",
    )
    parser.add_argument(
        "--nvrtc-no-cache", action="store_true", help="Disable NVRTC 12.9+ compiler caching"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pipeline-compilation.json")
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.variants, args.rounds, args.elements, args.repeat) < 1:
        parser.error("Counts must be positive")
    if args.backend != "cuda" and (
        args.cuda_path or args.disable_concurrent_nvvm or args.nvrtc_no_cache or args.cubin
    ):
        parser.error("NVRTC options require --backend cuda")
    if args.expect_cubin_cache and not args.cubin:
        parser.error("--expect-cubin-cache requires --cubin")
    if args.worker:
        result = run_sample(args)
        print("BENCHMARK_JSON=" + json.dumps(result))
        return

    import numpy as np

    samples: list[dict[str, Any]] = []
    reference = None
    rng = random.Random(2026)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeat):
        modes = list(args.modes)
        rng.shuffle(modes)
        for mode in modes:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--backend",
                args.backend,
                "--variants",
                str(args.variants),
                "--rounds",
                str(args.rounds),
                "--elements",
                str(args.elements),
            ]
            if args.cuda_path:
                command += ["--cuda-path", str(args.cuda_path.resolve())]
            if args.disable_concurrent_nvvm:
                command += ["--disable-concurrent-nvvm"]
            if args.nvrtc_no_cache:
                command += ["--nvrtc-no-cache"]
            if args.wall_trace:
                command += ["--wall-trace"]
            if args.cubin:
                command += ["--cubin", "--cubin-opt-level", str(args.cubin_opt_level)]
            if args.cache_path:
                command += ["--cache-path", str(args.cache_path.resolve())]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f"{mode} failed:\n{completed.stdout}\n{completed.stderr}")
            line = next(
                line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_JSON=")
            )
            sample = json.loads(line.removeprefix("BENCHMARK_JSON="))
            cubin_trace = [
                json.loads(line.removeprefix("CUBIN_TRACE="))
                for line in completed.stderr.splitlines()
                if line.startswith("CUBIN_TRACE=")
            ]
            if args.cubin:
                assert len(cubin_trace) == sample["kernels"], completed.stderr
                assert all(entry["success"] for entry in cubin_trace), completed.stderr
                assert len({entry["program_id"] for entry in cubin_trace}) == sample["kernels"]
                assert len({entry["fingerprint"] for entry in cubin_trace}) == 1
                if args.wall_trace:
                    events = sample["wall_trace"] + cubin_trace
                    envelope = (
                        max(e["end_ns"] for e in events) - min(e["start_ns"] for e in events)
                    ) / 1e9
                    assert envelope <= sample["elapsed_s"] + 0.01
                if args.expect_cubin_cache:
                    expected_hit = args.expect_cubin_cache == "warm"
                    assert all(
                        entry["cache_hit"] == expected_hit for entry in cubin_trace
                    ), cubin_trace
                sample["cubin_trace"] = cubin_trace
            sample["repeat"] = repeat
            values = np.array(sample["outputs"])
            if reference is None:
                reference = values
            else:
                np.testing.assert_allclose(values, reference, rtol=1e-5, atol=1e-6)
            samples.append(sample)
            args.output.write_text(json.dumps({"samples": samples}, indent=2) + "\n")
            print(f"{repeat + 1}/{args.repeat} {mode:18s} {sample['elapsed_s']:.3f} s", flush=True)

    print("\nMode                 Median (s)    Min (s)    Max (s)")
    for mode in args.modes:
        times = [sample["elapsed_s"] for sample in samples if sample["mode"] == mode]
        print(f"{mode:20s} {statistics.median(times):10.3f} {min(times):10.3f} {max(times):10.3f}")
    print(f"\nVerified matching outputs across all samples. Results: {args.output}")


if __name__ == "__main__":
    main()
