# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run isolated Slang API experiments and retain inputs, diagnostics, and generated code.

Build SlangPy and the native probe before running this script. Outputs are observations,
including expected compiler failures; this is not a test of a new public SlangPy API.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any


SIMPLE = """
RWStructuredBuffer<uint> output;
[shader("compute")]
[numthreads(1, 1, 1)]
void probe_main(uint3 tid : SV_DispatchThreadID) { output[tid.x] = 7; }
"""

SOURCES = {
    "simple": SIMPLE,
    "functional_import": "import slangpy;\n" + SIMPLE,
    "wave": SIMPLE.replace("= 7", "= WaveActiveSum(tid.x + 1)"),
    "wave_match": SIMPLE.replace("= 7", "= WaveMatch(tid.x).x"),
    "atomic64": SIMPLE.replace("<uint>", "<uint64_t>").replace(
        "output[tid.x] = 7;", "InterlockedAdd(output[0], uint64_t(1));"
    ),
    "half": SIMPLE.replace("<uint>", "<half>").replace("= 7", "= half(tid.x + 1)"),
    "inferred": "[require(_sm_6_6)] uint needsNew(uint x) { return x + 1; }\n"
    + SIMPLE.replace("= 7", "= needsNew(tid.x)"),
    "explicit": "[require(_sm_6_6)] uint needsNew(uint x) { return x + 1; }\n"
    + SIMPLE.replace('[shader("compute")]', '[require(_sm_6_6)]\n[shader("compute")]').replace(
        "= 7", "= needsNew(tid.x)"
    ),
    "late": SIMPLE.replace(
        "output[tid.x] = 7;", "__requireCapability(_sm_6_6); output[tid.x] = 7;"
    ),
    "cuda_late": SIMPLE.replace(
        "output[tid.x] = 7;", "__requireCapability(_cuda_sm_8_9); output[tid.x] = 7;"
    ),
    "spirv_bundle": "[require(SPV_EXT_physical_storage_buffer)] uint addressFeature() { return 7; }\n"
    + SIMPLE.replace("= 7", "= addressFeature()"),
    "ser": """
struct Payload { uint value; };
RaytracingAccelerationStructure scene;
RWStructuredBuffer<uint> output;
[shader("raygeneration")]
void probe_main()
{
    RayDesc ray;
    ray.Origin = float3(0); ray.Direction = float3(0, 0, 1);
    ray.TMin = 0; ray.TMax = 100;
    Payload payload = {};
    HitObject hit = HitObject::TraceRay(scene, 0, 255, 0, 0, 0, ray, payload);
    output[0] = hit.IsHit();
}
""",
}
SOURCES["ray_payload"] = (
    SOURCES["ser"]
    .replace(
        "HitObject hit = HitObject::TraceRay(scene, 0, 255, 0, 0, 0, ray, payload);",
        "TraceRay(scene, 0, 255, 0, 0, 0, ray, payload);",
    )
    .replace("output[0] = hit.IsHit();", "output[0] = payload.value;")
)


def collect_devices(output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import slangpy as spy

    inventory: dict[str, Any] = {"python": sys.executable, "slangpy": spy.__file__, "devices": {}}
    for name in ("d3d12", "vulkan", "cuda", "cpu", "metal", "wgpu"):
        device = None
        try:
            device = spy.Device(type=getattr(spy.DeviceType, name), enable_debug_layers=False)
            item: dict[str, Any] = {
                "adapter": device.info.adapter_name,
                "capabilities": list(device.capabilities),
                "features": [str(feature) for feature in device.features],
            }
            inventory["devices"][name] = item
            module = device.load_module_from_source("capability_probe_smoke", SIMPLE)
            program = device.link_program([module], [module.entry_point("probe_main")])
            kernel = device.create_compute_kernel(program)
            buffer = device.create_buffer(
                size=4, usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
            )
            kernel.dispatch(thread_count=[1, 1, 1], vars={"output": buffer})
            item["smoke_result"] = int(buffer.to_numpy().view(np.uint32)[0])
            if item["smoke_result"] != 7:
                raise RuntimeError(f"Unexpected GPU result {item['smoke_result']}")
        except Exception as error:
            inventory["devices"].setdefault(name, {})["error"] = str(error)
        finally:
            if device is not None:
                device.close()
    inventory_path = output_dir / "device_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    return inventory


def cases(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    root = Path(__file__).resolve().parents[2]

    def add(name: str, target: str, source: str = "simple", **options: Any) -> None:
        result.append({"name": name, "target": target, "source": source, **options})

    for model in ("6_6", "6_9"):
        add(f"dxil_cap_{model}", "dxil-asm", capabilities=[f"_sm_{model}"], whole_program=True)
        add(
            f"dxil_profile_{model}",
            "dxil-asm",
            capabilities=[f"_sm_{model}"],
            profile=f"sm_{model}",
            whole_program=True,
        )
    add("dxil_wave_match_cap", "dxil-asm", "wave_match", capabilities=["_sm_6_6"])
    add(
        "dxil_wave_match_profile",
        "dxil-asm",
        "wave_match",
        capabilities=["_sm_6_6"],
        profile="sm_6_6",
    )
    for model in ("6_6", "6_7"):
        add(
            f"ray_payload_{model}",
            "dxil-asm",
            "ray_payload",
            profile=f"sm_{model}",
            whole_program=True,
        )
    for version in ("1_3", "1_6"):
        add(f"spirv_cap_{version}", "spirv", capabilities=[f"_spirv_{version}"])
        add(
            f"spirv_baseline_{version}",
            "spirv",
            capabilities=[f"_spirv_{version}"],
            profile="spirv_1_0",
        )
    add("spirv_empty", "spirv")
    add("spirv_empty_strict", "spirv", strict=True)
    for label, profile in (("implicit", ""), ("minimal", "spirv_1_0"), ("full", "spirv_1_6")):
        add(
            f"spirv_bundle_{label}",
            "spirv",
            "spirv_bundle",
            capabilities=["_spirv_1_6"],
            profile=profile,
            strict=True,
        )
    for source in ("inferred", "explicit", "late"):
        for strict in (False, True):
            add(
                f"strict_{source}_{strict}",
                "hlsl",
                source,
                capabilities=["_sm_6_0"],
                profile="sm_6_0",
                strict=strict,
            )
    add("strict_empty_inferred", "hlsl", "inferred", strict=True)
    add("strict_baseline_inferred", "hlsl", "inferred", strict=True, capabilities=["hlsl"])
    for tier in ("5_0", "7_0", "7_5", "8_0", "8_6", "8_9", "9_0", "12_0"):
        add(f"cuda_{tier}", "ptx", capabilities=[f"cuda_sm_{tier}"])
    add("cuda_empty", "ptx")
    add("cuda_half_5_0", "ptx", "half", capabilities=["cuda_sm_5_0"])
    add("cuda_late_warn", "ptx", "cuda_late", capabilities=["cuda_sm_8_0"])
    add("cuda_late_strict", "ptx", "cuda_late", capabilities=["cuda_sm_8_0"], strict=True)
    add(
        "cuda_bridge_86",
        "ptx",
        capabilities=["cuda_sm_8_0"],
        downstream_args=["--gpu-architecture=compute_86"],
    )
    add(
        "cuda_conflicting_80_75",
        "ptx",
        capabilities=["cuda_sm_8_0"],
        downstream_args=["--gpu-architecture=compute_75"],
    )
    add(
        "cuda_duplicate_80",
        "ptx",
        capabilities=["cuda_sm_8_0"],
        downstream_args=["--gpu-architecture=compute_80"],
    )
    add(
        "cuda_unsupported_999",
        "ptx",
        capabilities=["cuda_sm_8_0"],
        downstream_args=["--gpu-architecture=compute_999"],
    )
    add("cuda_higher_implies_lower", "ptx", capabilities=["cuda_sm_7_0", "cuda_sm_9_0"])
    for label, caps in (
        ("model_only", ["_sm_6_9"]),
        ("native", ["_sm_6_9", "ser_hlsl_native"]),
        ("nvapi", ["_sm_6_9", "hlsl_nvapi"]),
        ("both", ["_sm_6_9", "ser_hlsl_native", "hlsl_nvapi"]),
    ):
        add(f"ser_{label}", "hlsl", "ser", capabilities=caps, profile="sm_6_9", strict=True)
        add(f"ser_{label}_permissive", "hlsl", "ser", capabilities=caps, profile="sm_6_9")
    add(
        "ser_native_dxil",
        "dxil-asm",
        "ser",
        capabilities=["_sm_6_9", "ser_hlsl_native"],
        profile="sm_6_9",
        whole_program=True,
    )
    add(
        "ser_nvapi_dxil",
        "dxil-asm",
        "ser",
        capabilities=["_sm_6_9", "hlsl_nvapi"],
        profile="sm_6_9",
        whole_program=True,
        downstream_args=[
            "-I" + str(root / "build/windows-msvc/_deps/nvapi-src"),
            "-DNV_SHADER_EXTN_SLOT=u999",
        ],
    )
    for backend, item in inventory.get("devices", {}).items():
        if "capabilities" not in item:
            continue
        target = {
            "d3d12": "dxil-asm",
            "vulkan": "spirv",
            "cuda": "ptx",
            "cpu": "cpp",
            "metal": "metal",
            "wgpu": "wgsl",
        }[backend]
        for source in ("simple", "functional_import", "wave", "atomic64"):
            add(
                f"device_{backend}_{source}",
                target,
                source,
                capabilities=item["capabilities"],
                ignore_unknown=True,
                profile="sm_6_6" if backend == "d3d12" else "",
            )
        add(
            f"device_{backend}_strict_wave",
            target,
            "wave",
            capabilities=item["capabilities"],
            ignore_unknown=True,
            strict=True,
            profile="sm_6_6" if backend == "d3d12" else "",
        )
        if backend == "vulkan":
            add(
                "device_vulkan_strict_atomic64",
                target,
                "atomic64",
                capabilities=item["capabilities"],
                ignore_unknown=True,
                strict=True,
            )
        # Mirrors the previously commented-out additions: old profiles, NVAPI, session scope.
        add(
            f"forward_{backend}_legacy",
            target,
            "functional_import",
            capabilities=item["capabilities"] + ["hlsl_nvapi"],
            ignore_unknown=True,
            profile="sm_6_6" if backend in ("d3d12", "vulkan") else "",
            option_scope="session",
        )
    for target in ("metal", "wgsl", "cpp"):
        add(f"source_{target}", target, capabilities=[target])
    return result


def inspect_output(path: Path, target: str) -> dict[str, Any]:
    data = path.read_bytes()
    result: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    if target == "spirv":
        words = struct.unpack(f"<{len(data) // 4}I", data)
        result["spirv_version"] = f"{(words[1] >> 16) & 255}.{(words[1] >> 8) & 255}"
        capabilities = []
        extensions = []
        position = 5
        while position < len(words):
            size, opcode = words[position] >> 16, words[position] & 65535
            if size == 0:
                raise ValueError("Invalid SPIR-V instruction length")
            if opcode == 17:
                capabilities.append(words[position + 1])
            if opcode == 10:
                extensions.append(
                    data[(position + 1) * 4 : (position + size) * 4].split(b"\0")[0].decode()
                )
            position += size
        result.update(spirv_capabilities=capabilities, spirv_extensions=extensions)
    else:
        text = data.decode("utf-8", errors="replace")
        if target == "ptx":
            result["ptx_target"] = re.findall(r"(?m)^\s*\.target\s+([^\r\n]+)", text)
            result["ptx_version"] = re.findall(r"(?m)^\s*\.version\s+([^\r\n]+)", text)
        if target == "dxil-asm":
            result["shader_models"] = re.findall(
                r'!\{!"(lib|cs|vs|ps)", i32 (\d+), i32 (\d+)\}', text
            )
        if target == "hlsl":
            result["nvapi_hitobject"] = "NvHitObject" in text
            result["native_hitobject"] = "dx::HitObject" in text
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--dxc", type=Path)
    parser.add_argument("--nvrtc", type=Path)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    environment: dict[str, Any] = {
        "compiler_library": str(args.library.resolve()),
        "compiler_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "python": sys.executable,
    }
    if args.nvrtc and args.nvrtc.is_file():
        runtime = ctypes.CDLL(str(args.nvrtc.resolve()))
        major, minor, count = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if runtime.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
            raise RuntimeError("nvrtcVersion failed")
        if runtime.nvrtcGetNumSupportedArchs(ctypes.byref(count)) != 0:
            raise RuntimeError("nvrtcGetNumSupportedArchs failed")
        architectures = (ctypes.c_int * count.value)()
        if runtime.nvrtcGetSupportedArchs(architectures) != 0:
            raise RuntimeError("nvrtcGetSupportedArchs failed")
        environment.update(
            nvrtc_version=f"{major.value}.{minor.value}",
            nvrtc_library=str(args.nvrtc.resolve()),
            nvrtc_supported_architectures=list(architectures),
        )
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    for cache in ("CUDA_CACHE_PATH", "OPTIX_CACHE_PATH"):
        cache_path = args.output.resolve() / cache.lower()
        cache_path.mkdir(exist_ok=True)
        os.environ[cache] = str(cache_path)
    inventory = (
        json.loads(args.inventory.read_text()) if args.inventory else collect_devices(args.output)
    )
    root = Path(__file__).resolve().parents[2]
    results = []
    failures = 0
    for case in cases(inventory):
        if args.filter and args.filter not in case["name"]:
            continue
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=True)
        source = directory / "input.slang"
        source.write_text(SOURCES[case["source"]], encoding="utf-8")
        output = directory / "output.bin"
        command = [
            str(args.probe.resolve()),
            "--library",
            str(args.library.resolve()),
            "--target",
            case["target"],
            "--source",
            str(source.resolve()),
            "--output",
            str(output.resolve()),
            "--entry",
            "probe_main",
            "--include",
            str(root / "slangpy" / "slang"),
        ]
        for capability in case.get("capabilities", []):
            command.extend(["--capability", capability])
        for argument in case.get("downstream_args", []):
            command.extend(["--downstream-arg", argument])
        for key in ("profile", "option_scope"):
            if case.get(key):
                command.extend(["--" + key.replace("_", "-"), case[key]])
        for key in ("strict", "ignore_unknown", "whole_program"):
            if case.get(key):
                command.extend(["--" + key.replace("_", "-"), "1"])
        for key in ("dxc", "nvrtc"):
            if getattr(args, key):
                command.extend(["--" + key, str(getattr(args, key).resolve())])
        (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            (directory / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            result = json.loads(completed.stdout)
            result.update(case=case, returncode=completed.returncode)
            if result["status"] == "ok":
                result["artifact"] = inspect_output(output, case["target"])
            elif completed.returncode != 1:
                failures += 1
        except (subprocess.TimeoutExpired, ValueError) as error:
            result = {"case": case, "status": "harness_error", "error": str(error)}
            failures += 1
        results.append(result)
        (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        artifact = result.get("artifact", {})
        observed = (
            artifact.get("spirv_version")
            or artifact.get("ptx_target")
            or artifact.get("shader_models")
            or ""
        )
        print(f"{case['name']}: {result['status']} {observed}", flush=True)
        (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Recorded {len(results)} probes; harness errors: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
