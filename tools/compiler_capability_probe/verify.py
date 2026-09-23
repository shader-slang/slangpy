# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the measured baseline; a future upstream fix should change these observations.

This intentionally checks both successes and diagnostic failures. Do not interpret the
expected failures as behavior SlangPy should preserve: consult the workaround ledger.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def verify(directory: Path) -> dict[str, dict[str, Any]]:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    cases = {result["case"]["name"]: result for result in results}
    checks = 0

    def expect(name: str, status: str, diagnostic: str = "", **artifact: Any) -> None:
        nonlocal checks
        case = cases[name]
        assert case["status"] == status, (name, case)
        assert case["returncode"] == (0 if status == "ok" else 1), (name, case)
        if diagnostic:
            message = case.get("diagnostics", "") + case.get("error", "")
            assert diagnostic in message, (name, message)
        for key, value in artifact.items():
            assert case["artifact"].get(key) == value, (name, key, case["artifact"])
        checks += 1

    for model in ("6_6", "6_9"):
        expect(f"dxil_cap_{model}", "ok", shader_models=[["lib", "6", "3"]])
        expect(f"dxil_profile_{model}", "ok", shader_models=[["lib", "6", model.split("_")[1]]])
    expect("dxil_wave_match_cap", "error", "WaveMatch")
    expect("dxil_wave_match_profile", "ok", shader_models=[["cs", "6", "6"]])
    for model in ("6_6", "6_7"):
        expect(f"ray_payload_{model}", "ok", shader_models=[["lib", "6", model.split("_")[1]]])
    expect("spirv_cap_1_3", "ok", spirv_version="1.5")
    expect("spirv_baseline_1_3", "ok", spirv_version="1.3")
    expect("spirv_cap_1_6", "ok", spirv_version="1.6")
    expect("spirv_baseline_1_6", "ok", spirv_version="1.6")
    expect("spirv_empty", "ok", spirv_version="1.5")
    expect("spirv_empty_strict", "ok", spirv_version="1.5")
    expect("spirv_bundle_implicit", "ok")
    expect("spirv_bundle_full", "ok")
    expect("spirv_bundle_minimal", "error", "E41013")
    for source in ("inferred", "late"):
        expect(f"strict_{source}_False", "ok", "E41012")
        expect(f"strict_{source}_True", "error", "E41013")
    expect("strict_explicit_True", "ok")
    expect("strict_empty_inferred", "ok")
    expect("strict_baseline_inferred", "error", "E41013")
    for tier in ("5_0", "7_0", "8_0", "8_9", "9_0"):
        expect(f"cuda_{tier}", "ok", ptx_target=["sm_" + tier.replace("_", "")])
    for tier in ("7_5", "8_6", "12_0"):
        expect(f"cuda_{tier}", "error", "Unknown capability")
    expect("cuda_empty", "ok", ptx_target=["sm_50"])
    expect("cuda_half_5_0", "ok", ptx_target=["sm_60"])
    expect("cuda_late_warn", "ok", "E41012", ptx_target=["sm_80"])
    expect("cuda_late_strict", "error", "E41013")
    expect("cuda_higher_implies_lower", "ok", ptx_target=["sm_90"])
    for name in (
        "cuda_bridge_86",
        "cuda_conflicting_80_75",
        "cuda_duplicate_80",
        "cuda_unsupported_999",
    ):
        expect(name, "error", "defined more than once")
    expect("ser_native", "error", "hlsl_nvapi")
    expect("ser_nvapi", "error", "ser_hlsl_native")
    expect("ser_native_permissive", "ok", native_hitobject=True, nvapi_hitobject=False)
    expect("ser_nvapi_permissive", "ok", native_hitobject=False, nvapi_hitobject=True)
    expect("ser_both", "ok", native_hitobject=False, nvapi_hitobject=True)
    expect("ser_native_dxil", "ok", shader_models=[["lib", "6", "9"]])
    expect("ser_nvapi_dxil", "ok", shader_models=[["lib", "6", "9"]])
    for backend in ("d3d12", "vulkan", "cuda", "cpu"):
        if f"device_{backend}_simple" in cases:
            expect(f"device_{backend}_simple", "ok")
            expect(f"device_{backend}_functional_import", "ok")
            expect(f"forward_{backend}_legacy", "ok")
    if "device_vulkan_strict_wave" in cases:
        expect("device_vulkan_strict_wave", "error", "spvGroupNonUniformArithmetic")
        expect("device_vulkan_strict_atomic64", "ok")
    if "device_cuda_simple" in cases:
        # This baseline inventory is a CC 7.5 device; Slang drops the unknown 7.2/7.5 tiers.
        expect("device_cuda_simple", "ok", ptx_target=["sm_70"])
    for target in ("metal", "wgsl", "cpp"):
        expect(f"source_{target}", "ok")
    assert all(case["status"] in ("ok", "error") for case in cases.values())
    tags = sorted({case["build_tag"] for case in cases.values()})
    assert len(tags) == 1, tags
    print(
        f"{directory}: {checks} observation checks passed; {len(cases)} probes; compiler {tags[0]}"
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    reference = None
    for directory in args.directories:
        cases = verify(directory)
        if reference is not None:
            assert cases.keys() == reference.keys(), "Different probe inventories"
            for name, case in cases.items():
                assert case["status"] == reference[name]["status"], name
                artifact = {
                    k: v
                    for k, v in case.get("artifact", {}).items()
                    if k not in ("sha256", "bytes")
                }
                previous = {
                    k: v
                    for k, v in reference[name].get("artifact", {}).items()
                    if k not in ("sha256", "bytes")
                }
                assert artifact == previous, (name, artifact, previous)
            print(
                f"All {len(cases)} statuses and inspected output properties match the first compiler."
            )
        else:
            reference = cases


if __name__ == "__main__":
    main()
