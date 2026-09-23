# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the measured baseline; a future upstream fix should change these observations.

This intentionally checks both successes and diagnostic failures. Do not interpret the
expected failures as behavior SlangPy should preserve: consult the workaround ledger.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def verify(directory: Path) -> dict[str, dict[str, Any]]:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    cases = {result["case"]["name"]: result for result in results}
    assert len(cases) == len(results), "Duplicate case names"
    checks = 0
    checked_names: set[str] = set()

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
        checked_names.add(name)

    if "dxil_cap_6_6" in cases:
        verify_baseline(expect, cases)
    if any(name.startswith("interaction_") for name in cases):
        verify_profiles(expect, cases)
        assert {name for name in cases if name.startswith("interaction_")} <= checked_names
    assert checks > 0, "No recognized probe suite"
    assert all(case["status"] in ("ok", "error") for case in cases.values())
    tags = sorted({case["build_tag"] for case in cases.values()})
    assert len(tags) == 1, tags
    print(
        f"{directory}: {checks} observation checks passed; {len(cases)} probes; compiler {tags[0]}"
    )
    return cases


def verify_baseline(expect: Callable[..., None], cases: dict[str, dict[str, Any]]) -> None:
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


def verify_profiles(expect: Callable[..., None], cases: dict[str, dict[str, Any]]) -> None:
    """Check the complete interaction suite, including failures after successful validation."""

    def check(name: str, status: str = "ok", diagnostic: str = "", **artifact: Any) -> None:
        expect("interaction_" + name, status, diagnostic, **artifact)

    for mode in ("permissive", "strict"):
        for name in ("dx_lower_raw", "dx_lower_inferred"):
            check(f"{name}_{mode}", shader_models=[["cs", "6", "0"]])
        check(f"dx_lower_wave_{mode}", "error", "Opcode WaveMatch not valid in shader model cs_6_0")
        assert cases[f"interaction_dx_lower_wave_{mode}"]["stage"] == "codegen"
        for name in (
            "dx_equal_wave",
            "dx_higher_wave",
            "dx_profile_only_wave",
            "dx_6_6_with_6_9",
            "dx_native_implied",
            "dx_native_alias",
        ):
            check(f"{name}_{mode}", shader_models=[["cs", "6", "6"]])
            assert "E41012" not in cases[f"interaction_{name}_{mode}"].get("diagnostics", ""), name
        for name in ("dx_6_6_without_6_9", "dx_native_removed"):
            if mode == "strict":
                check(f"{name}_{mode}", "error", "E41013")
            else:
                check(f"{name}_{mode}", diagnostic="E41012", shader_models=[["cs", "6", "6"]])
        if mode == "strict":
            for name in ("dx_ser_lower", "dx_ser_equal"):
                check(f"{name}_{mode}", "error", "hlsl_nvapi")
                assert cases[f"interaction_{name}_{mode}"]["stage"] == "load_module"
        else:
            check("dx_ser_lower_permissive", "error", "requires shader model 6.9 or greater")
            check(
                "dx_ser_equal_permissive",
                diagnostic="hlsl_nvapi",
                shader_models=[["lib", "6", "9"]],
            )
        check(f"dx_ser_both_6_6_{mode}", shader_models=[["lib", "6", "6"]])
        check(f"dx_ser_both_lower_{mode}", "error", "unsupported lib_6_1 or lib_6_2")

    for version in ("1_0", "1_3", "1_6"):
        for label in ("empty", "raw_lower", "raw_equal", "raw_higher", "alias_higher"):
            output_version = (
                "1.6" if label in ("raw_higher", "alias_higher") else version.replace("_", ".")
            )
            check(f"spv_{version}_{label}_simple", spirv_version=output_version)
            name = f"spv_{version}_{label}_spirv_bundle"
            if version == "1_6" or label == "alias_higher":
                check(name, spirv_version=output_version)
            else:
                check(name, "error", "SPV_EXT_physical_storage_buffer")
                assert "E41013" in cases["interaction_" + name]["diagnostics"]
    check("spv_feature_added", spirv_version="1.6")
    check("spv_feature_removed", spirv_version="1.6")
    check("spv_extension_implied_version", spirv_version="1.3")
    check("family_dxil-asm_spirv_1_6", "error", "E41013")
    check("family_dxil-asm_glsl_460", shader_models=[["cs", "6", "0"]])
    check("family_dxil-asm_cs_6_6", shader_models=[["cs", "6", "6"]])
    check("family_dxil-asm_ps_6_6", "error", "E36107")
    check("family_spirv_sm_6_6", spirv_version="1.4")
    check("family_spirv_sm_6_6_raw", spirv_version="1.4")
    check("family_spirv_glsl_460", spirv_version="1.3")
    check("family_spirv_metallib_2_4", spirv_version="1.5")
    for profile in ("sm_6_6", "spirv_1_6"):
        check(f"family_ptx_{profile}", ptx_target=["sm_80"])
    for name in ("metal_metallib_2_4", "metal_sm_6_6", "wgsl_sm_6_6", "cpp_sm_6_6"):
        check("family_" + name)
    for profile in ("sm_6_typo", "spirv_9_9", "cuda_sm_8_0"):
        check("unknown_" + profile, "error", "Unknown profile: " + profile)
        case = cases["interaction_unknown_" + profile]
        assert case["stage"] == "resolve_options"
        assert case["profile_lookup:" + profile] == "0"
    for label in ("all", "no_raw_versions"):
        if f"interaction_device_d3d12_{label}" in cases:
            check(f"device_d3d12_{label}", shader_models=[["cs", "6", "6"]])
        if f"interaction_device_vulkan_{label}" in cases:
            check(f"device_vulkan_{label}", spirv_version="1.6")


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
