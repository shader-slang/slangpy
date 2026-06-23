# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)


PROGRAM_REPORT_KEYS = {
    "label",
    "alive",
    "create_time",
    "compile_time",
    "compile_slang_time",
    "compile_downstream_time",
    "create_pipeline_time",
    "entry_point_reports",
    "pipeline_reports",
}

ENTRY_POINT_REPORT_KEYS = {
    "name",
    "start_time",
    "end_time",
    "create_time",
    "compile_time",
    "compile_slang_time",
    "compile_downstream_time",
    "is_cached",
    "cache_size",
}

PIPELINE_REPORT_KEYS = {
    "type",
    "start_time",
    "end_time",
    "create_time",
    "is_cached",
    "cache_size",
}


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compilation_reports(test_id: str, device_type: spy.DeviceType):
    device = spy.Device(
        type=device_type,
        enable_compilation_reports=True,
        label=f"compilation-reports-{device_type.name}",
    )
    try:
        module = device.load_module_from_source(
            module_name=f"compilation_reports_{test_id}",
            source="""
                [shader("compute")]
                [numthreads(1, 1, 1)]
                void compute_main(uint3 tid: SV_DispatchThreadID) { }
            """,
        )
        program = device.link_program([module], [module.entry_point("compute_main")])
        pipeline = device.create_compute_pipeline(program, defer_target_compilation=False)

        assert pipeline is not None

        program_report = program.get_compilation_report()
        assert isinstance(program_report, dict)
        assert set(program_report.keys()) == PROGRAM_REPORT_KEYS
        assert program_report["alive"] is True

        entry_point_reports = program_report["entry_point_reports"]
        assert isinstance(entry_point_reports, list)
        assert len(entry_point_reports) == 1
        assert set(entry_point_reports[0].keys()) == ENTRY_POINT_REPORT_KEYS
        assert entry_point_reports[0]["name"] == "compute_main"

        pipeline_reports = program_report["pipeline_reports"]
        assert isinstance(pipeline_reports, list)
        assert len(pipeline_reports) == 1
        assert set(pipeline_reports[0].keys()) == PIPELINE_REPORT_KEYS
        assert pipeline_reports[0]["type"] == "compute"

        device_reports = device.get_compilation_reports()
        assert isinstance(device_reports, list)
        assert len(device_reports) == 1
        assert device_reports[0] == program_report
    finally:
        device.close()
