# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import sys

import slangpy as spy
from slangpy.testing import helpers
from slangpy.testing.helpers import test_id  # type: ignore (pytest fixture)

# TODO: Due to a bug in "Apple clang", the exception binding in nanobind
# raises RuntimeError instead of SlangCompileError
SlangCompileError = RuntimeError if sys.platform == "darwin" else spy.SlangCompileError


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Loading non-existing module must raise an exception
    with pytest.raises(Exception, match='Failed to load slang module "does_not_exist.slang"'):
        device.load_module("does_not_exist.slang")

    # Compilation errors must raise an exception
    with pytest.raises(SlangCompileError, match="unexpected end of file, expected identifier"):
        device.load_module("test_shader_compile_error.slang")

    # Loading a valid module must succeed
    module = device.load_module("test_shader_foo.slang")
    assert len(module.entry_points) == 4
    main_a = module.entry_point("main_a")
    assert main_a.name == "main_a"
    assert main_a.stage == spy.ShaderStage.compute
    assert main_a.layout.compute_thread_group_size == [1, 1, 1]
    main_b = module.entry_point("main_b")
    assert main_b.name == "main_b"
    assert main_b.stage == spy.ShaderStage.compute
    assert main_b.layout.compute_thread_group_size == [16, 8, 1]
    main_vs = module.entry_point("main_vs")
    assert main_vs.name == "main_vs"
    assert main_vs.stage == spy.ShaderStage.vertex
    main_fs = module.entry_point("main_fs")
    assert main_fs.name == "main_fs"
    assert main_fs.stage == spy.ShaderStage.fragment

    # Check back refs to device and session are correct
    assert module.session == device.slang_session
    assert module.session.device == device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_from_source(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Compilation errors must raise an exception
    with pytest.raises(SlangCompileError, match="unexpected end of file, expected identifier"):
        device.load_module_from_source(
            module_name=f"compile_error_from_source_{test_id}", source="bar"
        )

    # Loading a valid module must succeed
    module = device.load_module_from_source(
        module_name=f"module_from_source_{test_id}",
        source="""
        struct Foo {
            uint a;
        };

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main(uint3 tid: SV_DispatchThreadID, uniform Foo foo) { }
    """,
    )
    assert len(module.entry_points) == 1
    main = module.entry_point("main")
    assert main.name == "main"
    assert main.stage == spy.ShaderStage.compute
    assert main.layout.compute_thread_group_size == [1, 1, 1]

    # Check back refs to device and session are correct
    assert module.session == device.slang_session
    assert module.session.device == device


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_module_from_source_dedup(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    source = """
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main(uint3 tid: SV_DispatchThreadID) { }
    """

    # Loading the same module name with the same source twice must succeed.
    name = f"dedup_same_name_{test_id}"
    module_a = device.load_module_from_source(module_name=name, source=source)
    module_b = device.load_module_from_source(module_name=name, source=source)
    assert module_a is not None
    assert module_b is not None

    # Loading the same module name with different source must raise an exception.
    with pytest.raises(SlangCompileError, match="already loaded with different source"):
        device.load_module_from_source(
            module_name=name,
            source="""
                [shader("compute")]
                [numthreads(2, 2, 1)]
                void main(uint3 tid: SV_DispatchThreadID) { }
            """,
        )

    # Loading the same source with two different module names must succeed.
    module_c = device.load_module_from_source(module_name=f"dedup_name_1_{test_id}", source=source)
    module_d = device.load_module_from_source(module_name=f"dedup_name_2_{test_id}", source=source)
    assert module_c is not None
    assert module_d is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_program(device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)

    # Loading non-existing module must raise an exception
    with pytest.raises(Exception, match='Failed to load slang module "does_not_exist.slang"'):
        device.load_program(
            module_name="does_not_exist.slang",
            entry_point_names=["main"],
        )

    # Loading non-existing entry point must raise an exception
    with pytest.raises(Exception, match='Entry point "does_not_exist" not found'):
        device.load_program(
            module_name="test_print.slang",
            entry_point_names=["does_not_exist"],
        )

    # Compilation errors must raise an exception
    with pytest.raises(SlangCompileError, match="unexpected end of file, expected identifier"):
        device.load_program(
            module_name="test_shader_compile_error.slang",
            entry_point_names=["main"],
        )

    # Loading valid programs must succeed
    device.load_program(module_name="test_shader_foo.slang", entry_point_names=["main_a"])
    device.load_program(module_name="test_shader_foo.slang", entry_point_names=["main_b"])
    device.load_program(
        module_name="test_shader_foo.slang", entry_point_names=["main_vs", "main_fs"]
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compose_modules(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    session = device.slang_session

    # Load two separate modules
    module_a = device.load_module_from_source(
        module_name=f"compose_module_a_{test_id}",
        source="""
        [shader("compute")]
        [numthreads(1, 1, 1)]
        void entry_a(uint3 tid: SV_DispatchThreadID) { }
    """,
    )

    module_b = device.load_module_from_source(
        module_name=f"compose_module_b_{test_id}",
        source="""
        [shader("compute")]
        [numthreads(2, 2, 1)]
        void entry_b(uint3 tid: SV_DispatchThreadID) { }
    """,
    )

    # Verify source modules are not composed
    assert not module_a.is_composed
    assert not module_b.is_composed
    assert len(module_a.source_modules) == 0
    assert len(module_b.source_modules) == 0

    # Compose the modules
    composed = session.compose_modules(
        name=f"composed_module_{test_id}", modules=[module_a, module_b]
    )

    # Verify composed module properties
    assert composed.is_composed
    assert len(composed.source_modules) == 2
    assert composed.name == f"composed_module_{test_id}"

    # Verify entry points from both modules are accessible
    entry_points = composed.entry_points
    entry_point_names = [ep.name for ep in entry_points]
    assert "entry_a" in entry_point_names
    assert "entry_b" in entry_point_names

    # Verify individual entry point access
    entry_a = composed.entry_point("entry_a")
    assert entry_a.name == "entry_a"
    assert entry_a.stage == spy.ShaderStage.compute
    assert entry_a.layout.compute_thread_group_size == [1, 1, 1]

    entry_b = composed.entry_point("entry_b")
    assert entry_b.name == "entry_b"
    assert entry_b.stage == spy.ShaderStage.compute
    assert entry_b.layout.compute_thread_group_size == [2, 2, 1]

    # Verify layout exists
    layout = composed.layout
    assert layout is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_compose_modules_link_program(test_id: str, device_type: spy.DeviceType):
    device = helpers.get_device(type=device_type)
    session = device.slang_session

    # Load two modules with entry points
    module_a = device.load_module_from_source(
        module_name=f"link_module_a_{test_id}",
        source="""
        [shader("compute")]
        [numthreads(4, 4, 1)]
        void compute_a(uint3 tid: SV_DispatchThreadID) { }
    """,
    )

    module_b = device.load_module_from_source(
        module_name=f"link_module_b_{test_id}",
        source="""
        [shader("compute")]
        [numthreads(8, 8, 1)]
        void compute_b(uint3 tid: SV_DispatchThreadID) { }
    """,
    )

    # Compose the modules
    composed = session.compose_modules(
        name=f"link_composed_{test_id}", modules=[module_a, module_b]
    )

    # Link a program using entry points from the composed module
    entry_a = composed.entry_point("compute_a")
    entry_b = composed.entry_point("compute_b")

    program = session.link_program(modules=[composed], entry_points=[entry_a, entry_b])

    assert program is not None
    assert program.layout is not None
    assert len(program.layout.entry_points) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
