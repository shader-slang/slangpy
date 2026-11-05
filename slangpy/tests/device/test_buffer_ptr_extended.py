# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import nullcontext
from pathlib import Path
import hashlib
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
import slangpy as spy
from slangpy.testing import helpers


# Run the shared pointer contract on the default backends in ordinary PR CI.


SHADER = "test_buffer_ptr_extended.slang"
CANARY = 0xDEADBEEF


# Shared compilation and device helpers


@pytest.fixture(params=helpers.DEFAULT_DEVICE_TYPES)
def device(request: pytest.FixtureRequest) -> spy.Device:
    return helpers.get_device(request.param)


def _module(device: spy.Device) -> spy.SlangModule:
    return device.load_module(SHADER)


def _kernel(device: spy.Device, name: str) -> spy.ComputeKernel:
    return device.create_compute_kernel(program=device.load_program(SHADER, [name]))


def _compile(device: spy.Device, source: str, entry: str = "boundary") -> spy.ComputeKernel:
    module = device.load_module_from_source(
        "buffer_ptr_case_" + hashlib.sha256(source.encode()).hexdigest(), source
    )
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point(entry)])
    )
    _ = kernel.pipeline
    return kernel


@pytest.fixture
def slangc() -> Path:
    override = os.environ.get("SLANGC")
    if override:
        assert Path(override).is_file(), "SLANGC must name an existing executable"
        return Path(override)
    marker = Path(spy.__file__).parent / ".build_dir"
    if marker.is_file():
        build = Path(marker.read_text().strip()).parent
        candidate = build / "_deps/slang-src/bin" / ("slangc.exe" if os.name == "nt" else "slangc")
        if candidate.is_file():
            return candidate
    executable = shutil.which("slangc")
    if executable:
        return Path(executable)
    pytest.skip("Slang CLI required for emitted-code tests (set SLANGC)")


def _emit(
    slangc: Path,
    source: Path,
    output: Path,
    entry: str,
    target: str,
    includes: list[Path],
    defines: list[str],
) -> str:
    command = [
        str(slangc),
        str(source),
        "-entry",
        entry,
        "-stage",
        "compute",
        "-target",
        target,
        "-O3",
        "-o",
        str(output),
    ]
    for path in includes:
        command += ["-I", str(path)]
    command += ["-D" + define for define in defines]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return output.read_text()


# Byte access, subscripts and explicit alignment


@pytest.mark.parametrize("access", ["Read", "ReadWrite"])
@pytest.mark.parametrize("load_alignment", [0, 4])
def test_generic_subscript(device: spy.Device, access: str, load_alignment: int) -> None:
    writable = access == "ReadWrite"
    update = "p[1] = read_sum(p);" if writable else ""
    source = f"""
import bufferptr;
uint read_sum<Access A, let alignment : uint>(BufferPtr<uint,A,alignment> p)
{{
    return p[0] + p[1];
}}
[shader("compute")][numthreads(1,1,1)]
void generic_subscript(uint64_t bits, RWStructuredBuffer<uint> result)
{{
    let p = BufferPtr<uint,Access.{access},{load_alignment}>(bits);
    result[0] = read_sum(p);
    {update}
    result[1] = read_sum(p);
}}
"""
    module = device.load_module_from_source(f"generic_subscript_{access}_{load_alignment}", source)
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point("generic_subscript")])
    )
    buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.array([99, 10, 20, 99], dtype=np.uint32),
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=8)
    ptr = (
        spy.to_rw_buffer_ptr(buffer, byte_offset=4)
        if writable
        else spy.to_buffer_ptr(buffer, byte_offset=4)
    )
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(
        buffer,
        spy.ResourceState.unordered_access if writable else spy.ResourceState.shader_resource,
    )
    kernel.dispatch((1, 1, 1), command_encoder=encoder, bits=ptr.data, result=result)
    device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [30, 40 if writable else 30])
    np.testing.assert_array_equal(
        buffer.to_numpy().view(np.uint32), [99, 10, 30 if writable else 20, 99]
    )


@pytest.mark.parametrize("src_byte_offset,dst_byte_offset", [(16, 32), (48, 64)])
def test_interior_access(device: spy.Device, src_byte_offset: int, dst_byte_offset: int) -> None:
    src_data = np.arange(64, dtype=np.uint32)
    dst_data = np.arange(1000, 1064, dtype=np.uint32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=src_data)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=dst_data)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=16 * 16)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, "interior_access").dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=spy.to_buffer_ptr(src, byte_offset=src_byte_offset, alignment=16),
        dst=spy.to_rw_buffer_ptr(dst, byte_offset=dst_byte_offset, alignment=16),
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    src_index = src_byte_offset // 4
    dst_index = dst_byte_offset // 4
    read_indices = [0, 1, 0, 1, 1, 1, 1, 1]
    expected_reads = np.concatenate(
        [
            data[index + i * 4 : index + (i + 1) * 4]
            for data, index in [(src_data, src_index), (dst_data, dst_index)]
            for i in read_indices
        ]
    )
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), expected_reads)
    a = src_data[src_index : src_index + 4]
    b = src_data[src_index + 4 : src_index + 8]
    expected = dst_data.copy()
    expected[dst_index : dst_index + 32] = np.concatenate(
        [a, b, a + 100, b + 200, a + 300, b + 400, a + 500, b + 600]
    )
    # Check the complete allocation so an omitted/doubled base also corrupts canaries.
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


def test_pointer_alignment(device: spy.Device) -> None:
    src = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.arange(16, dtype=np.uint32)
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=7 * 4)
    _kernel(device, "pointer_alignment").dispatch(
        (1, 1, 1), src=spy.to_buffer_ptr(src, alignment=16), result=result
    )
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [3, 3, 3, 2, 2, 1, 1])


@pytest.mark.parametrize("copy_kernel", ["copy_values", "plain_values"])
def test_writable_aligned_and_packed_access(device: spy.Device, copy_kernel: str) -> None:
    data = np.arange(32, dtype=np.float32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=data)
    expected = np.full(32, -99, dtype=np.float32)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, copy_kernel).dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=spy.to_buffer_ptr(src),
        dst=spy.to_rw_buffer_ptr(dst),
    )
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, "update_values").dispatch(
        (1, 1, 1), command_encoder=encoder, dst=spy.to_rw_buffer_ptr(dst)
    )
    device.submit_command_buffer(encoder.finish())
    device.wait()
    expected[4:8] = data[4:8]
    expected[16:18] = data[16:18]
    expected[9:12] = data[9:12] + [1, 2, 3]
    expected[13:15] = data[13:15]
    np.testing.assert_array_equal(dst.to_numpy().view(np.float32), expected)


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 is not selected"
)
def test_plain_d3d_byte_offsets() -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    data = np.arange(1, 17, dtype=np.uint32)
    # A nontrivial finite double spanning two words verifies the whole load/store.
    data[6:8] = np.array([123.5], dtype=np.float64).view(np.uint32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=data)
    expected = np.full(16, 0xDEADBEEF, dtype=np.uint32)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, "plain_d3d_offset").dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=spy.to_buffer_ptr(src),
        dst=spy.to_rw_buffer_ptr(dst),
    )
    device.submit_command_buffer(encoder.finish())
    expected[1:5] = data[1:5]
    expected[6:8] = data[6:8]
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


@pytest.mark.parametrize(
    "element_type,value,words",
    [
        ("uint64_t", "42", [42]),
        ("uint64_t", "uint64_t(42)", [42]),
        ("uint64_t", "42ull", [42]),
        ("uint64_t", "0x1122334455667788ull", [0x1122334455667788]),
        ("uint64_t", "0xffffffffffffffffull", [0xFFFFFFFFFFFFFFFF]),
        ("int64_t", "42ll", [42]),
        ("int64_t", "-0x1122334455667788ll", [0xEEDDCCBBAA998878]),
        ("uint64_t", "input_value", [0x8877665544332211]),
        ("int64_t", "int64_t(input_value)", [0x8877665544332211]),
        (
            "vector<uint64_t,2>",
            "vector<uint64_t,2>(42ull,0x1122334455667788ull)",
            [42, 0x1122334455667788],
        ),
        (
            "vector<int64_t,2>",
            "vector<int64_t,2>(42ll,-0x1122334455667788ll)",
            [42, 0xEEDDCCBBAA998878],
        ),
        ("IntegerRecord", "{42ull,0x1122334455667788ull}", [42, 0x1122334455667788]),
        ("BufferPtr<>", "BufferPtr<>(0x1122334455667788ull)", [0x1122334455667788]),
        ("BufferPtr<>", "BufferPtr<>()", [0xFFFFFFFFFFFFFFFF]),
        ("PointerRecord", "{BufferPtr<>(0x1122334455667788ull),42ull}", [0x1122334455667788, 42]),
    ],
)
def test_integer_and_pointer_stores(
    device: spy.Device, element_type: str, value: str, words: list[int]
) -> None:
    source = f"""
import bufferptr;
struct IntegerRecord {{ uint64_t small; uint64_t wide; }}
struct PointerRecord {{ BufferPtr<> pointer; uint64_t count; }}
[shader("compute")][numthreads(1,1,1)]
void integer_pointer_stores(RWByteAddressBuffer raw, uniform RWBufferPtr<> dst, uniform uint64_t input_value)
{{
    {element_type} value = {value};
    raw.Store<{element_type}>(16, value);
    dst.as<{element_type}>()[0] = value;
}}
"""
    module = device.load_module_from_source(
        "integer_stores_" + hashlib.sha256(source.encode()).hexdigest(), source
    )
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point("integer_pointer_stores")])
    )
    expected = np.full(16, 0xDEADBEEF, dtype=np.uint32)
    raw = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(raw, spy.ResourceState.unordered_access)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    kernel.dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        raw=raw,
        dst=spy.to_rw_buffer_ptr(dst, byte_offset=16, alignment=16),
        input_value=0x8877665544332211,
    )
    device.submit_command_buffer(encoder.finish())
    payload = np.array(words, dtype=np.uint64).view(np.uint32)
    expected[4 : 4 + len(payload)] = payload
    # Compare standard raw-buffer stores with wrapper stores, including surrounding canaries.
    np.testing.assert_array_equal(raw.to_numpy().view(np.uint32), expected)
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


def test_aligned_records(device: spy.Device) -> None:
    data = np.arange(32, dtype=np.uint32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=data)
    expected = np.full(32, 0xDEADBEEF, dtype=np.uint32)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    _kernel(device, "aligned_records").dispatch(
        (1, 1, 1),
        src=spy.to_buffer_ptr(src, alignment=16),
        dst=spy.to_rw_buffer_ptr(dst, alignment=16),
    )
    expected[4:10] = data[4:10]
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


# Pointer arithmetic and element stride


def test_pointer_operations(device: spy.Device) -> None:
    src = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([10, 20, 30, 40], dtype=np.uint32)
    )
    dst = device.create_buffer(
        usage=spy.BufferUsage.unordered_access, data=np.zeros(8, dtype=np.uint32)
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=18 * 8)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, "pointer_operations").dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=spy.to_buffer_ptr(src),
        dst=spy.to_rw_buffer_ptr(dst),
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(
        result.to_numpy().view(np.uint64),
        [20, 10, 40, 20, 30, 40, 30, 30, 30, 1, 30, 1, 1, 1, 50, 1, 1, 42],
    )
    expected = np.zeros(8, dtype=np.uint32)
    expected[0] = 30
    expected[1] = 20
    expected.view(np.uint64)[2] = 42
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


@pytest.mark.parametrize("name,elements", [("vector3", 3), ("matrix3x4", 12), ("matrix4x3", 12)])
def test_vector_matrix_stride(device: spy.Device, name: str, elements: int) -> None:
    data = np.arange(1, 3 * elements + 1, dtype=np.float32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=data)
    structured = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=data, struct_size=elements * 4
    )
    dst = device.create_buffer(
        usage=spy.BufferUsage.unordered_access,
        data=np.full(3 * elements + 4, -99, dtype=np.float32),
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=5 * 8)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _kernel(device, "stride_" + name).dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=spy.to_buffer_ptr(src),
        dst=spy.to_rw_buffer_ptr(dst),
        structured=structured,
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    values = result.to_numpy().view(np.uint64)
    assert values[0] == elements * 4
    np.testing.assert_array_equal(values[1:], np.full(4, values[1], dtype=np.uint64))
    expected = np.concatenate(
        [np.tile(data[elements : 2 * elements], 3), np.full(4, -99, dtype=np.float32)]
    )
    np.testing.assert_array_equal(dst.to_numpy().view(np.float32), expected)


def test_nested_padded_values(device: spy.Device) -> None:
    module = device.load_module(SHADER)
    layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<NestedValues>")
    ).element_type_layout
    stride = layout.stride
    sentinel = np.full(4 * stride // 4, CANARY, dtype=np.uint32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=sentinel)
    cursor = spy.BufferCursor(layout, src)
    cursor.load()
    for record in (1, 2):
        for index in (0, 1):
            cursor[record]["items"][index]["key"].write(1000 * record + index)
            cursor[record]["items"][index]["value"].write(100 * record + index)
            cursor[record]["positions"][index].write(
                spy.float3(10 * record + index, 20 * record + index, 30 * record + index)
            )
        cursor[record]["tail"].write(10000 * record)
    cursor.apply()
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=sentinel)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=8 * 4)
    _kernel(device, "nested_values").dispatch(
        (4, 1, 1),
        src=spy.to_buffer_ptr(src, stride),
        dst=spy.to_rw_buffer_ptr(dst, stride),
        result=result,
    )
    expected = [1130 + 3 * (i % 2) for i in range(4)] + [22040 + 2 * (i % 2) for i in range(4)]
    observed_result = result.to_numpy().view(np.uint32)
    # Base-record reads and out-of-range writes must remain ordinary failures.
    np.testing.assert_array_equal(observed_result[:4], expected[:4])
    observed = dst.to_numpy().view(np.uint32)
    np.testing.assert_array_equal(observed[: stride // 4], sentinel[: stride // 4])
    np.testing.assert_array_equal(observed[2 * stride // 4 :], sentinel[2 * stride // 4 :])
    np.testing.assert_array_equal(observed_result, expected)
    readback = spy.BufferCursor(layout, dst)
    readback.load()
    for index in (0, 1):
        assert readback[1]["items"][index]["key"].read() == 2000 + index
        assert readback[1]["items"][index]["value"].read() == 200 + index
        np.testing.assert_array_equal(
            readback[1]["positions"][index].read(), [20 + index, 40 + index, 60 + index]
        )
    assert readback[1]["tail"].read() == 20000


@pytest.mark.parametrize(
    "typ,index", [("PaddedInner", 0), ("UnpaddedNode", 1), ("NestedValues", 2)]
)
def test_padded_pointer_stride(device: spy.Device, typ: str, index: int) -> None:
    module = device.load_module(SHADER)
    layout = module.layout.get_type_layout(
        module.layout.find_type_by_name(f"StructuredBuffer<{typ}>")
    ).element_type_layout
    storage = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=1024)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=7 * 8)
    _kernel(device, "stride_metadata").dispatch(
        (1, 1, 1), root=spy.to_buffer_ptr(storage, 16), result=result
    )
    observed = result.to_numpy().view(np.uint64)
    # Native pointer stepping is an independent baseline for the selected layout.
    if device.info.type != spy.DeviceType.d3d12:
        assert observed[index + 3] == layout.stride
    # Check address arithmetic without dereferencing the potentially misaligned pointer.
    assert observed[index] == layout.stride


@pytest.mark.parametrize("typ,stride_index", [("PaddedInner", 0), ("NestedValues", 2)])
@pytest.mark.parametrize("load_alignment", [0, 8])
def test_padded_backward_access(
    device: spy.Device, typ: str, stride_index: int, load_alignment: int
) -> None:
    module = device.load_module(SHADER)
    layout = module.layout.get_type_layout(
        module.layout.find_type_by_name(f"StructuredBuffer<{typ}>")
    ).element_type_layout
    stride = layout.stride
    sentinel = np.full(7 * stride // 4, CANARY, dtype=np.uint32)
    src = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=sentinel)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=sentinel)
    root = spy.to_buffer_ptr(src, 3 * stride, alignment=8)
    metadata = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=7 * 8)
    _kernel(device, "stride_metadata").dispatch((1, 1, 1), root=root, result=metadata)
    # Reject a broken stride before dereferencing possibly underaligned records.
    assert metadata.to_numpy().view(np.uint64)[stride_index] == stride

    cursor = spy.BufferCursor(layout, src)
    cursor.load()
    for record in range(1, 5):
        if typ == "PaddedInner":
            cursor[record]["key"].write((record << 40) | record)
            cursor[record]["value"].write(record * 100)
        else:
            for index in (0, 1):
                cursor[record]["items"][index]["key"].write((record << 40) | index)
                cursor[record]["items"][index]["value"].write(record * 100 + index)
                cursor[record]["positions"][index].write(
                    spy.float3(record * 10 + index, record * 20 + index, record * 30 + index)
                )
            cursor[record]["tail"].write(record * 1000)
    cursor.apply()
    source = (
        Path(__file__).with_suffix(".slang").read_text()
        + f"""
[shader("compute")][numthreads(1,1,1)]
void boundary(uniform BufferPtr<> src, uniform RWBufferPtr<> dst) {{
    let p = src.as<{typ}>().with_load_alignment<{load_alignment}>();
    let output = dst.as<{typ}>().with_load_alignment<{load_alignment}>();
    var previous = p;
    --previous;
    output[-2] = p[-1];
    output[-1] = (p - uint(1))[0];
    output[0] = previous[int64_t(-1)];
    ++previous;
    output[1] = previous[1];
}}
"""
    )
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(src, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    _compile(device, source).dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        src=root,
        dst=spy.to_rw_buffer_ptr(dst, 3 * stride, alignment=8),
    )
    device.submit_command_buffer(encoder.finish())
    observed = dst.to_numpy().view(np.uint32)
    np.testing.assert_array_equal(observed[: stride // 4], sentinel[: stride // 4])
    np.testing.assert_array_equal(observed[5 * stride // 4 :], sentinel[5 * stride // 4 :])
    readback = spy.BufferCursor(layout, dst)
    readback.load()
    for target, record in enumerate((2, 2, 1, 4), start=1):
        if typ == "PaddedInner":
            assert readback[target]["key"].read() == (record << 40) | record
            assert readback[target]["value"].read() == record * 100
        else:
            for index in (0, 1):
                assert readback[target]["items"][index]["key"].read() == (record << 40) | index
                assert readback[target]["items"][index]["value"].read() == record * 100 + index
                np.testing.assert_array_equal(
                    readback[target]["positions"][index].read(),
                    [record * 10 + index, record * 20 + index, record * 30 + index],
                )
            assert readback[target]["tail"].read() == record * 1000


# Conversions, comparisons and rejected operations


@pytest.mark.parametrize(
    "pointees", [("uint", "uint"), ("uint", "void"), ("void", "uint"), ("void", "void")]
)
@pytest.mark.parametrize(
    "accesses",
    [("Read", "Read"), ("ReadWrite", "ReadWrite"), ("Read", "ReadWrite"), ("ReadWrite", "Read")],
)
def test_pointer_equality(
    device: spy.Device, pointees: tuple[str, str], accesses: tuple[str, str]
) -> None:
    if device.info.type == spy.DeviceType.d3d12 and accesses[0] != accesses[1]:
        pytest.skip("D3D SRV/UAV comparisons are tested as compile-time rejections")
    a_type = f"BufferPtr<{pointees[0]},Access.{accesses[0]},4>"
    b_type = f"BufferPtr<{pointees[1]},Access.{accesses[1]},16>"
    checks = [
        ("a == b", 1),
        ("a != b", 0),
        ("b == a", 1),
        ("b != a", 0),
        ("a == different", 0),
        ("a != different", 1),
        ("different == a", 0),
        ("different != a", 1),
        ("a == other_buffer", 0),
        ("a != other_buffer", 1),
        ("other_buffer == a", 0),
        ("other_buffer != a", 1),
        ("a == b_null", 0),
        ("a != b_null", 1),
        ("a_null == b", 0),
        ("a_null != b", 1),
        ("a_null == b_null", 1),
        ("a_null != b_null", 0),
        ("a == nullptr", 0),
        ("nullptr == a", 0),
        ("b_null == nullptr", 1),
        ("nullptr == b_null", 1),
        ("a != nullptr", 1),
        ("nullptr != a", 1),
        ("b_null != nullptr", 0),
        ("nullptr != b_null", 0),
        ("bool(a)", 1),
        ("bool(a_null)", 0),
        ("a.bits() == bits", 1),
        ("a.bits() == other_bits", 0),
        ("!a", 0),
        ("!a_null", 1),
        ("a ? 1 : 0", 1),
        ("a_null ? 1 : 0", 0),
        ("a && b", 1),
        ("a_null || b_null", 0),
        ("a < later", 1),
        ("a <= later", 1),
        ("later > a", 1),
        ("later >= a", 1),
    ]
    stores = "\n".join(f"result[{i}] = {expr};" for i, (expr, _) in enumerate(checks))
    source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void pointer_equality(uniform uint64_t bits, uniform uint64_t other_bits,
                      uniform uint64_t other_buffer_bits, RWStructuredBuffer<uint> result)
{{
    let a = {a_type}(bits);
    let b = {b_type}(bits);
    let different = {b_type}(other_bits);
    let other_buffer = {b_type}(other_buffer_bits);
    let later = {a_type}(other_bits);
    {a_type} a_null = nullptr;
    {b_type} b_null = nullptr;
    {stores}
    if (a) result[{len(checks)}] = 1; else result[{len(checks)}] = 0;
    if (b_null) result[{len(checks)+1}] = 1; else result[{len(checks)+1}] = 0;
}}
"""
    module = device.load_module_from_source(
        "pointer_equality_" + hashlib.sha256(source.encode()).hexdigest(), source
    )
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point("pointer_equality")])
    )
    buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access, size=64
    )
    other_buffer = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access, size=64
    )
    factory = spy.to_buffer_ptr if accesses[0] == "Read" else spy.to_rw_buffer_ptr
    result = device.create_buffer(
        usage=spy.BufferUsage.unordered_access, size=(len(checks) + 2) * 4
    )
    kernel.dispatch(
        (1, 1, 1),
        bits=factory(buffer, byte_offset=16).data,
        other_bits=factory(buffer, byte_offset=32).data,
        other_buffer_bits=factory(other_buffer, byte_offset=16).data,
        result=result,
    )
    np.testing.assert_array_equal(
        result.to_numpy().view(np.uint32), [value for _, value in checks] + [1, 0]
    )


@pytest.mark.parametrize("op", ["==", "!="])
@pytest.mark.parametrize(
    "pointees,accesses,d3d_only",
    [
        (("uint", "float"), ("Read", "Read"), False),
        (("float", "uint"), ("ReadWrite", "Read"), False),
        (("uint", "uint"), ("Read", "ReadWrite"), True),
        (("uint", "void"), ("ReadWrite", "Read"), True),
        (("void", "uint"), ("Read", "ReadWrite"), True),
        (("void", "void"), ("ReadWrite", "Read"), True),
    ],
)
def test_rejected_pointer_equality(
    device: spy.Device,
    op: str,
    pointees: tuple[str, str],
    accesses: tuple[str, str],
    d3d_only: bool,
) -> None:
    if d3d_only and device.info.type != spy.DeviceType.d3d12:
        pytest.skip("Native cross-access comparisons are tested as supported")
    a_type = f"BufferPtr<{pointees[0]},Access.{accesses[0]},4>"
    b_type = f"BufferPtr<{pointees[1]},Access.{accesses[1]},16>"
    source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void rejected_pointer_equality(uniform uint64_t bits, RWStructuredBuffer<uint> result)
{{
    result[0] = {a_type}(bits) {op} {b_type}(bits);
}}
"""
    message = "matching SRV/UAV access" if d3d_only else "matching pointee types or a void pointee"
    with pytest.raises((spy.SlangCompileError, RuntimeError), match=message):
        module = device.load_module_from_source(
            "rejected_equality_" + hashlib.sha256(source.encode()).hexdigest(), source
        )
        kernel = device.create_compute_kernel(
            program=device.link_program([module], [module.entry_point("rejected_pointer_equality")])
        )
        _ = kernel.pipeline


@pytest.mark.parametrize("family", ["BufferPtr", "RWBufferPtr"])
@pytest.mark.parametrize(
    "operand",
    [
        "0",
        "int(number)",
        "uint(number)",
        "int64_t(number)",
        "number",
        "float(number)",
        "double(number)",
    ],
)
def test_rejected_numeric_equality(device: spy.Device, family: str, operand: str) -> None:
    for op in ("==", "!="):
        for left, right in (("p", operand), (operand, "p")):
            source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void numeric_equality(uniform {family}<uint> p, uniform uint64_t number,
                      RWStructuredBuffer<uint> result) {{
    result[0] = {left} {op} {right};
}}
"""
            with pytest.raises(
                (spy.SlangCompileError, RuntimeError),
                match="BufferPtr cannot be compared with a number",
            ):
                module = device.load_module_from_source(
                    "numeric_equality_" + hashlib.sha256(source.encode()).hexdigest(), source
                )
                kernel = device.create_compute_kernel(
                    program=device.link_program([module], [module.entry_point("numeric_equality")])
                )
                _ = kernel.pipeline


@pytest.mark.parametrize("family", ["BufferPtr", "RWBufferPtr"])
@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_rejected_numeric_ordering(device: spy.Device, family: str, op: str) -> None:
    # Ordering already rejects numbers without an added overload. Preserve that boundary.
    for left, right in (("p", "number"), ("number", "p")):
        source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void numeric_ordering(uniform {family}<uint> p, uniform uint64_t number,
                      RWStructuredBuffer<uint> result) {{
    result[0] = {left} {op} {right};
}}
"""
        with pytest.raises((spy.SlangCompileError, RuntimeError), match="error\\[E30019\\]"):
            module = device.load_module_from_source(
                "numeric_ordering_" + hashlib.sha256(source.encode()).hexdigest(), source
            )
            kernel = device.create_compute_kernel(
                program=device.link_program([module], [module.entry_point("numeric_ordering")])
            )
            _ = kernel.pipeline


@pytest.mark.parametrize(
    "backend",
    [
        backend
        for backend in helpers.DEFAULT_DEVICE_TYPES
        if backend in (spy.DeviceType.vulkan, spy.DeviceType.cuda, spy.DeviceType.metal)
    ],
)
@pytest.mark.parametrize("family", ["Ptr", "BufferPtr"])
@pytest.mark.parametrize("c_style", [False, True])
def test_native_access_casts(backend: spy.DeviceType, family: str, c_style: bool) -> None:
    device = helpers.get_device(backend)
    read_type = f"{family}<uint,Access.Read>"
    write_type = f"{family}<uint,Access.ReadWrite>"
    void_type = f"{family}<void,Access.Read>"

    def cast(typ: str, value: str) -> str:
        return f"({typ})({value})" if c_style else f"{typ}({value})"

    source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void access_casts(uint64_t bits, RWStructuredBuffer<uint64_t> result)
{{
    let writable = {write_type}(bits);
    let readable = {cast(read_type, "writable")};
    result[0] = readable[0];
    result[1] = readable[1];
    result[2] = uint64_t(readable) == bits;
    let untyped = {cast(void_type, "writable")};
    let restored = {cast(write_type, "untyped")};
    result[3] = uint64_t(restored) == bits;
    restored[1] = readable[0] + readable[1];
}}
"""
    module = device.load_module_from_source(f"native_access_casts_{family}_{c_style}", source)
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point("access_casts")])
    )
    # Read-to-write is tested only on memory that is actually writable.
    buffer = device.create_buffer(
        usage=spy.BufferUsage.unordered_access, data=np.array([99, 10, 20, 99], dtype=np.uint32)
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=4 * 8)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(buffer, spy.ResourceState.unordered_access)
    kernel.dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        bits=spy.to_rw_buffer_ptr(buffer, byte_offset=4).data,
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(result.to_numpy().view(np.uint64), [10, 20, 1, 1])
    np.testing.assert_array_equal(buffer.to_numpy().view(np.uint32), [99, 10, 30, 99])


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 is not selected"
)
@pytest.mark.parametrize(
    "body",
    [
        "BufferPtr<uint> p = BufferPtr<uint>(RWBufferPtr<uint>(bits)); result[0] = p[0];",
        "RWBufferPtr<uint> p = RWBufferPtr<uint>(BufferPtr<uint>(bits)); p[0] = 1;",
        "BufferPtr<uint> p = (BufferPtr<uint>)RWBufferPtr<uint>(bits); result[0] = p[0];",
        "RWBufferPtr<uint> p = (RWBufferPtr<uint>)BufferPtr<uint>(bits); p[0] = 1;",
    ],
)
def test_d3d_rejected_access_casts(body: str) -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    source = "import bufferptr; RWStructuredBuffer<uint> result; "
    source += '[shader("compute")][numthreads(1,1,1)] void main(uint64_t bits) {' + body + "}"
    with pytest.raises((spy.SlangCompileError, RuntimeError), match="D3D12 BufferPtr access casts"):
        module = device.load_module_from_source(
            "d3d_rejected_access_casts_" + hashlib.sha256(source.encode()).hexdigest(), source
        )
        kernel = device.create_compute_kernel(
            program=device.link_program([module], [module.entry_point("main")])
        )
        _ = kernel.pipeline


def test_rejected_implicit_pointee_cast(device: spy.Device) -> None:
    source = """
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void boundary(uniform BufferPtr<uint> p, RWStructuredBuffer<float> result) {
    BufferPtr<float> q = p;
    result[0] = q[0];
}
"""
    with pytest.raises((spy.SlangCompileError, RuntimeError), match="error\\[E30019\\]"):
        _compile(device, source)
    _compile(device, source.replace("q = p;", "q = BufferPtr<float>(p);"))


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (
            "BufferPtr<> p; p.store<uint>(0, 1);",
            "'store' is not a member of 'BufferPtr<void, Access.Read, 0>'",
        ),
        ("BufferPtr<uint> p; p[0] = 1;", "error[E30011]: left of '=' is not an l-value"),
        (
            "BufferPtr<uint,Access.ReadWrite> p = BufferPtr<uint,Access.Read>();",
            "error[E30019]: type mismatch in expression",
        ),
        (
            "BufferPtr<uint,Access.Read> p = BufferPtr<uint,Access.ReadWrite>();",
            "error[E30019]: type mismatch in expression",
        ),
        (
            "BufferPtr<uint,Access.Read,3> p = BufferPtr<uint,Access.Read,3>(bits); result[0] = p[0];",
            "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16",
        ),
        (
            "let p = BufferPtr<uint,Access.Read,1>(bits); result[0] = uint(p.bits());",
            "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16",
        ),
        (
            "let p = BufferPtr<uint,Access.Read,2>(bits); result[0] = uint(p.bits());",
            "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16",
        ),
        (
            "BufferPtr<void,Access.Read,4> p; BufferPtr<float4,Access.Read,16> q = p.as<float4>();",
            "error[E30019]: type mismatch in expression",
        ),
        (
            "BufferPtr<> p = BufferPtr<>(bits); result[0] = p.load_aligned<uint,2>();",
            "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16",
        ),
        (
            "BufferPtr<> p = BufferPtr<>(bits); result[0] = p.load_aligned<uint,0>();",
            "Portable buffer access requires at least four-byte load_alignment",
        ),
        (
            "BufferPtr<> p = BufferPtr<>(bits); result[0] = p.load<uint16_t>();",
            "Portable access size must be divisible by four",
        ),
        (
            "result[0] = invalid_alignment.load_aligned<uint,4>();",
            "BufferPtr load_alignment must be 0 (plain access), 4, 8 or 16",
        ),
        pytest.param("RWBufferPtr<uint>(bits)[0] = 1;", None, id="valid-write-control"),
    ],
)
def test_operation_validation(device: spy.Device, body: str, diagnostic: str | None) -> None:
    source = "import bufferptr; RWStructuredBuffer<uint> result; "
    source += (
        '[shader("compute")][numthreads(1,1,1)] void operation_validation(uniform uint64_t bits'
    )
    if "invalid_alignment" in body:
        source += ", BufferPtr<uint,Access.Read,6> invalid_alignment"
    source += ") {" + body + "}"
    # A valid control traverses the same pipeline path; unrelated backend failures
    # must not satisfy a rejection test. Static assertions can surface at codegen.
    expectation = (
        pytest.raises((spy.SlangCompileError, RuntimeError), match=re.escape(diagnostic))
        if diagnostic is not None
        else nullcontext()
    )
    with expectation:
        module = device.load_module_from_source(
            "operation_validation_" + hashlib.sha256(source.encode()).hexdigest(), source
        )
        kernel = device.create_compute_kernel(
            program=device.link_program([module], [module.entry_point("operation_validation")])
        )
        _ = kernel.pipeline


# Mutation and aliasing against Ptr and RWStructuredBuffer


@pytest.mark.parametrize("access", ["Read", "ReadWrite"])
@pytest.mark.parametrize(
    "operation", ["out", "inout", "field_out", "field_inout", "field_assign", "field_compound"]
)
def test_mutation_boundaries(device: spy.Device, access: str, operation: str) -> None:
    expression = "p[0].x" if operation.startswith("field_") else "p[0]"
    typ = "Record" if operation.startswith("field_") else "uint"
    if operation.endswith("out"):
        statement = f"{'replace_value' if operation in ('out', 'field_out') else 'increment'}({expression});"
        diagnostic = "error[E30047]"
    else:
        statement = f"{expression} {'+=' if operation == 'field_compound' else '='} 7;"
        diagnostic = "error[E30047]" if operation == "field_compound" else "error[E30011]"
    source = f"""
import bufferptr;
struct Record {{ uint x; uint y; }}
void replace_value(out uint value) {{ value = 7; }}
void increment(inout uint value) {{ value += 7; }}
[shader("compute")][numthreads(1,1,1)]
void boundary(uniform uint64_t bits) {{
    let p = BufferPtr<{typ},Access.{access}>(bits);
    {statement}
}}
"""
    if access == "Read":
        with pytest.raises((spy.SlangCompileError, RuntimeError), match=re.escape(diagnostic)):
            _compile(device, source)
    else:
        kernel = _compile(device, source)
        data = np.array([CANARY, CANARY, 10, 20, CANARY, CANARY], dtype=np.uint32)
        buffer = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=data)
        kernel.dispatch((1, 1, 1), bits=spy.to_rw_buffer_ptr(buffer, 8).data)
        data[2] = 17 if operation in ("inout", "field_inout", "field_compound") else 7
        np.testing.assert_array_equal(buffer.to_numpy().view(np.uint32), data)


@pytest.mark.parametrize("family", ["wrapper", "structured", "native"])
@pytest.mark.parametrize(
    "suffix,expected", [("", [11, 99]), ("_field", [7, 20]), ("_compound", [17, 20])]
)
def test_sequential_field_aliases(
    device: spy.Device, family: str, suffix: str, expected: list[int]
) -> None:
    if family == "native" and device.info.type == spy.DeviceType.d3d12:
        pytest.skip("D3D12 has no native Ptr baseline")
    kernel = _kernel(device, "alias_" + family + suffix)
    buffer = device.create_buffer(
        usage=spy.BufferUsage.unordered_access,
        data=np.array([10, 20], dtype=np.uint32),
        struct_size=8 if family == "structured" else 0,
    )
    if family == "structured":
        kernel.dispatch((1, 1, 1), vars={"records": buffer})
    else:
        kernel.dispatch((1, 1, 1), bits=spy.to_rw_buffer_ptr(buffer).data)
    np.testing.assert_array_equal(buffer.to_numpy().view(np.uint32), expected)


# Pointer tables, traversal and dynamic dispatch


def test_divergent_raw_table(device: spy.Device) -> None:
    values = np.arange(8 * 8, dtype=np.uint32).reshape(8, 8)
    buffers = [
        device.create_buffer(usage=spy.BufferUsage.shader_resource, data=row) for row in values
    ]
    pointers = np.array([spy.to_buffer_ptr(buffer, 8).data for buffer in buffers], dtype=np.uint64)
    table = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=pointers)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=129 * 4)
    encoder = device.create_command_encoder()
    for buffer in buffers:
        encoder.set_buffer_state(buffer, spy.ResourceState.shader_resource)
    _kernel(device, "divergent").dispatch(
        (129, 1, 1), command_encoder=encoder, pointer_table=table, result=result, count=129
    )
    device.submit_command_buffer(encoder.finish())
    device.wait()
    indices = np.arange(129)
    np.testing.assert_array_equal(
        result.to_numpy().view(np.uint32), values[indices % 8, indices % 4 + 2]
    )


def test_gpu_pointer_chain_and_null_flow(device: spy.Device) -> None:
    module = device.load_module(SHADER)
    layout = module.layout.get_type_layout(
        module.layout.find_type_by_name("StructuredBuffer<ChainNode>")
    ).element_type_layout
    stride = layout.stride
    sentinel = np.full(5 * stride // 4, CANARY, dtype=np.uint32)
    nodes = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access, data=sentinel
    )
    a = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([99, 99, 3, 7, 99], dtype=np.uint32)
    )
    b = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([99, 99, 99, 11, 99], dtype=np.uint32)
    )
    root = spy.to_buffer_ptr(nodes, stride)
    metadata = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=7 * 8)
    _kernel(device, "stride_metadata").dispatch((1, 1, 1), root=root, result=metadata)
    observed_stride = metadata.to_numpy().view(np.uint64)
    assert (
        observed_stride[6] == stride
    ), f"Unsafe chain stride: {observed_stride.tolist()}, reflected node stride={stride}"
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(nodes, spy.ResourceState.unordered_access)
    _kernel(device, "build_chain").dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        nodes_ro=root,
        nodes_rw=spy.to_rw_buffer_ptr(nodes, stride),
        a=spy.to_buffer_ptr(a, 8),
        b=spy.to_buffer_ptr(b, 12),
    )
    # The RO encoding is stored, not dereferenced, during setup. Transition before traversal.
    encoder.set_buffer_state(nodes, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(a, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(b, spy.ResourceState.shader_resource)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=65 * 8)
    _kernel(device, "follow_chain").dispatch(
        (65, 1, 1), command_encoder=encoder, root=root, result=result
    )
    device.submit_command_buffer(encoder.finish())
    expected = np.array([[81, 3], [68, 2], [0, 0]], dtype=np.uint32)[np.arange(65) % 3]
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32).reshape(-1, 2), expected)
    observed = nodes.to_numpy().view(np.uint32)
    np.testing.assert_array_equal(observed[: stride // 4], sentinel[: stride // 4])
    np.testing.assert_array_equal(observed[4 * stride // 4 :], sentinel[4 * stride // 4 :])


def test_dynamic_record(device: spy.Device) -> None:
    module = _module(device)
    typ = module.layout.find_type_by_name("StructuredBuffer<DynamicPointerRecord>")
    layout = module.layout.get_type_layout(typ).element_type_layout
    data = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([10, 20, 30, 40], dtype=np.uint32)
    )
    record = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=128)
    cursor = spy.BufferCursor(layout, record)
    cursor[0]["tag"].write(123)
    cursor[0]["ptr"].write(spy.to_buffer_ptr(data, 8))
    cursor[0]["tail"].write(456)
    cursor.apply()
    composed = device.slang_session.compose_modules(
        "buffer_ptr_dynamic",
        [module],
        [spy.TypeConformance("IPointerRecord", "DynamicPointerRecord", 0)],
    )
    program = device.link_program([composed], [composed.entry_point("read_dynamic")])
    kernel = device.create_compute_kernel(program=program)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=4)
    kernel.dispatch((1, 1, 1), records=record, result=result)
    assert result.to_numpy().view(np.uint32)[0] == 123 + 40 + 456


def test_recursive_marshalling_and_dispatch(device: spy.Device) -> None:
    module = device.load_module(SHADER)
    a = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([99, 3, 7, 99], dtype=np.uint32)
    )
    b = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=np.array([99, 99, 11, 13, 99], dtype=np.uint32)
    )
    storage = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.full(128, CANARY, dtype=np.uint32),
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=16)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(a, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(b, spy.ResourceState.shader_resource)
    encoder.set_buffer_state(storage, spy.ResourceState.unordered_access)
    _kernel(device, "nested_marshalling").dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        a=spy.to_buffer_ptr(a, 4),
        b=spy.to_buffer_ptr(b, 8),
        storage=spy.to_rw_buffer_ptr(storage, 128),
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    np.testing.assert_array_equal(result.to_numpy().view(np.uint32), [14, 51, 58, 58])
    composed = device.slang_session.compose_modules(
        "buffer_ptr_nested_dynamic",
        [module],
        [spy.TypeConformance("INestedPointers", "NestedPointers", 0)],
    )
    kernel = device.create_compute_kernel(
        program=device.link_program([composed], [composed.entry_point("nested_dynamic")])
    )
    # DynamicStorage is 128 bytes. Read the second slot directly after a GPU state transition.
    raw = storage.to_numpy().view(np.uint32)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(storage, spy.ResourceState.shader_resource)
    kernel.dispatch(
        (1, 1, 1), command_encoder=encoder, storage=storage, record_index=1, result=result
    )
    device.submit_command_buffer(encoder.finish())
    assert result.to_numpy().view(np.uint32)[0] == 58
    np.testing.assert_array_equal(raw[:32], np.full(32, CANARY, dtype=np.uint32))
    np.testing.assert_array_equal(raw[64:], np.full(64, CANARY, dtype=np.uint32))


# Upstream compiler regressions with strict expected failures


@pytest.mark.parametrize("value", ["double(42.0)", "42.0L"])
def test_constant_double_store(
    device: spy.Device, request: pytest.FixtureRequest, value: str
) -> None:
    if not device.has_feature(spy.Feature.double):
        pytest.skip("Requires float64 shader support")
    source = f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void constant_double_store(RWByteAddressBuffer raw, uniform RWBufferPtr<> dst)
{{
    raw.Store<double>(16, {value});
    dst.as<double>()[0] = {value};
}}
"""
    module = device.load_module_from_source(
        "constant_double_store_" + hashlib.sha256(source.encode()).hexdigest(), source
    )
    kernel = device.create_compute_kernel(
        program=device.link_program([module], [module.entry_point("constant_double_store")])
    )
    expected = np.full(12, 0xDEADBEEF, dtype=np.uint32)
    raw = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    dst = device.create_buffer(usage=spy.BufferUsage.unordered_access, data=expected)
    encoder = device.create_command_encoder()
    encoder.set_buffer_state(raw, spy.ResourceState.unordered_access)
    encoder.set_buffer_state(dst, spy.ResourceState.unordered_access)
    kernel.dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        raw=raw,
        dst=spy.to_rw_buffer_ptr(dst, byte_offset=16, alignment=16),
    )
    device.submit_command_buffer(encoder.finish())
    observed = [raw.to_numpy().view(np.uint32), dst.to_numpy().view(np.uint32)]
    for actual in observed:
        np.testing.assert_array_equal(actual[:4], expected[:4])
        np.testing.assert_array_equal(actual[6:], expected[6:])
    if device.info.type == spy.DeviceType.d3d12:
        # Only the incorrect value is expected; compile/device failures must still fail.
        # strict=True turns a compiler fix into a failing XPASS so we revisit double support.
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Slang narrows constant double stores: https://github.com/shader-slang/slang/issues/14259",
                strict=True,
                raises=AssertionError,
            )
        )
    expected[4:6] = np.array([42.0], dtype=np.float64).view(np.uint32)
    for actual in observed:
        np.testing.assert_array_equal(actual, expected)


def test_native_signed_pointer_offsets(device: spy.Device, request: pytest.FixtureRequest) -> None:
    if device.info.type == spy.DeviceType.d3d12:
        pytest.skip("D3D12 has no native Ptr baseline")
    source = """
[shader("compute")][numthreads(1,1,1)]
void boundary(uniform uint64_t bits, RWStructuredBuffer<uint64_t> result) {
    let p = Ptr<uint>(bits);
    result[0] = uint64_t(p + 1);
    result[1] = uint64_t(p + int(-1));
    result[2] = uint64_t(p + int64_t(-2));
}
"""
    kernel = _compile(device, source)
    storage = device.create_buffer(usage=spy.BufferUsage.shader_resource, size=128)
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=24)
    bits = spy.to_buffer_ptr(storage, 32).data
    kernel.dispatch((1, 1, 1), bits=bits, result=result)
    observed = result.to_numpy().view(np.uint64)
    assert observed[0] == bits + 4
    if device.info.type == spy.DeviceType.vulkan:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Native Vulkan Ptr negative offsets: https://github.com/shader-slang/slang/issues/14305",
                strict=True,
                raises=AssertionError,
            )
        )
    # Addresses stay inside the allocation; suspect results are never dereferenced.
    np.testing.assert_array_equal(observed[1:], [bits - 4, bits - 8])


# Keep the direct compiler control independent of BufferPtr and Falcor imports.
_D3D_NESTED_LAYOUT_SOURCE = """
// BPTR-TEST-003: no BufferPtr or Falcor shader imports are involved.
struct Inner
{
    uint64_t key;
    uint value;
}

struct Outer
{
    Inner items[2];
    uint tail;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void setup(RWStructuredBuffer<Outer> records, uniform uint seed)
{
    Outer value = { { { uint64_t(seed), seed + 1 }, { uint64_t(seed + 2), seed + 3 } }, seed + 4 };
    records[0] = value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void probe(StructuredBuffer<Outer> typed, ByteAddressBuffer raw, RWStructuredBuffer<uint> result)
{
    result[0] = typed[0].tail;
    result[1] = raw.Load<Outer>(0).tail;
    result[2] = raw.Load<uint>(32);
    result[3] = sizeof(Outer);
    result[4] = sizeof(Outer, CDataLayout);
}
"""


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 not selected"
)
def test_d3d_nested_layout_consistency(request: pytest.FixtureRequest) -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    source = _D3D_NESTED_LAYOUT_SOURCE
    storage = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.full(30, CANARY, dtype=np.uint32),
        struct_size=40,
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=20)
    encoder = device.create_command_encoder()
    setup = _compile(device, source, "setup")
    setup.dispatch((1, 1, 1), command_encoder=encoder, records=storage, seed=37)
    encoder.set_buffer_state(storage, spy.ResourceState.shader_resource)
    probe = _compile(device, source, "probe")
    probe.dispatch((1, 1, 1), command_encoder=encoder, typed=storage, raw=storage, result=result)
    device.submit_command_buffer(encoder.finish())
    observed = result.to_numpy().view(np.uint32)
    assert observed[0] == 41
    assert observed[2] == 41
    request.node.add_marker(
        pytest.mark.xfail(
            reason="D3D raw/structured layout mismatch: https://github.com/shader-slang/slang/issues/14311",
            strict=True,
            raises=AssertionError,
        )
    )
    assert observed[1] == observed[0]


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 not selected"
)
@pytest.mark.parametrize("load_alignment", [0, 8])
def test_d3d_wrapper_nested_layout(request: pytest.FixtureRequest, load_alignment: int) -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    source = (
        _D3D_NESTED_LAYOUT_SOURCE
        + f"""
import bufferptr;
[shader("compute")][numthreads(1,1,1)]
void boundary(StructuredBuffer<Outer> typed, uniform BufferPtr<> ptr,
              RWStructuredBuffer<uint> result) {{
    let value = ptr.as<Outer>().with_load_alignment<{load_alignment}>()[0];
    result[0] = typed[0].tail;
    result[1] = value.tail;
    result[2] = uint(value.items[0].key);
    result[3] = value.items[0].value;
    result[4] = uint(value.items[1].key);
    result[5] = value.items[1].value;
}}
"""
    )
    storage = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.full(30, CANARY, dtype=np.uint32),
        struct_size=40,
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=24)
    encoder = device.create_command_encoder()
    setup = _compile(device, _D3D_NESTED_LAYOUT_SOURCE, "setup")
    setup.dispatch((1, 1, 1), command_encoder=encoder, records=storage, seed=37)
    encoder.set_buffer_state(storage, spy.ResourceState.shader_resource)
    _compile(device, source).dispatch(
        (1, 1, 1),
        command_encoder=encoder,
        typed=storage,
        ptr=spy.to_buffer_ptr(storage, alignment=8),
        result=result,
    )
    device.submit_command_buffer(encoder.finish())
    observed = result.to_numpy().view(np.uint32)
    assert observed[0] == 41
    np.testing.assert_array_equal(observed[2:], [37, 38, 39, 40])
    request.node.add_marker(
        pytest.mark.xfail(
            reason="D3D BufferPtr/structured layout mismatch: https://github.com/shader-slang/slang/issues/14311",
            strict=True,
            raises=AssertionError,
        )
    )
    assert observed[1] == observed[0]


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 not selected"
)
@pytest.mark.parametrize("equal_stride", [False, True], ids=["single", "equal_stride"])
def test_d3d_nested_parameter_reflection(
    request: pytest.FixtureRequest, equal_stride: bool
) -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    members = "Inner items[2]; uint tail; uint other;" if equal_stride else "Inner item; uint tail;"
    first = "items[0]" if equal_stride else "item"
    # These are DXC structured-buffer strides, independent of the suspect reflection.
    stride = 40 if equal_stride else 24
    source = f"""
struct Inner {{ uint64_t key; uint value; }}
struct Outer {{ {members} }}
[shader("compute")][numthreads(1,1,1)]
void boundary(StructuredBuffer<Outer> typed, RWStructuredBuffer<uint> result) {{
    result[0] = typed[0].tail;
    result[1] = uint(typed[0].{first}.key);
}}
"""
    kernel = _compile(device, source)
    # Query the actual linked parameter, not sizeof or a standalone natural type layout.
    parameter = next(
        p for p in kernel.program.layout.entry_points[0].parameters if p.name == "typed"
    )
    layout = parameter.type_layout.element_type_layout
    reflected_stride = layout.stride
    if equal_stride:
        assert reflected_stride == stride
    tail_byte_offset = next(f.offset for f in layout.fields if f.name == "tail")
    assert tail_byte_offset % 4 == 0 and 8 <= tail_byte_offset < stride
    data = np.full(stride, CANARY, dtype=np.uint32)
    data[0:2] = [7, 0]
    data[tail_byte_offset // 4] = 41
    storage = device.create_buffer(
        usage=spy.BufferUsage.shader_resource, data=data, struct_size=stride
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=8)
    kernel.dispatch((1, 1, 1), typed=storage, result=result)
    observed = result.to_numpy().view(np.uint32)
    assert observed[1] == 7
    request.node.add_marker(
        pytest.mark.xfail(
            reason="D3D parameter reflection mismatch: https://github.com/shader-slang/slang/issues/14311",
            strict=True,
            raises=AssertionError,
        )
    )
    assert observed[0] == 41, f"Reflected tail byte_offset={tail_byte_offset}, read={observed[0]}"
    assert reflected_stride == stride


@pytest.mark.skipif(
    spy.DeviceType.d3d12 not in helpers.DEFAULT_DEVICE_TYPES, reason="D3D12 not selected"
)
@pytest.mark.parametrize("equal_stride", [False, True], ids=["single", "equal_stride"])
def test_d3d_nested_raw_store(request: pytest.FixtureRequest, equal_stride: bool) -> None:
    device = helpers.get_device(spy.DeviceType.d3d12)
    members = "Inner items[2]; uint tail; uint other;" if equal_stride else "Inner item; uint tail;"
    first = "items[0]" if equal_stride else "item"
    stride = 40 if equal_stride else 24
    source = f"""
struct Inner {{ uint64_t key; uint value; }}
struct Outer {{ {members} }}
[shader("compute")][numthreads(1,1,1)]
void write_raw(RWByteAddressBuffer raw, uniform uint seed) {{
    Outer value = {{}};
    value.{first}.key = uint64_t(seed);
    value.tail = seed + 4;
    raw.Store<Outer>(0, value);
}}
[shader("compute")][numthreads(1,1,1)]
void read_typed(StructuredBuffer<Outer> typed, RWStructuredBuffer<uint> result) {{
    result[0] = typed[0].tail;
    result[1] = uint(typed[0].{first}.key);
}}
"""
    storage = device.create_buffer(
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        data=np.full(stride, CANARY, dtype=np.uint32),
        struct_size=stride,
    )
    result = device.create_buffer(usage=spy.BufferUsage.unordered_access, size=8)
    write = _compile(device, source, "write_raw")
    read = _compile(device, source, "read_typed")
    encoder = device.create_command_encoder()
    write.dispatch((1, 1, 1), command_encoder=encoder, raw=storage, seed=37)
    encoder.set_buffer_state(storage, spy.ResourceState.shader_resource)
    read.dispatch((1, 1, 1), command_encoder=encoder, typed=storage, result=result)
    device.submit_command_buffer(encoder.finish())
    observed = result.to_numpy().view(np.uint32)
    assert observed[1] == 37
    # An incorrect aggregate store must not escape the first record.
    np.testing.assert_array_equal(storage.to_numpy().view(np.uint32)[stride // 4 :], CANARY)
    request.node.add_marker(
        pytest.mark.xfail(
            reason="D3D raw-store/structured-read mismatch: https://github.com/shader-slang/slang/issues/14311",
            strict=True,
            raises=AssertionError,
        )
    )
    assert observed[0] == 41


# Emitted code: SPIR-V alignment and Metal feasibility


@pytest.mark.skipif(
    spy.DeviceType.vulkan not in helpers.DEFAULT_DEVICE_TYPES, reason="Vulkan not selected"
)
@pytest.mark.parametrize("alignment", [4, 8, 16])
@pytest.mark.parametrize("family", ["native", "member", "free", "typed"])
def test_spirv_alignment_operands(
    slangc: Path, tmp_path: Path, alignment: int, family: str
) -> None:
    expressions = {
        "native": (
            "loadAligned<A>(Ptr<uint4>(src) + 1)",
            "storeAligned<A>(Ptr<uint4>(dst) + 2, value)",
        ),
        "member": (
            "BufferPtr<>(src).load_aligned<uint4,A>(16)",
            "RWBufferPtr<>(dst).store_aligned<uint4,A>(32, value)",
        ),
        "free": (
            "loadAligned<A>(BufferPtr<uint4>(src) + 1)",
            "storeAligned<A>(RWBufferPtr<uint4>(dst) + 2, value)",
        ),
        "typed": ("BufferPtr<uint4,Access.Read,A>(src)[1]", "RWBufferPtr<uint4,A>(dst)[2] = value"),
    }
    load, store = expressions[family]
    source = tmp_path / "aligned.slang"
    source.write_text(
        f"""
import bufferptr;
static const uint A = {alignment};
[shader("compute")][numthreads(1,1,1)]
void aligned(uniform uint64_t src, uniform uint64_t dst) {{
    let value = {load};
    {store};
}}
"""
    )
    assembly = _emit(
        slangc,
        source,
        tmp_path / "aligned.spvasm",
        "aligned",
        "spirv-asm",
        [Path(spy.__file__).parent / "slang"],
        ["__TARGET_VULKAN__=1"],
    )
    loads = re.findall(r"OpLoad\s+%v4uint\b[^\n]*\bAligned\s+(\d+)", assembly)
    stores = re.findall(r"OpStore\s+[^\n]*\bAligned\s+(\d+)", assembly)
    assert loads == [str(alignment)], assembly
    assert stores == [str(alignment)], assembly


@pytest.mark.parametrize(
    "entry",
    [
        "nested_marshalling",
        "build_chain",
        "follow_chain",
        "typed_access",
        "aligned_access",
        "vector_matrix_access",
        "pointer_tables",
        "casts_and_null",
    ],
)
def test_metal_emission(slangc: Path, tmp_path: Path, entry: str) -> None:
    emitted = _emit(
        slangc,
        Path(__file__).with_suffix(".slang"),
        tmp_path / "probe.metal",
        entry,
        "metal",
        [Path(spy.__file__).parent / "slang"],
        ["__TARGET_METAL__=1"],
    )
    assert "kernel" in emitted and entry in emitted
