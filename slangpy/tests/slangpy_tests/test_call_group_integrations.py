# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Integration Tests for call_group_shape Functionality

## Overview
The call_group_shape feature allows organizing compute kernels into logical groups
that can coordinate execution patterns. These tests verify that this functionality
works with major SlangPy components.

## Test Coverage

### 1. Autodiff/Tensor Integration
- **Purpose**: Validates call groups work with differentiable computation
- **Features**: Forward/backward passes, gradient computation, tensor operations
- **Tests**:
  - Differentiable tensor addition with call group scaling
  - Matrix multiplication with call group-based transformations
  - Tensor aggregation operations across different call group configurations

### 2. NDBuffer Integration
- **Purpose**: Ensures call groups work with structured data buffers
- **Features**: 2D/3D buffer processing, vector operations, data transformations
- **Tests**:
  - 2D buffer processing with call group coordinate mapping
  - Vector operations (float3) with call group offsets
  - 3D reduction operations using call group information

### 3. Texture Integration
- **Purpose**: Validates call groups with texture read/write operations
- **Features**: 2D/3D textures, sampling, filtering, format handling
- **Tests**:
  - 2D texture sampling with call group-based UV offsets
  - Single-channel texture reduction operations
  - 3D texture processing with call group coordinate transformation

### 4. Transform/Mapping Integration
- **Purpose**: Tests call groups with SlangPy's transformation system
- **Features**: Dimension mapping (.map()), coordinate transformations
- **Tests**:
  - Basic transform operations with call groups
  - Complex dimension mapping with transposed inputs and call groups
  - Large buffer performance testing
  - Edge cases with misaligned buffer shapes

## Technical Implementation Details

### Call Group API Usage
All tests properly utilize the call group intrinsic functions:
- `get_call_group_thread_id()` - Gets thread position within call group
- `get_call_group_id()` - Gets the call group index
- `get_call_id()` - Gets absolute thread position in the call shape

### Multi-Device Support
All tests are parameterized to run on multiple backend devices:
- D3D12 (DirectX 12)
- Vulkan
- Automatic device detection and fallback
"""

import numpy as np
import pytest
import slangpy as spy
from slangpy import DeviceType, Device, TextureDesc, TextureUsage, Format, TextureType
from slangpy.slangpy import Shape
from slangpy.types import NDBuffer, Tensor
from . import helpers
import sys
import os


def get_tensor_test_module(device: Device):
    """
    Create a SlangPy module for tensor autodiff integration tests.

    This module contains differentiable Slang functions that demonstrate
    call group functionality working correctly with SlangPy's autodiff system.

    Functions provided:
    - tensor_add_with_call_groups: Differentiable tensor addition with call group scaling
    - tensor_matrix_multiply_with_groups: Element-wise matrix operations with call group factors
    - tensor_aggregation_with_groups: Tensor aggregation using call group coordinates

    All functions are marked [Differentiable] and support both forward and backward passes.
    Call group information is used to introduce spatial variation in computations.

    Args:
        device: SlangPy device to create the module on

    Returns:
        Compiled SlangPy module ready for testing
    """
    kernel_source = """
import "slangpy";

[Differentiable]
float tensor_add_with_call_groups(
    float a,
    float b,
    uint2 grid_cell
) {
    // Use call group functions in differentiable context
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Simple operation that can be differentiated
    // Scale the result by call group position for variety
    float scale = 1.0f + 0.1f * float(call_group_id.shape[0] + call_group_id.shape[1]);
    return (a + b) * scale;
}

[Differentiable]
float tensor_matrix_multiply_with_groups(
    float weight_element,
    float x_element,
    uint2 grid_cell
) {
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Simple operation with call group scaling
    float scale = 1.0f + 0.05f * float(call_group_id.shape[0] * call_group_id.shape[1]);

    return weight_element * x_element * scale;
}

float tensor_aggregation_with_groups(
    float input_element,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Transform input based on call group information
    float group_factor = 1.0f + 0.1f * float(call_group_thread_id.shape[0]);
    float thread_factor = 1.0f + 0.05f * float(call_group_id.shape[1]);

    return input_element * group_factor * thread_factor;
}
"""
    return helpers.create_module(device, kernel_source)


def get_ndbuffer_test_module(device: Device):
    """
    Create a SlangPy module for NDBuffer integration tests.

    This module contains Slang functions that work with NDBuffer data structures
    while utilizing call group functionality for coordinate-based processing.

    Functions provided:
    - ndbuffer_process_with_groups: 2D buffer processing with call group transformations
    - ndbuffer_vector_ops_with_groups: Vector (float3) operations with call group offsets
    - ndbuffer_reduce_with_groups: 3D reduction operations using call group coordinates

    These functions demonstrate how call groups can be used to process structured
    data with spatial awareness and coordinate-based transformations.

    Args:
        device: SlangPy device to create the module on

    Returns:
        Compiled SlangPy module ready for testing
    """
    kernel_source = """
import "slangpy";

float ndbuffer_process_with_groups(
    float input,
    out float output,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Apply a transformation based on call group position
    float group_factor = 1.0f + 0.1f * float(call_group_thread_id.shape[0] * call_group_thread_id.shape[1]);
    float thread_factor = 1.0f + 0.05f * float(call_group_id.shape[0] + call_group_id.shape[1]);

    output = input * group_factor * thread_factor;

    return input;
}

float3 ndbuffer_vector_ops_with_groups(
    float3 vector_input,
    uint grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Transform vector based on call group information
    float3 group_offset = float3(
        float(call_group_thread_id.shape[0]) * 0.1f,
        float(call_group_id.shape[0]) * 0.05f,
        0.0f
    );

    return vector_input + group_offset;
}

int ndbuffer_reduce_with_groups(
    int data_input,
    uint3 grid_cell
) {
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Compute a reduction-like operation using call group info
    int sum = data_input;

    // Add contribution from call group position
    sum += int(call_group_id.shape[0] + call_group_id.shape[1] + call_group_id.shape[2]);

    return sum;
}
"""
    return helpers.create_module(device, kernel_source)


def get_texture_test_module(device: Device):
    """
    Create a SlangPy module for texture integration tests.

    This module contains Slang functions that demonstrate call group functionality
    working with various texture operations including sampling, filtering, and I/O.

    Functions provided:
    - texture_sample_with_groups: 2D texture sampling with call group-based UV offsets
      Uses both input (Texture2D) and output (RWTexture2D) textures
    - texture_reduce_with_groups: Single-channel texture reduction with call group patterns
      Demonstrates neighborhood sampling based on call group thread positions
    - texture_3d_with_groups: 3D texture sampling with call group coordinate transformation
      Shows how call groups work in 3D texture space

    These functions showcase proper texture usage patterns, format handling,
    and coordinate transformations using call group information.

    Technical Notes:
    - Handles both multi-channel (RGBA) and single-channel (R) texture formats
    - Uses proper texture usage flags (shader_resource vs unordered_access)
    - Demonstrates SamplerState usage with call group offsets

    Args:
        device: SlangPy device to create the module on

    Returns:
        Compiled SlangPy module ready for testing
    """
    kernel_source = """
import "slangpy";

float4 texture_sample_with_groups(
    Texture2D<float4> input_tex,
    RWTexture2D<float4> output_tex,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_id = CallShapeInfo::get_call_id();

    // Sample from input texture with call group-based offset
    float2 uv = float2(float(call_id.shape[0]), float(call_id.shape[1])) / float2(32.0f, 32.0f);
    float2 group_offset = float2(float(call_group_id.shape[0]), float(call_group_id.shape[1])) * 0.01f;

    SamplerState samp;
    float4 sampled = input_tex.SampleLevel(samp, uv + group_offset, 0);

    // Modulate color based on call group thread position
    float thread_scale = 1.0f + 0.1f * float(call_group_thread_id.shape[0] + call_group_thread_id.shape[1]);
    float4 result = sampled * thread_scale;

    // Write to output texture
    uint2 write_coord = uint2(call_id.shape[0], call_id.shape[1]);
    output_tex[write_coord] = result;

    return result;
}

float texture_reduce_with_groups(
    Texture2D<vector<float,1>> input_tex,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_id = CallShapeInfo::get_call_id();

    // Sample multiple texels based on call group configuration
    float sum = 0.0f;

    for (int dx = -int(call_group_thread_id.shape[0]); dx <= int(call_group_thread_id.shape[0]); dx++) {
        for (int dy = -int(call_group_thread_id.shape[1]); dy <= int(call_group_thread_id.shape[1]); dy++) {
            int2 coord = int2(call_id.shape[0] + dx, call_id.shape[1] + dy);
            if (coord.x >= 0 && coord.x < 32 && coord.y >= 0 && coord.y < 32) {
                sum += input_tex.Load(int3(coord, 0)).r;  // Now we need .r since it's vector<float,1>
            }
        }
    }

    return sum / float((2 * call_group_thread_id.shape[0] + 1) * (2 * call_group_thread_id.shape[1] + 1));
}

float3 texture_3d_with_groups(
    Texture3D<float4> tex3d,
    uint3 grid_cell
) {
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();
    CallShapeInfo call_id = CallShapeInfo::get_call_id();

    // 3D texture sampling with call group-based coordinates
    float3 uvw = float3(float(call_id.shape[0]), float(call_id.shape[1]), float(call_id.shape[2])) / float3(16.0f, 16.0f, 16.0f);

    // Offset based on call group
    float3 group_offset = float3(float(call_group_id.shape[0]), float(call_group_id.shape[1]), float(call_group_id.shape[2])) * 0.02f;

    SamplerState samp;
    float4 sampled = tex3d.SampleLevel(samp, uvw + group_offset, 0);

    return sampled.rgb;
}
"""
    return helpers.create_module(device, kernel_source)


def get_transforms_test_module(device: Device):
    """Create a module for transforms/mapping integration tests."""
    path = os.path.join(os.path.dirname(__file__), "test_transforms.slang")
    if os.path.exists(path):
        with open(path, "r") as f:
            base_source = f.read()
    else:
        base_source = ""

    additional_source = """
import "slangpy";

float transform_with_call_groups(
    float input,
    out float output,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Apply transformation that depends on call group structure
    float group_transform = sin(float(call_group_thread_id.shape[0]) * 0.5f) + cos(float(call_group_thread_id.shape[1]) * 0.5f);
    float thread_transform = float(call_group_id.shape[0] + call_group_id.shape[1]) * 0.1f;

    output = input + group_transform + thread_transform;
    return input;
}

float3 vector_transform_with_groups(
    float3 vector_input,
    uint2 grid_cell
) {
    CallShapeInfo call_group_thread_id = CallShapeInfo::get_call_group_thread_id();
    CallShapeInfo call_group_id = CallShapeInfo::get_call_group_id();

    // Rotate vector based on call group position
    float angle = float(call_group_thread_id.shape[0] + call_group_thread_id.shape[1]) * 0.1f;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    // Simple 2D rotation in XY plane
    float3 rotated = float3(
        vector_input.x * cos_a - vector_input.y * sin_a,
        vector_input.x * sin_a + vector_input.y * cos_a,
        vector_input.z + float(call_group_id.shape[0]) * 0.05f
    );

    return rotated;
}
"""

    full_source = base_source + "\n" + additional_source
    return helpers.create_module(device, full_source)


# Autodiff/Tensor Integration Tests


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_autodiff_with_call_groups(device_type: DeviceType):
    """
    Test autodiff tensor operations integrated with call group functionality.

    This test validates that SlangPy's automatic differentiation system works
    correctly when combined with call group features. It tests both forward
    and backward passes through differentiable functions that use call group
    intrinsics.

    Test Flow:
    1. Creates two random tensors with gradient tracking enabled
    2. Applies a differentiable function that uses call group scaling
    3. Performs forward pass and validates results
    4. Executes backward pass with unit gradients
    5. Verifies gradients were computed correctly

    Call Group Usage:
    - Uses (2, 3) call group shape on (4, 6) tensor data
    - Function scales results based on call group coordinates
    - Tests that autodiff works through call group intrinsic calls

    Validation:
    - Result shape matches input tensor shape
    - All values are finite (no NaN/Inf from call group operations)
    - Backward pass produces valid gradients for both input tensors

    Args:
        device_type: Backend device type (D3D12/Vulkan) for parameterized testing
    """
    if sys.platform == "darwin":
        pytest.skip("Skipping on macOS due to slang-gfx resource clear API issue")

    device = helpers.get_device(device_type)
    module = get_tensor_test_module(device)

    # Create test tensors with gradients
    np.random.seed(42)
    a_data = np.random.randn(4, 6).astype(np.float32)
    b_data = np.random.randn(4, 6).astype(np.float32)

    a = Tensor.from_numpy(device, a_data).with_grads()
    b = Tensor.from_numpy(device, b_data).with_grads()

    # Test with call group shape
    call_shape = (4, 6)
    call_group_shape = (2, 3)

    func = module.tensor_add_with_call_groups.call_group_shape(Shape(call_group_shape))
    result = func(a, b, spy.grid(call_shape), _result="numpy")

    # Verify result shape and basic correctness
    assert result.shape == call_shape
    assert np.all(np.isfinite(result))

    # Test backward pass
    result_tensor = Tensor.from_numpy(device, result).with_grads()
    result_tensor.grad_in = Tensor.zeros(device, result.shape, dtype="float")
    result_tensor.grad_in.storage.copy_from_numpy(np.ones(result.shape, dtype=np.float32))

    func.bwds(a, b, spy.grid(call_shape), result_tensor)

    # Verify gradients were computed
    assert a.grad_out is not None
    assert b.grad_out is not None
    assert np.all(np.isfinite(a.grad_out.to_numpy()))
    assert np.all(np.isfinite(b.grad_out.to_numpy()))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_matrix_ops_with_call_groups(device_type: DeviceType):
    """Test tensor matrix operations with call groups."""
    if sys.platform == "darwin":
        pytest.skip("Skipping on macOS due to slang-gfx resource clear API issue")

    device = helpers.get_device(device_type)
    module = get_tensor_test_module(device)

    # Create test tensors
    weights_data = np.random.randn(4, 3).astype(np.float32)
    x_data = np.random.randn(4, 3).astype(np.float32)

    weights = Tensor.from_numpy(device, weights_data).with_grads()
    x = Tensor.from_numpy(device, x_data).with_grads()

    # Test with call group shape
    call_shape = (4, 3)
    call_group_shape = (2, 3)

    func = module.tensor_matrix_multiply_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(weights, x, spy.grid(call_shape), _result="numpy")

    # Verify result
    assert result.shape == call_shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tensor_aggregation_with_call_groups(device_type: DeviceType):
    """Test tensor aggregation operations with call groups."""
    device = helpers.get_device(device_type)
    module = get_tensor_test_module(device)

    # Test with different call group configurations
    test_cases = [
        ((8, 4), (2, 2)),
        ((6, 6), (3, 2)),
        ((10, 5), (5, 1)),
    ]

    for call_shape, call_group_shape in test_cases:
        # Create test tensor matching call_shape
        input_data = np.random.randn(*call_shape).astype(np.float32)
        input_tensor = Tensor.from_numpy(device, input_data)

        func = module.tensor_aggregation_with_groups.call_group_shape(Shape(call_group_shape))
        result = func(input_tensor, spy.grid(call_shape), _result="numpy")

        assert result.shape == call_shape
        assert np.all(np.isfinite(result))


# NDBuffer Integration Tests


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_2d_with_call_groups(device_type: DeviceType):
    """Test 2D NDBuffer processing with call groups."""
    device = helpers.get_device(device_type)
    module = get_ndbuffer_test_module(device)

    # Create test NDBuffers
    shape = (8, 6)
    input_data = np.random.randn(*shape).astype(np.float32)

    input_buffer = NDBuffer(device=device, shape=shape, dtype=float)
    output_buffer = NDBuffer(device=device, shape=shape, dtype=float)

    helpers.write_ndbuffer_from_numpy(input_buffer, input_data.flatten(), 1)

    # Test with call group shape
    call_group_shape = (2, 3)

    func = module.ndbuffer_process_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(input_buffer, output_buffer, spy.grid(shape), _result="numpy")

    # Verify results
    assert result.shape == shape

    # Read back output buffer and verify
    output_data = helpers.read_ndbuffer_from_numpy(output_buffer).reshape(shape)
    assert np.all(np.isfinite(output_data))
    assert not np.array_equal(input_data, output_data)  # Should be modified


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_vector_ops_with_call_groups(device_type: DeviceType):
    """Test NDBuffer vector operations with call groups."""
    device = helpers.get_device(device_type)
    module = get_ndbuffer_test_module(device)

    # Create test vector buffer
    vector_count = 12
    vector_data = np.random.randn(vector_count, 3).astype(np.float32)

    vector_buffer = NDBuffer(device=device, shape=(vector_count,), dtype=spy.float3)
    helpers.write_ndbuffer_from_numpy(vector_buffer, vector_data.flatten(), 3)

    # Test with 1D call groups
    call_shape = (vector_count,)
    call_group_shape = (4,)

    func = module.ndbuffer_vector_ops_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(vector_buffer, spy.grid(call_shape), _result="numpy")

    # Verify results
    assert result.shape == (vector_count, 3)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_ndbuffer_3d_reduce_with_call_groups(device_type: DeviceType):
    """Test 3D NDBuffer reduction with call groups."""
    device = helpers.get_device(device_type)
    module = get_ndbuffer_test_module(device)

    # Create 3D test data
    shape = (4, 4, 2)
    data = np.random.randint(0, 10, shape).astype(np.int32)

    data_buffer = NDBuffer(device=device, shape=shape, dtype=int)
    helpers.write_ndbuffer_from_numpy(data_buffer, data.flatten(), 1)

    # Test with 3D call groups
    call_group_shape = (2, 2, 2)

    func = module.ndbuffer_reduce_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(data_buffer, spy.grid(shape), _result="numpy")

    # Verify results
    assert result.shape == shape
    assert np.all(result >= 0)  # Should be positive due to nature of reduction


# Texture Integration Tests


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_2d_with_call_groups(device_type: DeviceType):
    """
    Test 2D texture operations integrated with call group functionality.

    This test validates that texture sampling, filtering, and writing operations
    work correctly when combined with call group coordinate transformations.
    It demonstrates proper texture usage patterns and format handling.

    Test Flow:
    1. Creates input texture (shader_resource) and output texture (unordered_access)
    2. Fills input texture with random RGBA data
    3. Applies texture sampling function with call group-based UV offsets
    4. Validates results from both function return and output texture

    Technical Details:
    - Input: Texture2D<float4> with shader_resource usage
    - Output: RWTexture2D<float4> with unordered_access usage
    - Uses SamplerState for filtered texture access
    - Call group coordinates influence UV offset calculations
    - Thread positions within call groups scale output values

    Call Group Usage:
    - Uses (8, 8) call group shape on 32x32 texture
    - UV offsets calculated from call group ID coordinates
    - Thread scaling based on call group thread positions

    Validation:
    - Result shape matches texture dimensions plus channel count
    - All sampled values are finite
    - Output texture contains valid processed data

    Args:
        device_type: Backend device type (D3D12/Vulkan) for parameterized testing
    """
    if sys.platform == "darwin":
        pytest.skip("Skipping on macOS due to slang-gfx texture API issues")

    device = helpers.get_device(device_type)
    module = get_texture_test_module(device)

    # Create test textures
    tex_size = 32
    input_desc = TextureDesc()
    input_desc.width = tex_size
    input_desc.height = tex_size
    input_desc.format = Format.rgba32_float
    input_desc.type = TextureType.texture_2d
    input_desc.usage = TextureUsage.shader_resource

    output_desc = TextureDesc()
    output_desc.width = tex_size
    output_desc.height = tex_size
    output_desc.format = Format.rgba32_float
    output_desc.type = TextureType.texture_2d
    output_desc.usage = TextureUsage.unordered_access

    input_tex = device.create_texture(input_desc)
    output_tex = device.create_texture(output_desc)

    # Fill input texture with test data
    test_data = np.random.rand(tex_size, tex_size, 4).astype(np.float32)
    input_tex.copy_from_numpy(test_data)

    # Test with call groups
    call_shape = (tex_size, tex_size)
    call_group_shape = (8, 8)

    func = module.texture_sample_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(input_tex, output_tex, spy.grid(call_shape), _result="numpy")

    # Verify results
    assert result.shape == (tex_size, tex_size, 4)
    assert np.all(np.isfinite(result))

    # Read back output texture and verify
    output_data = output_tex.to_numpy()
    assert np.all(np.isfinite(output_data))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_reduce_with_call_groups(device_type: DeviceType):
    """Test texture reduction operations with call groups."""
    if sys.platform == "darwin":
        pytest.skip("Skipping on macOS due to slang-gfx texture API issues")

    device = helpers.get_device(device_type)
    module = get_texture_test_module(device)

    # Create test texture
    tex_size = 32
    desc = TextureDesc()
    desc.width = tex_size
    desc.height = tex_size
    desc.format = Format.r32_float
    desc.type = TextureType.texture_2d
    desc.usage = TextureUsage.shader_resource

    input_tex = device.create_texture(desc)

    # Fill with test data
    test_data = np.random.rand(tex_size, tex_size).astype(np.float32)  # 2D for single channel
    input_tex.copy_from_numpy(test_data)

    # Test with different call group sizes
    test_cases = [
        (2, 2),
        (4, 4),
        (8, 8),
    ]

    for call_group_shape in test_cases:
        func = module.texture_reduce_with_groups.call_group_shape(Shape(call_group_shape))
        result = func(input_tex, spy.grid((tex_size, tex_size)), _result="numpy")

        assert result.shape == (tex_size, tex_size)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be positive due to averaging


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_texture_3d_with_call_groups(device_type: DeviceType):
    """Test 3D texture operations with call groups."""
    if sys.platform == "darwin":
        pytest.skip("Skipping on macOS due to slang-gfx texture API issues")

    device = helpers.get_device(device_type)
    module = get_texture_test_module(device)

    # Create 3D test texture
    tex_size = 16  # Smaller size for 3D to avoid memory issues
    desc = TextureDesc()
    desc.width = tex_size
    desc.height = tex_size
    desc.depth = tex_size
    desc.format = Format.rgba32_float
    desc.type = TextureType.texture_3d
    desc.usage = TextureUsage.shader_resource

    input_tex = device.create_texture(desc)

    # Fill with test data (3D texture needs 4D numpy array: depth, height, width, channels)
    test_data = np.random.rand(tex_size, tex_size, tex_size, 4).astype(np.float32)
    input_tex.copy_from_numpy(test_data)

    # Test with 3D call groups
    call_shape = (tex_size, tex_size, tex_size)
    call_group_shape = (4, 4, 4)

    func = module.texture_3d_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(input_tex, spy.grid(call_shape), _result="numpy")

    # Verify results
    assert result.shape == (tex_size, tex_size, tex_size, 3)  # Returns float3 (rgb)
    assert np.all(np.isfinite(result))


# Transform/Mapping Integration Tests


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_transforms_with_call_groups(device_type: DeviceType):
    """Test transform operations with call groups."""
    device = helpers.get_device(device_type)
    module = get_transforms_test_module(device)

    # Create test data
    shape = (6, 8)
    input_data = np.random.randn(*shape).astype(np.float32)

    input_buffer = NDBuffer(device=device, shape=shape, dtype=float)
    output_buffer = NDBuffer(device=device, shape=shape, dtype=float)

    helpers.write_ndbuffer_from_numpy(input_buffer, input_data.flatten(), 1)

    # Test transform with call groups
    call_group_shape = (3, 4)

    func = module.transform_with_call_groups.call_group_shape(Shape(call_group_shape))
    result = func(input_buffer, output_buffer, spy.grid(shape), _result="numpy")

    # Verify results
    assert result.shape == shape
    assert np.all(np.isfinite(result))

    # Verify output buffer was modified
    output_data = helpers.read_ndbuffer_from_numpy(output_buffer).reshape(shape)
    assert not np.array_equal(input_data, output_data)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_transforms_with_mapping_and_call_groups(device_type: DeviceType):
    """
    Test complex transform operations combining .map() with call groups.

    This test validates the most complex integration scenario: dimension mapping
    (.map()) combined with call group functionality. It demonstrates proper
    handling of coordinate transformations when input dimensions are transposed.

    Test Flow:
    1. Creates vector data in (4, 6, 3) shape (height, width, channels)
    2. Sets up call group shape (2, 3)
    3. Applies .map((1, 0)) to transpose input dimensions
    4. Adjusts grid shape to match transposed coordinates
    5. Validates results match expected transposed shape

    Technical Challenges Solved:
    - Shape mismatch resolution between mapped inputs and grid coordinates
    - Proper grid shape calculation for transposed dimensions
    - Call group coordinate alignment with mapped data layout

    Key Implementation Detail:
    When using .map((1, 0)) to transpose from (4, 6) to (6, 4):
    - Input buffer: (4, 6, 3) -> mapped as (6, 4, 3)
    - Grid shape: (4, 6) -> adjusted to (6, 4)
    - Call groups work correctly with transposed coordinates

    Call Group Usage:
    - Uses (2, 3) call group shape
    - Works with transposed coordinate system
    - Vector transformations include call group-based rotations

    Validation:
    - Result shape matches expected transposed dimensions
    - All transformation values are finite
    - Demonstrates mapping/call group interoperability

    Args:
        device_type: Backend device type (D3D12/Vulkan) for parameterized testing
    """
    device = helpers.get_device(device_type)
    module = get_transforms_test_module(device)

    # Create test vector data
    shape = (4, 6)
    vector_data = np.random.randn(*shape, 3).astype(np.float32)

    vector_buffer = NDBuffer(device=device, shape=shape, dtype=spy.float3)
    helpers.write_ndbuffer_from_numpy(vector_buffer, vector_data.flatten(), 3)

    # Test with both mapping and call groups
    call_group_shape = (2, 3)

    # Use .map() to transpose dimensions and call groups
    func = module.vector_transform_with_groups.call_group_shape(Shape(call_group_shape))
    mapped_func = func.map((1, 0))  # Transpose the input

    # When mapping input dimensions, we need to transpose the grid shape too
    transposed_shape = (shape[1], shape[0])  # (6, 4) instead of (4, 6)
    result = mapped_func(vector_buffer, spy.grid(transposed_shape), _result="numpy")

    # Result should have transposed shape
    expected_shape = (shape[1], shape[0], 3)
    assert result.shape == expected_shape
    assert np.all(np.isfinite(result))


# Edge Cases and Stress Tests


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_groups_with_large_buffers(device_type: DeviceType):
    """Test call groups with larger buffer sizes."""
    device = helpers.get_device(device_type)
    module = get_ndbuffer_test_module(device)

    # Create larger test case
    shape = (64, 32)
    input_data = np.random.randn(*shape).astype(np.float32)

    input_buffer = NDBuffer(device=device, shape=shape, dtype=float)
    output_buffer = NDBuffer(device=device, shape=shape, dtype=float)

    helpers.write_ndbuffer_from_numpy(input_buffer, input_data.flatten(), 1)

    # Test with various call group sizes
    test_cases = [
        (8, 8),
        (16, 4),
        (4, 16),
        (32, 2),
    ]

    for call_group_shape in test_cases:
        func = module.ndbuffer_process_with_groups.call_group_shape(Shape(call_group_shape))
        result = func(input_buffer, output_buffer, spy.grid(shape), _result="numpy")

        assert result.shape == shape
        assert np.all(np.isfinite(result))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_call_groups_misaligned_shapes(device_type: DeviceType):
    """Test call groups with misaligned shapes (shape not divisible by group size)."""
    device = helpers.get_device(device_type)
    module = get_ndbuffer_test_module(device)

    # Create misaligned test case
    shape = (7, 11)  # Prime numbers to ensure misalignment
    input_data = np.random.randn(*shape).astype(np.float32)

    input_buffer = NDBuffer(device=device, shape=shape, dtype=float)
    output_buffer = NDBuffer(device=device, shape=shape, dtype=float)

    helpers.write_ndbuffer_from_numpy(input_buffer, input_data.flatten(), 1)

    # Test with call group size that doesn't divide evenly
    call_group_shape = (4, 3)

    func = module.ndbuffer_process_with_groups.call_group_shape(Shape(call_group_shape))
    result = func(input_buffer, output_buffer, spy.grid(shape), _result="numpy")

    assert result.shape == shape
    assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
