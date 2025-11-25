// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sgl.device.rhi;

void test()
{
    // Simply test that the struct definitions are available.
    { rhi::IndirectDrawArguments tmp;}
    { rhi::IndirectDrawIndexedArguments tmp;}
    { rhi::IndirectDispatchArguments tmp;}
    { rhi::AccelerationStructureInstanceDescD3D12 tmp;}
    { rhi::AccelerationStructureInstanceDescVulkan tmp;}
    { rhi::AccelerationStructureInstanceDescOptix tmp;}
    { rhi::AccelerationStructureInstanceDescMetal tmp;}
}
