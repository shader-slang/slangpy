// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}
