// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

public void copy_values(float input, out float output) {
    output = input;
}

public void copy_vectors(float3 input, out float3 output) {
    output = input;
}

public float3 add_vectors(float3 a, float3 b) {
    return a + b;
}

public void copy_matrix(float2x3 input, out float2x3 output) {
    output = input;
}

public float2x3 add_matrix(float2x3 a, float2x3 b) {
    return a + b;
}
