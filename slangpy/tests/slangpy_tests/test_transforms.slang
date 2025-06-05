// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import "slangpy";

void copy_values(float input, out float output) {
    output = input;
}

void copy_vectors(float3 input, out float3 output) {
    output = input;
}

float3 add_vectors(float3 a, float3 b) {
    return a + b;
}

void copy_matrix(float2x3 input, out float2x3 output) {
    output = input;
}

float2x3 add_matrix(float2x3 a, float2x3 b) {
    return a + b;
}
