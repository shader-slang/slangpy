// SPDX-License-Identifier: Apache-2.0
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
