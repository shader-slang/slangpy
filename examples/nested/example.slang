// SPDX-License-Identifier: Apache-2.0
import "slangpy";

void copy(float4 src, out float4 dest)
{
    dest = src;
}

void copy_vector<let N : int>(vector<float, N> src, out vector<float, N> dest)
{
    dest = src;
}

void copy_generic<T>(T src, out T dest)
{
    dest = src;
}
