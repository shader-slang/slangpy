// SPDX-License-Identifier: Apache-2.0


float add(float a, float b)
{
    return a + b;
}

float multiply(float a, float b)
{
    return a * b;
}

float copy(float val)
{
    return val;
}

void copy_generic<T>(T src, out T dest)
{
    dest = src;
}
