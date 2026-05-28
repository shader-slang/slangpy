// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

interface IBinaryOp : IDifferentiable {
    [Differentiable]
    static float apply(float a, float b);
}

struct AddOp : IBinaryOp {
    [Differentiable]
    static float apply(float a, float b) { return a + b; }
}

struct SubOp : IBinaryOp {
    [Differentiable]
    static float apply(float a, float b) { return a - b; }
}

struct MulOp : IBinaryOp {
    [Differentiable]
    static float apply(float a, float b) { return a * b; }
}
