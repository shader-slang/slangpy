// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

public interface IBinaryOp : IDifferentiable {
    [Differentiable]
    static float apply(float a, float b);
}

public struct AddOp : IBinaryOp {
    [Differentiable]
    public static float apply(float a, float b) { return a + b; }
}

public struct SubOp : IBinaryOp {
    [Differentiable]
    public static float apply(float a, float b) { return a - b; }
}

public struct MulOp : IBinaryOp {
    [Differentiable]
    public static float apply(float a, float b) { return a * b; }
}
