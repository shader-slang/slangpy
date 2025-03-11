// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

implementing NeuralNetworks;

// Helper typedef for an interface that is constrained to a differentiable, scalar element type
// Currently implemented by half, float and double, though subject to change
// This is used in many model implementations and slang's internal interface for this is a bit unwieldy,
// so we define a nicer name centrally here. This might be replaced by a custom interface in the future.
public typealias IReal = __BuiltinFloatingPointType;
