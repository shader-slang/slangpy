// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

// These have to be #includes rather than __include to avoid slang issue #6547.
// This causes an internal compiler error when we spread namespaces across
// multiple files
#include "activations/ELU.slang"
#include "activations/Exp.slang"
#include "activations/Identity.slang"
#include "activations/LeakyReLU.slang"
#include "activations/ReLU.slang"
#include "activations/Sigmoid.slang"
#include "activations/SmeLU.slang"
#include "activations/Swish.slang"
#include "activations/Tanh.slang"
