// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
implementing NeuralNetworks;

// These have to be #includes rather than __include to avoid slang issue #6547.
// This causes an internal compiler error when we spread namespaces across
// multiple files
#include "losses/L1.slang"
#include "losses/L2.slang"
