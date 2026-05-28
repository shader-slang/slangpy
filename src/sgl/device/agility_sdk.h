// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/config.h"
#include <slang-rhi/agility-sdk.h>

// -------------------------------------------------------------------------------------------------
// D3D12 Agility SDK
// -------------------------------------------------------------------------------------------------

#if SGL_HAS_AGILITY_SDK
#define SGL_EXPORT_AGILITY_SDK SLANG_RHI_EXPORT_AGILITY_SDK
#else
#define SGL_EXPORT_AGILITY_SDK
#endif
