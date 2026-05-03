// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// ImGui configuration file. See imconfig.h for more details.

namespace sgl {
class Texture;
}
#define ImTextureID ::sgl::Texture*

#include "sgl/math/vector_types.h"

#define IM_VEC2_CLASS_EXTRA                                                                                            \
    constexpr ImVec2(const ::sgl::float2& f)                                                                           \
        : x(f.x)                                                                                                       \
        , y(f.y)                                                                                                       \
    {                                                                                                                  \
    }                                                                                                                  \
    operator ::sgl::float2() const                                                                                     \
    {                                                                                                                  \
        return ::sgl::float2(x, y);                                                                                    \
    }

#define IM_VEC3_CLASS_EXTRA                                                                                            \
    constexpr ImVec3(const ::sgl::float3& f)                                                                           \
        : x(f.x)                                                                                                       \
        , y(f.y)                                                                                                       \
        , z(f.z)                                                                                                       \
    {                                                                                                                  \
    }                                                                                                                  \
    operator ::sgl::float3() const                                                                                     \
    {                                                                                                                  \
        return ::sgl::float3(x, y, z);                                                                                 \
    }

#define IM_VEC4_CLASS_EXTRA                                                                                            \
    constexpr ImVec4(const ::sgl::float4& f)                                                                           \
        : x(f.x)                                                                                                       \
        , y(f.y)                                                                                                       \
        , z(f.z)                                                                                                       \
        , w(f.w)                                                                                                       \
    {                                                                                                                  \
    }                                                                                                                  \
    operator ::sgl::float4() const                                                                                     \
    {                                                                                                                  \
        return ::sgl::float4(x, y, z, w);                                                                              \
    }
