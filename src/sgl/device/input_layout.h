// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/enum.h"

#include "sgl/device/device_child.h"
#include "sgl/device/formats.h"

#include <slang-rhi.h>

#include <string>

namespace sgl {

enum class InputSlotClass : uint8_t {
    per_vertex = static_cast<uint8_t>(rhi::InputSlotClass::PerVertex),
    per_instance = static_cast<uint8_t>(rhi::InputSlotClass::PerInstance),
};

SGL_ENUM_INFO(
    InputSlotClass,
    {
        {InputSlotClass::per_vertex, "per_vertex"},
        {InputSlotClass::per_instance, "per_instance"},
    }
);
SGL_ENUM_REGISTER(InputSlotClass);

struct InputElementDesc {
    /// The name of the corresponding parameter in shader code.
    std::string semantic_name;
    /// The index of the corresponding parameter in shader code.
    /// Only needed if multiple parameters share a semantic name.
    uint32_t semantic_index{0};
    /// The format of the data being fetched for this element.
    Format format{Format::undefined};
    /// The offset in bytes of this element from the start of the corresponding chunk of vertex stream data.
    size_t offset{0};
    /// The index of the vertex stream to fetch this element's data from.
    uint32_t buffer_slot_index{0};
};

struct VertexStreamDesc {
    /// The stride in bytes for this vertex stream.
    size_t stride{0};
    /// Whether the stream contains per-vertex or per-instance data.
    InputSlotClass slot_class{InputSlotClass::per_vertex};
    /// How many instances to draw per chunk of data.
    uint32_t instance_data_step_rate{0};
};

struct InputLayoutDesc {
    std::vector<InputElementDesc> input_elements;
    std::vector<VertexStreamDesc> vertex_streams;
};

class SGL_API InputLayout : public DeviceChild {
    SGL_OBJECT(InputLayout)
public:
    InputLayout(ref<Device> device, InputLayoutDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_input_layout.setNull(); }

    const InputLayoutDesc& desc() const { return m_desc; }

    rhi::IInputLayout* rhi_input_layout() const { return m_rhi_input_layout.get(); }

    std::string to_string() const override;

private:
    InputLayoutDesc m_desc;
    Slang::ComPtr<rhi::IInputLayout> m_rhi_input_layout;
};

} // namespace sgl
