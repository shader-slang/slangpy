// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "input_layout.h"

#include "sgl/device/device.h"
#include "sgl/device/helpers.h"

#include "sgl/core/format.h"
#include "sgl/core/type_utils.h"
#include "sgl/core/short_vector.h"
#include "sgl/core/string.h"

namespace sgl {

InputLayout::InputLayout(ref<Device> device, InputLayoutDesc desc)
    : DeviceChild(std::move(device))
    , m_desc(std::move(desc))
{
    short_vector<rhi::InputElementDesc, 16> rhi_input_elements;
    for (const auto& input_element : m_desc.input_elements) {
        rhi_input_elements.push_back({
            .semanticName = input_element.semantic_name.c_str(),
            .semanticIndex = input_element.semantic_index,
            .format = static_cast<rhi::Format>(input_element.format),
            .offset = narrow_cast<uint32_t>(input_element.offset),
            .bufferSlotIndex = input_element.buffer_slot_index,
        });
    }

    short_vector<rhi::VertexStreamDesc, 16> rhi_vertex_streams;
    for (const auto& vertex_stream : m_desc.vertex_streams) {
        rhi_vertex_streams.push_back({
            .stride = narrow_cast<uint32_t>(vertex_stream.stride),
            .slotClass = static_cast<rhi::InputSlotClass>(vertex_stream.slot_class),
            .instanceDataStepRate = narrow_cast<uint32_t>(vertex_stream.instance_data_step_rate),
        });
    }

    rhi::InputLayoutDesc rhi_desc{
        .inputElements = rhi_input_elements.data(),
        .inputElementCount = narrow_cast<uint32_t>(rhi_input_elements.size()),
        .vertexStreams = rhi_vertex_streams.data(),
        .vertexStreamCount = narrow_cast<uint32_t>(rhi_vertex_streams.size()),
    };
    SLANG_RHI_CALL(m_device->rhi_device()->createInputLayout(rhi_desc, m_rhi_input_layout.writeRef()));
}

inline std::string to_string(const InputElementDesc& desc)
{
    return fmt::format(
        "(semantic_name=\"{}\", semantic_index={}, format={}, offset={}, buffer_slot_index={})",
        desc.semantic_name,
        desc.semantic_index,
        desc.format,
        desc.offset,
        desc.buffer_slot_index
    );
}

inline std::string to_string(const VertexStreamDesc& desc)
{
    return fmt::format(
        "(stride={}, slot_class={}, instance_data_step_rate={})",
        desc.stride,
        desc.slot_class,
        desc.instance_data_step_rate
    );
}

std::string InputLayout::to_string() const
{
    return fmt::format(
        "InputLayout(\n"
        "  device = {},\n"
        "  input_elements = {},\n"
        "  vertex_streams = {}\n"
        ")",
        m_device,
        string::indent(string::list_to_string(m_desc.input_elements)),
        string::indent(string::list_to_string(m_desc.vertex_streams))
    );
}

} // namespace sgl
