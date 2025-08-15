// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/types.h"
#include "sgl/device/device_child.h"
#include "sgl/device/formats.h"
#include "sgl/device/native_handle.h"

#include "sgl/core/fwd.h"
#include "sgl/core/macros.h"
#include "sgl/core/enum.h"
#include "sgl/core/object.h"

#include <slang-rhi.h>

#include <limits>
#include <map>
#include <set>
#include <span>

namespace sgl {

enum class ResourceState : uint32_t {
    undefined = static_cast<uint32_t>(rhi::ResourceState::Undefined),
    general = static_cast<uint32_t>(rhi::ResourceState::General),
    vertex_buffer = static_cast<uint32_t>(rhi::ResourceState::VertexBuffer),
    index_buffer = static_cast<uint32_t>(rhi::ResourceState::IndexBuffer),
    constant_buffer = static_cast<uint32_t>(rhi::ResourceState::ConstantBuffer),
    stream_output = static_cast<uint32_t>(rhi::ResourceState::StreamOutput),
    shader_resource = static_cast<uint32_t>(rhi::ResourceState::ShaderResource),
    unordered_access = static_cast<uint32_t>(rhi::ResourceState::UnorderedAccess),
    render_target = static_cast<uint32_t>(rhi::ResourceState::RenderTarget),
    depth_read = static_cast<uint32_t>(rhi::ResourceState::DepthRead),
    depth_write = static_cast<uint32_t>(rhi::ResourceState::DepthWrite),
    present = static_cast<uint32_t>(rhi::ResourceState::Present),
    indirect_argument = static_cast<uint32_t>(rhi::ResourceState::IndirectArgument),
    copy_source = static_cast<uint32_t>(rhi::ResourceState::CopySource),
    copy_destination = static_cast<uint32_t>(rhi::ResourceState::CopyDestination),
    resolve_source = static_cast<uint32_t>(rhi::ResourceState::ResolveSource),
    resolve_destination = static_cast<uint32_t>(rhi::ResourceState::ResolveDestination),
    acceleration_structure = static_cast<uint32_t>(rhi::ResourceState::AccelerationStructure),
    acceleration_structure_build_output = static_cast<uint32_t>(rhi::ResourceState::AccelerationStructureBuildInput),
};

SGL_ENUM_INFO(
    ResourceState,
    {
        {ResourceState::undefined, "undefined"},
        {ResourceState::general, "general"},
        {ResourceState::vertex_buffer, "vertex_buffer"},
        {ResourceState::index_buffer, "index_buffer"},
        {ResourceState::constant_buffer, "constant_buffer"},
        {ResourceState::stream_output, "stream_output"},
        {ResourceState::shader_resource, "shader_resource"},
        {ResourceState::unordered_access, "unordered_access"},
        {ResourceState::render_target, "render_target"},
        {ResourceState::depth_read, "depth_read"},
        {ResourceState::depth_write, "depth_write"},
        {ResourceState::present, "present"},
        {ResourceState::indirect_argument, "indirect_argument"},
        {ResourceState::copy_source, "copy_source"},
        {ResourceState::copy_destination, "copy_destination"},
        {ResourceState::resolve_source, "resolve_source"},
        {ResourceState::resolve_destination, "resolve_destination"},
        {ResourceState::acceleration_structure, "acceleration_structure"},
        {ResourceState::acceleration_structure_build_output, "acceleration_structure_build_output"},
    }
);
SGL_ENUM_REGISTER(ResourceState);

enum class BufferUsage : uint32_t {
    none = static_cast<uint32_t>(rhi::BufferUsage::None),
    vertex_buffer = static_cast<uint32_t>(rhi::BufferUsage::VertexBuffer),
    index_buffer = static_cast<uint32_t>(rhi::BufferUsage::IndexBuffer),
    constant_buffer = static_cast<uint32_t>(rhi::BufferUsage::ConstantBuffer),
    shader_resource = static_cast<uint32_t>(rhi::BufferUsage::ShaderResource),
    unordered_access = static_cast<uint32_t>(rhi::BufferUsage::UnorderedAccess),
    indirect_argument = static_cast<uint32_t>(rhi::BufferUsage::IndirectArgument),
    copy_source = static_cast<uint32_t>(rhi::BufferUsage::CopySource),
    copy_destination = static_cast<uint32_t>(rhi::BufferUsage::CopyDestination),
    acceleration_structure = static_cast<uint32_t>(rhi::BufferUsage::AccelerationStructure),
    acceleration_structure_build_input = static_cast<uint32_t>(rhi::BufferUsage::AccelerationStructureBuildInput),
    shader_table = static_cast<uint32_t>(rhi::BufferUsage::ShaderTable),
    shared = static_cast<uint32_t>(rhi::BufferUsage::Shared),
};
SGL_ENUM_CLASS_OPERATORS(BufferUsage);

SGL_ENUM_INFO(
    BufferUsage,
    {
        {BufferUsage::none, "none"},
        {BufferUsage::vertex_buffer, "vertex_buffer"},
        {BufferUsage::index_buffer, "index_buffer"},
        {BufferUsage::constant_buffer, "constant_buffer"},
        {BufferUsage::shader_resource, "shader_resource"},
        {BufferUsage::unordered_access, "unordered_access"},
        {BufferUsage::indirect_argument, "indirect_argument"},
        {BufferUsage::copy_source, "copy_source"},
        {BufferUsage::copy_destination, "copy_destination"},
        {BufferUsage::acceleration_structure, "acceleration_structure"},
        {BufferUsage::acceleration_structure_build_input, "acceleration_structure_build_input"},
        {BufferUsage::shader_table, "shader_table"},
        {BufferUsage::shared, "shared"},
    }
);
SGL_ENUM_REGISTER(BufferUsage);

enum class TextureUsage : uint32_t {
    none = static_cast<uint32_t>(rhi::TextureUsage::None),
    shader_resource = static_cast<uint32_t>(rhi::TextureUsage::ShaderResource),
    unordered_access = static_cast<uint32_t>(rhi::TextureUsage::UnorderedAccess),
    render_target = static_cast<uint32_t>(rhi::TextureUsage::RenderTarget),
    depth_stencil = static_cast<uint32_t>(rhi::TextureUsage::DepthStencil),
    present = static_cast<uint32_t>(rhi::TextureUsage::Present),
    copy_source = static_cast<uint32_t>(rhi::TextureUsage::CopySource),
    copy_destination = static_cast<uint32_t>(rhi::TextureUsage::CopyDestination),
    resolve_source = static_cast<uint32_t>(rhi::TextureUsage::ResolveSource),
    resolve_destination = static_cast<uint32_t>(rhi::TextureUsage::ResolveDestination),
    typeless = static_cast<uint32_t>(rhi::TextureUsage::Typeless),
    shared = static_cast<uint32_t>(rhi::TextureUsage::Shared),
};
SGL_ENUM_CLASS_OPERATORS(TextureUsage);

SGL_ENUM_INFO(
    TextureUsage,
    {
        {TextureUsage::none, "none"},
        {TextureUsage::shader_resource, "shader_resource"},
        {TextureUsage::unordered_access, "unordered_access"},
        {TextureUsage::render_target, "render_target"},
        {TextureUsage::depth_stencil, "depth_stencil"},
        {TextureUsage::present, "present"},
        {TextureUsage::copy_source, "copy_source"},
        {TextureUsage::copy_destination, "copy_destination"},
        {TextureUsage::resolve_source, "resolve_source"},
        {TextureUsage::resolve_destination, "resolve_destination"},
        {TextureUsage::typeless, "typeless"},
        {TextureUsage::shared, "shared"},
    }
);
SGL_ENUM_REGISTER(TextureUsage);

enum class TextureType : uint32_t {
    texture_1d = static_cast<uint32_t>(rhi::TextureType::Texture1D),
    texture_1d_array = static_cast<uint32_t>(rhi::TextureType::Texture1DArray),
    texture_2d = static_cast<uint32_t>(rhi::TextureType::Texture2D),
    texture_2d_array = static_cast<uint32_t>(rhi::TextureType::Texture2DArray),
    texture_2d_ms = static_cast<uint32_t>(rhi::TextureType::Texture2DMS),
    texture_2d_ms_array = static_cast<uint32_t>(rhi::TextureType::Texture2DMSArray),
    texture_3d = static_cast<uint32_t>(rhi::TextureType::Texture3D),
    texture_cube = static_cast<uint32_t>(rhi::TextureType::TextureCube),
    texture_cube_array = static_cast<uint32_t>(rhi::TextureType::TextureCubeArray),
};

SGL_ENUM_INFO(
    TextureType,
    {
        {TextureType::texture_1d, "texture_1d"},
        {TextureType::texture_1d_array, "texture_1d_array"},
        {TextureType::texture_2d, "texture_2d"},
        {TextureType::texture_2d_array, "texture_2d_array"},
        {TextureType::texture_2d_ms, "texture_2d_ms"},
        {TextureType::texture_2d_ms_array, "texture_2d_ms_array"},
        {TextureType::texture_3d, "texture_3d"},
        {TextureType::texture_cube, "texture_cube"},
        {TextureType::texture_cube_array, "texture_cube_array"},
    }
);
SGL_ENUM_REGISTER(TextureType);

enum class MemoryType : uint32_t {
    device_local = static_cast<uint32_t>(rhi::MemoryType::DeviceLocal),
    upload = static_cast<uint32_t>(rhi::MemoryType::Upload),
    read_back = static_cast<uint32_t>(rhi::MemoryType::ReadBack),
};

SGL_ENUM_INFO(
    MemoryType,
    {
        {MemoryType::device_local, "device_local"},
        {MemoryType::upload, "upload"},
        {MemoryType::read_back, "read_back"},
    }
);
SGL_ENUM_REGISTER(MemoryType);

struct BufferRange {
    static constexpr uint64_t ALL = std::numeric_limits<uint64_t>::max();
    uint64_t offset{0};
    uint64_t size{ALL};

    auto operator<=>(const BufferRange&) const = default;

    std::string to_string() const { return fmt::format("BufferRange(offset={}, size={}", offset, size); }
};

enum class TextureAspect : uint32_t {
    all = static_cast<uint32_t>(rhi::TextureAspect::All),
    depth_only = static_cast<uint32_t>(rhi::TextureAspect::DepthOnly),
    stencil_only = static_cast<uint32_t>(rhi::TextureAspect::StencilOnly),
};
SGL_ENUM_INFO(
    TextureAspect,
    {
        {TextureAspect::all, "all"},
        {TextureAspect::depth_only, "depth_only"},
        {TextureAspect::stencil_only, "stencil_only"},
    }
);
SGL_ENUM_REGISTER(TextureAspect);

static constexpr uint32_t ALL_LAYERS = rhi::kAllLayers;
static constexpr uint32_t ALL_MIPS = rhi::kAllMips;

struct SubresourceRange {
    /// First array layer.
    uint32_t layer{0};
    /// Number of array layers.
    uint32_t layer_count{ALL_LAYERS};
    /// Most detailed mip level.
    uint32_t mip{0};
    /// Number of mip levels.
    uint32_t mip_count{ALL_MIPS};

    auto operator<=>(const SubresourceRange&) const = default;

    std::string to_string() const
    {
        return fmt::format(
            "SubresourceRange(layer={}, layer_count={}, mip={}, mip_count={}",
            layer,
            layer_count,
            mip,
            mip_count
        );
    }
};

class SGL_API Resource : public DeviceChild {
    SGL_OBJECT(Resource)
public:
    virtual ~Resource();

    virtual rhi::IResource* rhi_resource() const = 0;

    /// Get the native resource handle.
    NativeHandle native_handle() const;

protected:
    Resource() = delete;
    Resource(ref<Device> device);
};


struct BufferDesc {
    /// Buffer size in bytes.
    size_t size{0};

    /// Buffer size in number of struct elements. Can be used instead of \c size.
    size_t element_count{0};
    /// Struct size in bytes.
    size_t struct_size{0};
    /// Resource type layout of the buffer. Can be used instead of \c struct_size to specify the size of the struct.
    ref<const TypeLayoutReflection> resource_type_layout;

    /// Buffer format. Used when creating typed buffer views.
    Format format{Format::undefined};

    /// Memory type.
    MemoryType memory_type{MemoryType::device_local};

    /// Resource usage flags.
    BufferUsage usage{BufferUsage::none};
    /// Initial resource state.
    ResourceState default_state{ResourceState::undefined};

    /// Debug label.
    std::string label;

    // TODO(slang-rhi) we might want to move this out of the buffer desc
    /// Initial data to upload to the buffer.
    const void* data{nullptr};
    /// Size of the initial data in bytes.
    size_t data_size{0};
};

class SGL_API Buffer : public Resource {
    SGL_OBJECT(Buffer)
public:
    Buffer(ref<Device> device, BufferDesc desc);

    ~Buffer();

    virtual void _release_rhi_resources() override { m_rhi_buffer.setNull(); }

    const BufferDesc& desc() const { return m_desc; }

    size_t size() const { return m_desc.size; }
    size_t struct_size() const { return m_desc.struct_size; }
    Format format() const { return m_desc.format; }
    MemoryType memory_type() const { return m_desc.memory_type; }

    /// Map the whole buffer.
    /// Only available for buffers created with \c MemoryType::upload or \c MemoryType::read_back.
    void* map() const;

    template<typename T>
    T* map() const
    {
        return reinterpret_cast<T*>(map());
    }

    /// Unmap the buffer.
    void unmap() const;

    /// Returns true if buffer is currently mapped.
    bool is_mapped() const { return m_mapped_ptr != nullptr; }

    /// Returns a pointer to the CUDA memory.
    /// This is only supported if the buffer was created with ResourceUsage::shared
    /// and the device has CUDA interop enabled.
    void* cuda_memory() const;

    /**
     * Set buffer data from host memory.
     *
     * \param data Data to write.
     * \param size Size of the data in bytes.
     * \param offset Offset in the buffer to write to.
     */
    void set_data(const void* data, size_t size, DeviceOffset offset = 0);

    template<typename T>
    void set_element(size_t index, const T& value)
    {
        set_data(&value, sizeof(T), index * sizeof(T));
    }

    template<typename T>
    void set_elements(size_t index, std::span<const T> values)
    {
        set_data(values.data(), values.size() * sizeof(T), index * sizeof(T));
    }

    /**
     * Get buffer data to host memory.
     * \note If the buffer is in device local memory, this will wait until the data is copied back to host memory.
     *
     * \param data Data buffer to read to.
     * \param size Size of the data in bytes.
     * \param offset Offset in the buffer to read from.
     */
    void get_data(void* data, size_t size, DeviceOffset offset = 0);

    template<typename T>
    void get_element(size_t index)
    {
        T value;
        get_data(&value, sizeof(T), index * sizeof(T));
        return value;
    }

    template<typename T>
    std::vector<T> get_elements(size_t index = 0, size_t count = 0)
    {
        if (count == 0)
            count = (m_desc.size / sizeof(T)) - index;
        std::vector<T> values(count);
        get_data(values.data(), values.size() * sizeof(T), index * sizeof(T));
        return values;
    }

    DeviceAddress device_address() const { return m_rhi_buffer->getDeviceAddress(); }

    ref<BufferView> create_view(BufferViewDesc desc);

    /// Get the shared resource handle.
    /// Note: Buffer must be created with the \c BufferUsage::shared usage flag.
    NativeHandle shared_handle() const;

    /// Get bindless descriptor handle for read access.
    DescriptorHandle descriptor_handle_ro() const;
    /// Get bindless descriptor handle for read-write access.
    DescriptorHandle descriptor_handle_rw() const;

    virtual rhi::IResource* rhi_resource() const override { return m_rhi_buffer; }
    rhi::IBuffer* rhi_buffer() const { return m_rhi_buffer; }

    MemoryUsage memory_usage() const override;

    std::string to_string() const override;

private:
    BufferDesc m_desc;
    Slang::ComPtr<rhi::IBuffer> m_rhi_buffer;
    mutable ref<cuda::ExternalMemory> m_cuda_memory;
    mutable void* m_mapped_ptr{nullptr};
};

struct BufferViewDesc {
    Format format{Format::undefined};
    BufferRange range;
    std::string label;
};

class SGL_API BufferView : public DeviceChild {
    SGL_OBJECT(BufferView)
public:
    BufferView(ref<Device> device, ref<Buffer> buffer, BufferViewDesc desc);

    virtual void _release_rhi_resources() override { }

    Buffer* buffer() const { return m_buffer; }

    const BufferViewDesc& desc() const { return m_desc; }

    Format format() const { return m_desc.format; }
    const BufferRange& range() const { return m_desc.range; }
    std::string_view label() const { return m_desc.label; }

    /// Get the native buffer view handle.
    NativeHandle native_handle() const;

    std::string to_string() const override;

private:
    ref<Buffer> m_buffer;
    BufferViewDesc m_desc;
    // Slang::ComPtr<rhi::IBufferView> m_rhi_buffer_view;
};

struct SGL_API BufferOffsetPair {
    Buffer* buffer{nullptr};
    DeviceOffset offset{0};

    BufferOffsetPair() = default;
    BufferOffsetPair(Buffer* buffer, DeviceOffset offset = 0)
        : buffer(buffer)
        , offset(offset)
    {
    }
    BufferOffsetPair(const ref<Buffer>& buffer, DeviceOffset offset = 0)
        : buffer(buffer.get())
        , offset(offset)
    {
    }
};

struct SubresourceData {
    const void* data{nullptr};
    size_t size{0};
    size_t row_pitch{0};
    size_t slice_pitch{0};
};

struct OwnedSubresourceData : SubresourceData {
    std::unique_ptr<uint8_t[]> owned_data;
};

static constexpr uint32_t DEFAULT_ALIGNMENT = rhi::kDefaultAlignment;


struct TextureDesc {
    /// Texture type.
    TextureType type{TextureType::texture_2d};
    /// Texture format.
    Format format{Format::undefined};
    /// Width in pixels.
    uint32_t width{1};
    /// Height in pixels.
    uint32_t height{1};
    /// Depth in pixels.
    uint32_t depth{1};
    /// Array length.
    uint32_t array_length{1};
    /// Number of mip levels (ALL_MIPS for all mip levels).
    uint32_t mip_count{1};
    /// Number of samples per pixel.
    uint32_t sample_count{1};
    /// Quality level for multisampled textures.
    uint32_t sample_quality{0};

    // TODO(@skallweit): support clear value

    MemoryType memory_type{MemoryType::device_local};

    TextureUsage usage{TextureUsage::none};
    ResourceState default_state{ResourceState::undefined};

    /// Debug label.
    std::string label;

    // TODO(slang-rhi) we might want to move this out of the texture desc
    std::span<SubresourceData> data;
    // const void* data{nullptr};
    // size_t data_size{0};
};

struct SubresourceLayout {
    /// Dimensions of the subresource (in texels).
    uint3 size;

    /// Stride in bytes between columns (i.e. blocks) of the subresource tensor.
    size_t col_pitch;

    /// Stride in bytes between rows of the subresource tensor.
    size_t row_pitch;

    /// Stride in bytes between slices of the subresource tensor.
    size_t slice_pitch;

    /// Overall size required to fit the subresource data (typically size.z * slice_pitch).
    size_t size_in_bytes;

    /// Block width in texels (1 for uncompressed formats).
    size_t block_width;

    /// Block height in texels (1 for uncompressed formats).
    size_t block_height;

    /// Number of rows.
    /// For uncompressed formats this matches size.y.
    /// For compressed formats this matches align_up(size.y, block_height) / block_height.
    size_t row_count;
};

SubresourceLayout layout_from_rhilayout(const rhi::SubresourceLayout& rhi_layout);

class SGL_API Texture : public Resource {
    SGL_OBJECT(Texture)
public:
    Texture(ref<Device> device, TextureDesc desc);
    Texture(ref<Device> device, TextureDesc desc, rhi::ITexture* resource);

    ~Texture();

    virtual void _release_rhi_resources() override { m_rhi_texture.setNull(); }

    const TextureDesc& desc() const { return m_desc; }

    TextureType type() const { return m_desc.type; }
    Format format() const { return m_desc.format; }
    uint32_t width() const { return m_desc.width; }
    uint32_t height() const { return m_desc.height; }
    uint32_t depth() const { return m_desc.depth; }

    uint32_t array_length() const { return m_desc.array_length; }
    uint32_t mip_count() const { return m_desc.mip_count; }
    uint32_t layer_count() const
    {
        return m_desc.array_length
            * ((m_desc.type == TextureType::texture_cube || m_desc.type == TextureType::texture_cube_array) ? 6 : 1);
    }

    uint32_t subresource_count() const { return layer_count() * mip_count(); }

    uint32_t get_mip_width(uint32_t mip = 0) const { return mip < mip_count() ? std::max(1U, width() >> mip) : 0; }

    uint32_t get_mip_height(uint32_t mip = 0) const { return mip < mip_count() ? std::max(1U, height() >> mip) : 0; }

    uint32_t get_mip_depth(uint32_t mip = 0) const { return mip < mip_count() ? std::max(1U, depth() >> mip) : 0; }

    uint3 get_mip_size(uint32_t mip = 0) const
    {
        return uint3(get_mip_width(mip), get_mip_height(mip), get_mip_depth(mip));
    }

    /// Get layout of a texture subresource. By default, the row alignment used is
    /// that required by the target for direct buffer upload/download. Pass in 1
    /// for a completely packed layout.
    SubresourceLayout get_subresource_layout(uint32_t mip, uint32_t row_alignment = DEFAULT_ALIGNMENT) const;

    /**
     * Set subresource data from host memory.
     *
     * \param layer Layer index.
     * \param mip Mip level.
     * \param subresource_data Subresource data.
     */
    void set_subresource_data(uint32_t layer, uint32_t mip, SubresourceData subresource_data);

    /**
     * Get subresource data to host memory.
     * \note This will wait until the data is copied back to host memory.
     *
     * \param layer Layer index.
     * \param mip Mip level.
     * \return Subresource data.
     */
    OwnedSubresourceData get_subresource_data(uint32_t layer, uint32_t mip) const;

    ref<TextureView> create_view(TextureViewDesc desc);

    /// Get the shared resource handle.
    /// Note: Texture must be created with the \c TextureUsage::shared usage flag.
    NativeHandle shared_handle() const;

    virtual rhi::IResource* rhi_resource() const override { return m_rhi_texture; }
    rhi::ITexture* rhi_texture() const { return m_rhi_texture; }

    MemoryUsage memory_usage() const override;

    ref<Bitmap> to_bitmap(uint32_t layer = 0, uint32_t mip = 0) const;

    std::string to_string() const override;

private:
    TextureDesc m_desc;
    Slang::ComPtr<rhi::ITexture> m_rhi_texture;
};

struct TextureViewDesc {
    Format format{Format::undefined};
    TextureAspect aspect{TextureAspect::all};
    SubresourceRange subresource_range;
    std::string label;
};

class SGL_API TextureView : public DeviceChild {
    SGL_OBJECT(TextureView)
public:
    TextureView(ref<Device> device, ref<Texture> texture, TextureViewDesc desc);

    virtual void _release_rhi_resources() override { m_rhi_texture_view.setNull(); }

    Texture* texture() const { return m_texture.get(); }

    const TextureViewDesc& desc() const { return m_desc; }

    Format format() const { return m_desc.format; }
    TextureAspect aspect() const { return m_desc.aspect; }
    const SubresourceRange& subresource_range() const { return m_desc.subresource_range; }
    std::string_view label() const { return m_desc.label; }

    DescriptorHandle descriptor_handle_ro() const;
    DescriptorHandle descriptor_handle_rw() const;

    /// Get the native texture view handle.
    NativeHandle native_handle() const;

    rhi::ITextureView* rhi_texture_view() const { return m_rhi_texture_view.get(); }

    std::string to_string() const override;

private:
    ref<Texture> m_texture;
    TextureViewDesc m_desc;
    Slang::ComPtr<rhi::ITextureView> m_rhi_texture_view;
};

} // namespace sgl
