// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/fwd.h"
#include "sgl/device/shader_offset.h"
#include "sgl/device/reflection.h"
#include "sgl/device/cursor_utils.h"
#include "sgl/device/device.h"

#include "sgl/core/config.h"
#include "sgl/core/macros.h"

#include "sgl/device/cursor_access_wrappers.h"

#include <string_view>

namespace sgl {

/// Represents a single element of a given type in a block of memory, and
/// provides read/write tools to access its members via reflection.
class SGL_API BufferElementCursor : public CursorWriteWrappers<BufferElementCursor, size_t>,
                                    public CursorReadWrappers<BufferElementCursor, size_t> {
public:
    BufferElementCursor() = default;

    /// Create with none-owning view of specific block of memory
    BufferElementCursor(ref<const TypeLayoutReflection> layout, ref<BufferCursor> owner);

    ref<const TypeLayoutReflection> type_layout() const { return m_type_layout; }
    ref<const TypeReflection> type() const { return m_type_layout->type(); }

    slang::TypeLayoutReflection* slang_type_layout() const { return m_type_layout->slang_target(); }

    size_t offset() const { return m_offset; }

    bool is_valid() const { return m_buffer != nullptr; }

    std::string to_string() const;

    BufferElementCursor operator[](std::string_view name) const;
    BufferElementCursor operator[](uint32_t index) const;

    BufferElementCursor find_field(std::string_view name) const;
    BufferElementCursor find_element(uint32_t index) const;

    bool has_field(std::string_view name) const { return find_field(name).is_valid(); }
    bool has_element(uint32_t index) const { return find_element(index).is_valid(); }

    void set_pointer(uint64_t pointer_value);

    void set_data(const void* data, size_t size);

    template<typename T>
    BufferElementCursor& operator=(const T& value)
    {
        set(value);
        return *this;
    }

    template<typename T>
    T as() const
    {
        T value;
        get(value);
        return value;
    }

    template<typename T>
    void get(T& value) const;

    template<typename T>
    void set(const T& value);

    void _set_offset(size_t new_offset) { m_offset = new_offset; }

    /// CursorWriteWrappers, CursorReadWrappers
    void _set_data(size_t offset, const void* data, size_t size) const { write_data(offset, data, size); }
    void _get_data(size_t offset, void* data, size_t size) const { return read_data(offset, data, size); }
    size_t _get_offset() const { return m_offset; }
    static size_t _increment_offset(size_t offset, size_t diff) { return offset + diff; }
    DeviceType _get_device_type() const;

private:
    void write_data(size_t offset, const void* data, size_t size) const;
    void read_data(size_t offset, void* data, size_t size) const;

    ref<const TypeLayoutReflection> m_type_layout;
    ref<BufferCursor> m_buffer;
    size_t m_offset{0};

    friend class BufferCursor;
};

/// Represents a list of elements in a block of memory, and provides
/// simple interface to get a BufferElementCursor for each one. As
/// this can be the owner of its data, it is a ref counted object that
/// elements refer to.
class SGL_API BufferCursor : Object {
public:
    BufferCursor() = default;

    /// Create with none-owning view of specific block of memory. Number of
    /// elements is inferred from the size of the block and the type layout.
    BufferCursor(DeviceType device_type, ref<const TypeLayoutReflection> element_layout, void* data, size_t size);

    /// Create buffer + allocate space internally for a given number of elements.
    BufferCursor(DeviceType device_type, ref<const TypeLayoutReflection> element_layout, size_t element_count);

    /// Create as a view onto a buffer resource. Disable load_before_write to
    /// prevent automatic loading of current buffer state before writing data to it.
    BufferCursor(ref<const TypeLayoutReflection> element_layout, ref<Buffer> resource, bool load_before_write = true);

    /// Create as a view onto a section of a buffer resource. Disable load_before_write to
    /// prevent automatic loading of current buffer state before writing data to it.
    BufferCursor(
        ref<const TypeLayoutReflection> element_layout,
        ref<Buffer> resource,
        size_t size,
        size_t offset,
        bool load_before_write = true
    );

    ~BufferCursor();

    /// Get type layout of an element of the cursor.
    ref<const TypeLayoutReflection> element_type_layout() const { return m_element_type_layout; }

    /// Get type of an element of the cursor.
    ref<const TypeReflection> element_type() const { return m_element_type_layout->type(); }

    /// Get element at a given index.
    BufferElementCursor find_element(uint32_t index);

    /// Index operator to get element at a given index.
    BufferElementCursor operator[](uint32_t index) { return find_element(index); }

    /// Number of elements in the buffer.
    size_t element_count() const { return size() / element_stride(); }

    /// Size of element.
    size_t element_size() const { return m_element_type_layout->size(); }

    /// Stride of elements.
    size_t element_stride() const { return m_element_type_layout->stride(); }

    /// Size of whole buffer.
    size_t size() const { return m_size; }

    /// Check if internal buffer exists.
    bool is_loaded() const { return m_buffer != nullptr; }

    /// Write data to buffer (note: writes only to host memory).
    void write_data(size_t offset, const void* data, size_t size);

    /// Reads data from buffer (note: reads only from host memory).
    void read_data(size_t offset, void* data, size_t size) const;

    /// In case of GPU only buffers, loads all data from GPU.
    void load();

    /// In case of GPU only buffers, pushes all data to the GPU.
    void apply();

    /// Get the resource this cursor represents (if any).
    ref<Buffer> resource() const { return m_resource; }

    /// Get device type that determines the data layout rules.
    DeviceType get_device_type() const { return m_device_type; }

private:
    ref<const TypeLayoutReflection> m_element_type_layout;
    ref<Buffer> m_resource;
    DeviceType m_device_type;
    uint8_t* m_buffer{nullptr};
    size_t m_size{0};
    bool m_owner{false};
    DeviceOffset m_offset{0};
    bool m_load_before_write{true};
};


} // namespace sgl
