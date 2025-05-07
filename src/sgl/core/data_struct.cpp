// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
This is a modified version of the struct.cpp file from Mitsuba 3.0 found at
https://github.com/mitsuba-renderer/mitsuba3/blob/master/src/core/struct.cpp

Original license below:

Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
*/

#include "data_struct.h"

#include "sgl/core/config.h"
#include "sgl/core/format.h"
#include "sgl/core/maths.h"
#include "sgl/core/string.h"
#include "sgl/core/hash.h"

#include "sgl/math/float16.h"
#include "sgl/math/colorspace.h"

#include "sgl/stl/bit.h" // Replace with <bit> when available on all platforms.

#if SGL_HAS_ASMJIT
#include <asmjit/asmjit.h>
#if SGL_ARM64
#include <asmjit/arm.h>
#include <asmjit/arm/a64assembler.h>
#include <asmjit/arm/a64compiler.h>
#endif
#endif

#include <limits>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#define SGL_LOG_JIT_ASSEMBLY 0

namespace sgl {

size_t hash(const DataStruct::Field& field)
{
    size_t value = hash(field.name, field.type, field.flags, field.size, field.offset, field.default_value);
    for (const auto& [weight, name] : field.blend)
        value = hash_combine(value, hash(weight, name));
    return value;
}

std::string DataStruct::Field::to_string() const
{
    std::string result;
    result = fmt::format("Field(name=\"{}\", type={}, flags={}, size={}, offset={}", name, type, flags, size, offset);
    if (is_set(flags, Flags::default_))
        result += fmt::format(", default_value={}", default_value);
    if (!blend.empty()) {
        std::vector<std::string> blend_strings;
        for (const auto& [w, n] : blend)
            blend_strings.push_back(fmt::format("({}, \"{}\")", w, n));
        result += fmt::format(", blend=[{}]", string::join(blend_strings, ", "));
    }
    result += ")";
    return result;
}

DataStruct::DataStruct(bool pack, ByteOrder byte_order)
    : m_pack(pack)
{
    m_byte_order = byte_order == ByteOrder::host ? host_byte_order() : byte_order;
}

DataStruct& DataStruct::append(Field field)
{
    m_fields.push_back(field);
    return *this;
}

DataStruct&
DataStruct::append(std::string_view name, Type type, Flags flags, double default_value, const Field::BlendList& blend)
{
    size_t size = type_size(type);
    size_t offset = 0;
    if (!m_fields.empty()) {
        offset = m_fields.back().offset + m_fields.back().size;
        if (!m_pack)
            offset = align_to(size, offset);
    }
    return append({std::string(name), type, flags, size, offset, default_value, blend});
}

DataStruct::Field& DataStruct::field(std::string_view name)
{
    for (auto& field : m_fields)
        if (field.name == name)
            return field;
    SGL_THROW("Field \"{}\" not found.", name);
}

const DataStruct::Field& DataStruct::field(std::string_view name) const
{
    for (const auto& field : m_fields)
        if (field.name == name)
            return field;
    SGL_THROW("Field \"{}\" not found.", name);
}

bool DataStruct::has_field(std::string_view name) const
{
    for (const auto& field : m_fields)
        if (field.name == name)
            return true;
    return false;
}

size_t DataStruct::size() const
{
    if (m_fields.empty())
        return 0;
    size_t size = 0;
    for (const auto& field : m_fields)
        size = std::max(size, field.offset + field.size);
    if (!m_pack)
        size = align_to(alignment(), size);
    return size;
}

size_t DataStruct::alignment() const
{
    if (m_pack)
        return 1;
    size_t alignment = 1;
    for (const auto& field : m_fields)
        alignment = std::max(alignment, field.size);
    return alignment;
}

size_t hash(const DataStruct& struct_)
{
    size_t hash = 0;
    for (const auto& field : struct_)
        hash = hash_combine(hash, sgl::hash(field));
    return hash;
}

std::string DataStruct::to_string() const
{
    return fmt::format(
        "DataStruct(\n"
        "  pack = {},\n"
        "  byte_order = {},\n"
        "  fields = {},\n"
        "  size = {},\n"
        "  alignment = {}\n"
        ")",
        m_pack,
        m_byte_order,
        string::indent(string::list_to_string(m_fields)),
        size(),
        alignment()
    );
}

DataStruct::ByteOrder DataStruct::host_byte_order()
{
    if constexpr (stdx::endian::native == stdx::endian::little)
        return ByteOrder::little_endian;
    else
        return ByteOrder::big_endian;
}

std::pair<double, double> DataStruct::type_range(Type type)
{
#define RANGE(type)                                                                                                    \
    std::make_pair<double, double>(                                                                                    \
        static_cast<double>(std::numeric_limits<type>::min()),                                                         \
        static_cast<double>(std::numeric_limits<type>::max())                                                          \
    )

    switch (type) {
    case Type::int8:
        return RANGE(int8_t);
    case Type::int16:
        return RANGE(int16_t);
    case Type::int32:
        return RANGE(int32_t);
    case Type::int64: {
        auto range = RANGE(int64_t);
        range.first = std::nextafter(range.first, 0.f);
        range.second = std::nextafter(range.second, 0.f);
        return range;
    }
    case Type::uint8:
        return RANGE(uint8_t);
    case Type::uint16:
        return RANGE(uint16_t);
    case Type::uint32:
        return RANGE(uint32_t);
    case Type::uint64: {
        auto range = RANGE(uint64_t);
        range.second = std::nextafter(range.second, 0.f);
        return range;
    }
    case Type::float16:
        return RANGE(math::float16_t);
    case Type::float32:
        return RANGE(float);
    case Type::float64:
        return RANGE(double);
    }
    SGL_THROW("Invalid type.");
}

/// Op codes for conversion programs.
struct Op {
    enum class Type : uint8_t {
        load_mem,
        load_imm,
        save_mem,
        cast,
        // all following ops are for floating point values only
        linear_to_srgb,
        srgb_to_linear,
        multiply,
        multiply_add,
        round,
        clamp,
    };
    Type type;
    uint8_t reg;
    union {
        struct {
            size_t offset;
            DataStruct::Type type;
            bool swap;
        } load_mem;
        struct {
            double value;
        } load_imm;
        struct {
            size_t offset;
            DataStruct::Type type;
            bool swap;
        } save_mem;
        struct {
            DataStruct::Type from;
            DataStruct::Type to;
        } cast;
        struct {
            double value;
        } multiply;
        struct {
            uint8_t reg;
            double factor;
        } multiply_add;
        struct {
            double min;
            double max;
        } clamp;
    };
};

/// Virtual machine for running conversion programs.
struct VM {
    struct Value {
        DataStruct::Type type;
        union {
            /// Signed integer value (int8, int16, int32, int64).
            int64_t i;
            /// Unsigned integer value (uint8, uint16, uint32, uint64).
            uint64_t u;
            /// Single precision floating point value (float16, float32).
            float s;
            /// Double precision floating point value (float64).
            double d;
        };
    };

    const uint8_t* src;
    uint8_t* dst;
    Value registers[8];

    void run(std::span<const Op> code)
    {
        for (const Op& op : code) {
            Value& value = registers[op.reg];
            switch (op.type) {
            case Op::Type::load_mem:
                value = load(src, op.load_mem.offset, op.load_mem.type, op.load_mem.swap);
                break;
            case Op::Type::load_imm:
                value.type = DataStruct::Type::float64;
                value.d = op.load_imm.value;
                break;
            case Op::Type::save_mem:
                SGL_ASSERT(value.type == op.save_mem.type);
                save(dst, op.save_mem.offset, value, op.save_mem.swap);
                break;
            case Op::Type::cast: {
                SGL_ASSERT(value.type == op.cast.from);
                double tmp;
                if (DataStruct::is_integer(op.cast.from)) {
                    if (DataStruct::is_unsigned(op.cast.from))
                        tmp = static_cast<double>(value.u);
                    else
                        tmp = static_cast<double>(value.i);
                } else {
                    if (op.cast.from == DataStruct::Type::float64)
                        tmp = value.d;
                    else
                        tmp = value.s;
                }
                if (DataStruct::is_integer(op.cast.to)) {
                    if (DataStruct::is_unsigned(op.cast.to))
                        value.u = static_cast<uint64_t>(tmp);
                    else
                        value.i = static_cast<int64_t>(tmp);
                } else {
                    if (op.cast.to == DataStruct::Type::float64)
                        value.d = tmp;
                    else
                        value.s = static_cast<float>(tmp);
                }
                value.type = op.cast.to;
                break;
            }
            case Op::Type::linear_to_srgb:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                value.d = math::linear_to_srgb(value.d);
                break;
            case Op::Type::srgb_to_linear:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                value.d = math::srgb_to_linear(value.d);
                break;
            case Op::Type::multiply:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                value.d *= op.multiply.value;
                break;
            case Op::Type::multiply_add:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                SGL_ASSERT(registers[op.multiply_add.reg].type == DataStruct::Type::float64);
                value.d += registers[op.multiply_add.reg].d * op.multiply_add.factor;
                break;
            case Op::Type::round:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                value.d = std::rint(value.d);
                break;
            case Op::Type::clamp:
                SGL_ASSERT(value.type == DataStruct::Type::float64);
                value.d = std::clamp(value.d, op.clamp.min, op.clamp.max);
                break;
            }
        }
    }

    /// Load a value from memory.
    Value load(const uint8_t* base, size_t offset, DataStruct::Type type, bool swap)
    {
        Value value;
        value.type = type;

#define CASE(ctype, dst)                                                                                               \
    {                                                                                                                  \
        ctype v = *reinterpret_cast<const ctype*>(base + offset);                                                      \
        if (swap) [[unlikely]]                                                                                         \
            v = stdx::byteswap(v);                                                                                     \
        dst = v;                                                                                                       \
        break;                                                                                                         \
    }

        switch (type) {
        case DataStruct::Type::int8:
            value.i = *reinterpret_cast<const int8_t*>(base + offset);
            break;
        case DataStruct::Type::int16:
            CASE(int16_t, value.i)
        case DataStruct::Type::int32:
            CASE(int32_t, value.i)
        case DataStruct::Type::int64:
            CASE(int64_t, value.i)
        case DataStruct::Type::uint8:
            value.u = *reinterpret_cast<const uint8_t*>(base + offset);
            break;
        case DataStruct::Type::uint16:
            CASE(uint16_t, value.u)
        case DataStruct::Type::uint32:
            CASE(uint32_t, value.u)
        case DataStruct::Type::uint64:
            CASE(uint64_t, value.u)
        case DataStruct::Type::float16: {
            uint16_t v = *reinterpret_cast<const uint16_t*>(base + offset);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            value.s = math::float16_to_float32(v);
            break;
        }
        case DataStruct::Type::float32: {
            uint32_t v = *reinterpret_cast<const uint32_t*>(base + offset);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            value.s = stdx::bit_cast<float>(v);
            break;
        }
        case DataStruct::Type::float64: {
            uint64_t v = *reinterpret_cast<const uint64_t*>(base + offset);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            value.d = stdx::bit_cast<double>(v);
            break;
        }
        }

#undef CASE

        return value;
    }

    /// Save value to memory.
    void save(uint8_t* base, size_t offset, const Value& value, bool swap)
    {
#define CASE(ctype, src)                                                                                               \
    {                                                                                                                  \
        ctype v = static_cast<ctype>(src);                                                                             \
        if (swap) [[unlikely]]                                                                                         \
            v = stdx::byteswap(v);                                                                                     \
        *reinterpret_cast<ctype*>(base + offset) = v;                                                                  \
        break;                                                                                                         \
    }

        switch (value.type) {
        case DataStruct::Type::int8:
            *reinterpret_cast<int8_t*>(base + offset) = static_cast<int8_t>(value.i);
            break;
        case DataStruct::Type::int16:
            CASE(int16_t, value.i)
        case DataStruct::Type::int32:
            CASE(int32_t, value.i)
        case DataStruct::Type::int64:
            CASE(int64_t, value.i)
        case DataStruct::Type::uint8:
            *reinterpret_cast<uint8_t*>(base + offset) = static_cast<uint8_t>(value.u);
            break;
        case DataStruct::Type::uint16:
            CASE(uint16_t, value.u)
        case DataStruct::Type::uint32:
            CASE(uint32_t, value.u)
        case DataStruct::Type::uint64:
            CASE(uint64_t, value.u)
        case DataStruct::Type::float16: {
            uint16_t v = math::float32_to_float16(value.s);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            *reinterpret_cast<uint16_t*>(base + offset) = v;
            break;
        }
        case DataStruct::Type::float32: {
            uint32_t v = stdx::bit_cast<uint32_t>(value.s);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            *reinterpret_cast<uint32_t*>(base + offset) = v;
            break;
        }
        case DataStruct::Type::float64: {
            uint64_t v = stdx::bit_cast<uint64_t>(value.d);
            if (swap) [[unlikely]]
                v = stdx::byteswap(v);
            *reinterpret_cast<uint64_t*>(base + offset) = v;
            break;
        }
        }

#undef CASE
    }
};

/// Generate conversion program for converting from \c src_struct to \c dst_struct.
/// This generates code for the virtual machine.
/// If using a JIT compiler, this code can be compiled to native code.
std::vector<Op> generate_code(const DataStruct& src_struct, const DataStruct& dst_struct)
{
    const bool src_swap = src_struct.byte_order() != DataStruct::host_byte_order();
    const bool dst_swap = dst_struct.byte_order() != DataStruct::host_byte_order();

    std::vector<Op> code;
    std::map<std::string, uint8_t> src_regs;

    for (const auto& dst_field : dst_struct) {

        if (!dst_field.blend.empty()) {
            // Initialize accumulator.
            code.push_back({.type = Op::Type::load_imm, .reg = 0, .load_imm = {0.0}});

            // Accumulate source fields.
            for (const auto& [weight, name] : dst_field.blend) {
                const auto& src_field = src_struct.field(name);
                const auto src_range = DataStruct::type_range(src_field.type);
                uint8_t src_reg;
                auto it = src_regs.find(name);
                if (it != src_regs.end()) {
                    src_reg = it->second;
                } else {
                    // Load linear value from source.
                    src_reg = static_cast<uint8_t>(src_regs.size() + 1);
                    src_regs.emplace(name, src_reg);

                    // Load value from source struct.
                    code.push_back(
                        {.type = Op::Type::load_mem,
                         .reg = src_reg,
                         .load_mem = {src_field.offset, src_field.type, src_swap}}
                    );

                    // Convert to double.
                    code.push_back(
                        {.type = Op::Type::cast, .reg = src_reg, .cast = {src_field.type, DataStruct::Type::float64}}
                    );

                    // Normalize source value.
                    if (DataStruct::is_integer(src_field.type)
                        && is_set(src_field.flags, DataStruct::Flags::normalized))
                        code.push_back(
                            {.type = Op::Type::multiply, .reg = src_reg, .multiply = {1.0 / src_range.second}}
                        );

                    // Linearize source value.
                    if (is_set(src_field.flags, DataStruct::Flags::srgb_gamma))
                        code.push_back({.type = Op::Type::srgb_to_linear, .reg = src_reg});
                }

                // Add weighted value to accumulator.
                code.push_back({.type = Op::Type::multiply_add, .reg = 0, .multiply_add = {src_reg, weight}});
            }
            const auto dst_range = DataStruct::type_range(dst_field.type);

            // De-linearize destination value.
            if (is_set(dst_field.flags, DataStruct::Flags::srgb_gamma))
                code.push_back({.type = Op::Type::linear_to_srgb, .reg = 0});

            // De-normalize destination value.
            if (DataStruct::is_integer(dst_field.type) && is_set(dst_field.flags, DataStruct::Flags::normalized))
                code.push_back({.type = Op::Type::multiply, .reg = 0, .multiply = {dst_range.second}});

            // Round and clamp integers.
            if (DataStruct::is_integer(dst_field.type)) {
                code.push_back({.type = Op::Type::round, .reg = 0});
                code.push_back({.type = Op::Type::clamp, .reg = 0, .clamp = {dst_range.first, dst_range.second}});
            }

            code.push_back({.type = Op::Type::cast, .reg = 0, .cast = {DataStruct::Type::float64, dst_field.type}});

            // Save value to destination struct.
            code.push_back(
                {.type = Op::Type::save_mem, .reg = 0, .save_mem = {dst_field.offset, dst_field.type, dst_swap}}
            );

        } else if (src_struct.has_field(dst_field.name)) {
            const auto& src_field = src_struct.field(dst_field.name);

            // Load value from source struct.
            code.push_back(
                {.type = Op::Type::load_mem, .reg = 0, .load_mem = {src_field.offset, src_field.type, src_swap}}
            );

            // Convert value if types don't match.
            DataStruct::Flags flag_mask = DataStruct::Flags::normalized | DataStruct::Flags::srgb_gamma;
            if (src_field.type != dst_field.type || (src_field.flags & flag_mask) != (dst_field.flags & flag_mask)) {
                const auto src_range = DataStruct::type_range(src_field.type);
                const auto dst_range = DataStruct::type_range(dst_field.type);

                // Convert to double.
                code.push_back({.type = Op::Type::cast, .reg = 0, .cast = {src_field.type, DataStruct::Type::float64}});

                // Normalize source value.
                if (DataStruct::is_integer(src_field.type) && is_set(src_field.flags, DataStruct::Flags::normalized))
                    code.push_back({.type = Op::Type::multiply, .reg = 0, .multiply = {1.0 / src_range.second}});

                // Linearize source value.
                if (is_set(src_field.flags, DataStruct::Flags::srgb_gamma))
                    code.push_back({.type = Op::Type::srgb_to_linear, .reg = 0});

                // De-linearize destination value.
                if (is_set(dst_field.flags, DataStruct::Flags::srgb_gamma))
                    code.push_back({.type = Op::Type::linear_to_srgb, .reg = 0});

                // De-normalize destination value.
                if (DataStruct::is_integer(dst_field.type) && is_set(dst_field.flags, DataStruct::Flags::normalized))
                    code.push_back({.type = Op::Type::multiply, .reg = 0, .multiply = {dst_range.second}});

                // Round and clamp integers.
                if (DataStruct::is_integer(dst_field.type)) {
                    code.push_back({.type = Op::Type::round, .reg = 0});
                    code.push_back({.type = Op::Type::clamp, .reg = 0, .clamp = {dst_range.first, dst_range.second}});
                }

                code.push_back({.type = Op::Type::cast, .reg = 0, .cast = {DataStruct::Type::float64, dst_field.type}});
            }

            // Save value to destination struct.
            code.push_back(
                {.type = Op::Type::save_mem, .reg = 0, .save_mem = {dst_field.offset, dst_field.type, dst_swap}}
            );
        } else if (is_set(dst_field.flags, DataStruct::Flags::default_)) {
            // Set default value.
            code.push_back({.type = Op::Type::load_imm, .reg = 0, .load_imm = {dst_field.default_value}});
            code.push_back({.type = Op::Type::cast, .reg = 0, .cast = {DataStruct::Type::float64, dst_field.type}});
            code.push_back(
                {.type = Op::Type::save_mem, .reg = 0, .save_mem = {dst_field.offset, dst_field.type, dst_swap}}
            );
        } else {
            SGL_THROW("Field \"{}\" not found in source struct.", dst_field.name);
        }
    }

    return code;
}


/// Interface for conversion programs.
struct Program {
    virtual ~Program() = default;
    virtual void execute(const void* src, void* dst, size_t count) const = 0;
};

/// Conversion program for the virtual machine.
struct VMProgram : public Program {
    std::vector<Op> code;
    size_t src_size;
    size_t dst_size;

    void execute(const void* src, void* dst, size_t count) const override
    {
        VM vm;
        vm.src = static_cast<const uint8_t*>(src);
        vm.dst = static_cast<uint8_t*>(dst);

        for (size_t i = 0; i < count; ++i) {
            vm.run(code);
            vm.src += src_size;
            vm.dst += dst_size;
        }
    }

    static std::unique_ptr<Program> compile(const DataStruct& src_struct, const DataStruct& dst_struct)
    {
        auto program = std::make_unique<VMProgram>();
        program->code = generate_code(src_struct, dst_struct);
        program->src_size = src_struct.size();
        program->dst_size = dst_struct.size();
        return program;
    }
};

#if SGL_HAS_ASMJIT

/// Conversion program running just-in-time compiled X86 code.
struct X86Program : public Program {
    using ConvertFunc = void (*)(const void* src, void* dst, size_t count);

    ConvertFunc func;

    void execute(const void* src, void* dst, size_t count) const override { func(src, dst, count); }

    static std::unique_ptr<Program> compile(const DataStruct& src_struct, const DataStruct& dst_struct)
    {
        asmjit::CodeHolder code;
        code.init(runtime().environment(), runtime().cpuFeatures());

#if SGL_LOG_JIT_ASSEMBLY
        log_info("Compiling x86 program for converting from {} to {}", src_struct.to_string(), dst_struct.to_string());
        asmjit::StringLogger logger;
        logger.setFlags(asmjit::FormatFlags::kMachineCode);
        code.setLogger(&logger);
#endif
        asmjit::x86::Compiler c(&code);

        Builder builder(c);
        builder.build(src_struct, dst_struct);

        asmjit::Error err = c.finalize();
        if (err != asmjit::kErrorOk) {
            SGL_THROW("AsmJit failed: {}", asmjit::DebugUtils::errorAsString(err));
        }

#if SGL_LOG_JIT_ASSEMBLY
        log_info(logger.content().data());
#endif

        ConvertFunc func;
        runtime().add(&func, &code);

        auto program = std::make_unique<X86Program>();
        program->func = func;
        return program;
    }

private:
    /// Helper class to build X86 code.
    struct Builder {
        asmjit::x86::Compiler& c;
        bool has_avx;
        bool has_f16c;

        // Each register can hold either a 64-bit integer or a 64-bit floating point value.
        struct Register {
            uint32_t index;
            asmjit::x86::Gp gp;
            asmjit::x86::Xmm xmm;
        };

        std::map<uint32_t, Register> registers;

        Builder(asmjit::x86::Compiler& c)
            : c(c)
        {
            has_avx = runtime().cpuFeatures().x86().hasAVX();
            has_f16c = runtime().cpuFeatures().x86().hasF16C();
        }

        Register& get_register(uint32_t reg)
        {
            auto it = registers.find(reg);
            if (it != registers.end())
                return it->second;
            auto [it2, inserted] = registers.emplace(reg, Register{.index = reg});
            return it2->second;
        }

        /// Load a constant value.
        asmjit::x86::Mem const_(double value) { return c.newDoubleConst(asmjit::ConstPoolScope::kGlobal, value); }

        /// Move operation (double word).
        template<typename X, typename Y>
        void movd(const X& x, const Y& y)
        {
            has_avx ? c.vmovd(x, y) : c.movd(x, y);
        }

        /// Move operation (quad word).
        template<typename X, typename Y>
        void movq(const X& x, const Y& y)
        {
            has_avx ? c.vmovq(x, y) : c.movq(x, y);
        }

        /// Move operation (scalar single).
        template<typename X, typename Y>
        void movss(const X& x, const Y& y)
        {
            has_avx ? c.vmovss(x, y) : c.movss(x, y);
        }

        /// Move operation (scalar double).
        template<typename X, typename Y>
        void movsd(const X& x, const Y& y)
        {
            has_avx ? c.vmovsd(x, y) : c.movsd(x, y);
        }

        void movsd(const asmjit::x86::Xmm& x, const asmjit::x86::Xmm& y)
        {
            has_avx ? c.vmovsd(x, x, y) : c.movsd(x, y);
        }

        /// Floating point comparison.
        template<typename X, typename Y>
        void ucomisd(const X& x, const Y& y)
        {
            has_avx ? c.vucomisd(x, y) : c.ucomisd(x, y);
        }

        /// Floating point addition operation.
        template<typename X, typename Y>
        void addsd(const X& x, const Y& y)
        {
            has_avx ? c.vaddsd(x, x, y) : c.addsd(x, y);
        }

        /// Floating point subtraction operation.
        template<typename X, typename Y, typename Z>
        void subsd(const X& x, const Y& y, const Z& z)
        {
            if (has_avx) {
                c.vsubsd(x, y, z);
            } else {
                c.movsd(x, y);
                c.subsd(x, z);
            }
        }

        /// Floating point multiplication operation.
        template<typename X, typename Y>
        void mulsd(const X& x, const Y& y)
        {
            has_avx ? c.vmulsd(x, x, y) : c.mulsd(x, y);
        }

        /// Floating point division operation.
        template<typename X, typename Y>
        void divsd(const X& x, const Y& y)
        {
            has_avx ? c.vdivsd(x, x, y) : c.divsd(x, y);
        }

        /// Floating point square root operation.
        template<typename X, typename Y>
        void sqrtsd(const X& x, const Y& y)
        {
            has_avx ? c.vsqrtsd(x, x, y) : c.sqrtsd(x, y);
        }

        /// Floating point maximum operation.
        template<typename X, typename Y>
        void maxsd(const X& x, const Y& y)
        {
            has_avx ? c.vmaxsd(x, x, y) : c.maxsd(x, y);
        }

        /// Floating point minimum operation.
        template<typename X, typename Y>
        void minsd(const X& x, const Y& y)
        {
            has_avx ? c.vminsd(x, x, y) : c.minsd(x, y);
        }

        /// Floating point rounding operation.
        template<typename X, typename Y>
        void roundsd(const X& x, const Y& y, asmjit::x86::RoundImm mode)
        {
            has_avx ? c.vroundsd(x, x, y, mode) : c.roundsd(x, y, mode);
        }

        /// Convert scalar double to signed integer.
        template<typename X, typename Y>
        void cvtsd2si(const X& x, const Y& y)
        {
            has_avx ? c.vcvtsd2si(x, y) : c.cvtsd2si(x, y);
        }

        /// Convert signed integer to scalar double.
        template<typename X, typename Y>
        void cvtsi2sd(const X& x, const Y& y)
        {
            has_avx ? c.vcvtsi2sd(x, x, y) : c.cvtsi2sd(x, y);
        }

        /// Convert scalar double to scalar float.
        template<typename X, typename Y>
        void cvtsd2ss(const X& x, const Y& y)
        {
            has_avx ? c.vcvtsd2ss(x, x, y) : c.cvtsd2ss(x, y);
        }

        /// Floating point fused multiply-add operation.
        template<typename X, typename Y, typename Z>
        void fmadd213sd(const X& x, const Y& y, const Z& z)
        {
            if (has_avx) {
                c.vfmadd213sd(x, y, z);
            } else {
                c.mulsd(x, y);
                c.addsd(x, z);
            }
        }

        /// Floating point fused multiply-add operation.
        template<typename X, typename Y, typename Z>
        void fmadd231sd(const X& x, const Y& y, const Z& z)
        {
            if (has_avx) {
                c.vfmadd231sd(x, y, z);
            } else {
                c.mulsd(y, z);
                c.addsd(x, y);
            }
        }

        void build(const DataStruct& src_struct, const DataStruct& dst_struct)
        {
            std::vector<Op> ops = generate_code(src_struct, dst_struct);

            auto comment = [this](std::string text)
            {
                text = "### " + text;
                c.comment(text.c_str());
            };

            auto node
                = c.addFunc(asmjit::FuncSignature::build<void, const void*, void*, size_t>(asmjit::CallConvId::kHost));
            auto src = c.newIntPtr("src");
            auto dst = c.newIntPtr("dst");
            auto count = c.newInt64("count");
            auto idx = c.newInt64("idx");

            node->setArg(0, src);
            node->setArg(1, dst);
            node->setArg(2, count);

            asmjit::Label loop_start = c.newLabel();
            asmjit::Label loop_end = c.newLabel();

            c.test(count, count);
            c.jz(loop_end);
            c.xor_(idx, idx);

            c.bind(loop_start);

            for (const Op& op : ops) {
                using namespace asmjit;

                switch (op.type) {
                case Op::Type::load_mem: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format(
                        "load_mem (reg={}, type={}, offset={}, swap={})",
                        reg.index,
                        op.load_mem.type,
                        op.load_mem.offset,
                        op.load_mem.swap
                    ));
                    load(reg, src, static_cast<int32_t>(op.load_mem.offset), op.load_mem.type, op.load_mem.swap);
                    break;
                }
                case Op::Type::load_imm: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("load_imm (reg={}, value={})", reg.index, op.load_imm.value));
                    reg.xmm = c.newXmm();
                    movsd(reg.xmm, const_(op.load_imm.value));
                    break;
                }
                case Op::Type::save_mem: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format(
                        "save_mem (reg={}, type={}, offset={}, swap={})",
                        reg.index,
                        op.save_mem.type,
                        op.save_mem.offset,
                        op.save_mem.swap
                    ));
                    save(reg, dst, static_cast<int32_t>(op.save_mem.offset), op.save_mem.type, op.save_mem.swap);
                    break;
                }
                case Op::Type::cast: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("cast (reg={}, from={}, to={})", reg.index, op.cast.from, op.cast.to));
                    cast(reg, op.cast.from, op.cast.to);
                    break;
                }
                case Op::Type::linear_to_srgb: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("linear_to_srgb (reg={})", op.reg));
                    reg.xmm = gamma(reg.xmm, true);
                    break;
                }
                case Op::Type::srgb_to_linear: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("srgb_to_linear (reg={})", op.reg));
                    reg.xmm = gamma(reg.xmm, false);
                    break;
                }
                case Op::Type::multiply: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("multiply (reg={}, value={})", reg.index, op.multiply.value));
                    mulsd(reg.xmm, const_(op.multiply.value));
                    break;
                }
                case Op::Type::multiply_add: {
                    const Register& reg1 = get_register(op.reg);
                    const Register& reg2 = get_register(op.multiply_add.reg);
                    comment(fmt::format(
                        "multiply_add (reg1={}, reg2={}, factor={})",
                        reg1.index,
                        reg2.index,
                        op.multiply_add.factor
                    ));
                    fmadd231sd(reg1.xmm, reg2.xmm, const_(op.multiply_add.factor));
                    break;
                }
                case Op::Type::round: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("round (reg={})", reg.index));
                    roundsd(reg.xmm, reg.xmm, asmjit::x86::RoundImm::kNearest);
                    break;
                }
                case Op::Type::clamp: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("clamp (reg={}, min={}, max={})", reg.index, op.clamp.min, op.clamp.max));
                    maxsd(reg.xmm, const_(op.clamp.min));
                    minsd(reg.xmm, const_(op.clamp.max));
                    break;
                }
                }
            }

            c.inc(idx);
            c.add(src, asmjit::Imm(src_struct.size()));
            c.add(dst, asmjit::Imm(dst_struct.size()));
            c.cmp(idx, count);
            c.jne(loop_start);

            c.bind(loop_end);

            c.ret();
            c.endFunc();
        };

        /// Load a value from memory (base + offset) to a register.
        /// Optionally swaps the endianess of the value.
        /// Integer values are stored in a general purpose register.
        /// Floating point values are stored in a XMM register.
        /// Half precision floating point values are converted to single precision,
        /// either using the F16C instruction set or a software implementation.
        void load(Register& reg, const asmjit::x86::Gp& base, int32_t offset, DataStruct::Type type, bool swap)
        {
            using namespace asmjit;

            if (DataStruct::is_integer(type))
                reg.gp = c.newInt64();
            else
                reg.xmm = c.newXmm();

            switch (type) {
            case DataStruct::Type::int8:
                c.movsx(reg.gp, x86::byte_ptr(base, offset));
                break;
            case DataStruct::Type::int16:
                if (swap) {
                    x86::Gp tmp = c.newUInt16();
                    c.movzx(tmp, x86::word_ptr(base, offset));
                    c.xchg(tmp.r8Lo(), tmp.r8Hi());
                    c.movsx(reg.gp, tmp);
                } else {
                    c.movsx(reg.gp, x86::word_ptr(base, offset));
                }
                break;
            case DataStruct::Type::int32:
                if (swap) {
                    x86::Gp tmp = c.newUInt32();
                    c.mov(tmp, x86::dword_ptr(base, offset));
                    c.bswap(tmp);
                    c.movsxd(reg.gp, tmp);
                } else {
                    c.movsxd(reg.gp.r32(), x86::dword_ptr(base, offset));
                }
                break;
            case DataStruct::Type::int64:
                c.mov(reg.gp, x86::qword_ptr(base, offset));
                if (swap)
                    c.bswap(reg.gp.r64());
                break;
            case DataStruct::Type::uint8:
                c.movzx(reg.gp, x86::byte_ptr(base, offset));
                break;
            case DataStruct::Type::uint16:
                c.movzx(reg.gp, x86::word_ptr(base, offset));
                if (swap)
                    c.xchg(reg.gp.r8Lo(), reg.gp.r8Hi());
                break;
            case DataStruct::Type::uint32:
                c.mov(reg.gp.r32(), x86::dword_ptr(base, offset));
                if (swap)
                    c.bswap(reg.gp.r32());
                break;
            case DataStruct::Type::uint64:
                c.mov(reg.gp, x86::qword_ptr(base, offset));
                if (swap)
                    c.bswap(reg.gp.r64());
                break;
            case DataStruct::Type::float16: {
                x86::Gp tmp = c.newUInt16();
                c.movzx(tmp, x86::word_ptr(base, offset));
                if (swap)
                    c.xchg(tmp.r8Lo(), tmp.r8Hi());
                if (has_avx && has_f16c) {
                    c.vmovq(reg.xmm, tmp);
                    c.vcvtph2ps(reg.xmm, reg.xmm);
                } else {
                    InvokeNode* node;
                    c.invoke(
                        &node,
                        imm((void*)math::float16_to_float32),
                        FuncSignature::build<float, uint16_t>(CallConvId::kHost)
                    );
                    node->setArg(0, tmp);
                    node->setRet(0, reg.xmm);
                }
                break;
            }
            case DataStruct::Type::float32:
                if (swap) {
                    x86::Gp tmp = c.newUInt32();
                    c.mov(tmp, x86::dword_ptr(base, offset));
                    c.bswap(tmp);
                    movd(reg.xmm, tmp);
                } else {
                    movss(reg.xmm, x86::dword_ptr(base, offset));
                }
                break;
            case DataStruct::Type::float64:
                if (swap) {
                    x86::Gp tmp = c.newUInt64();
                    c.mov(tmp, x86::qword_ptr(base, offset));
                    c.bswap(tmp);
                    movq(reg.xmm, tmp);
                } else {
                    movsd(reg.xmm, x86::qword_ptr(base, offset));
                }
                break;
            }
        }

        /// Save a value from a register to memory (base + offset).
        /// Optionally swaps the endianess of the value.
        void save(const Register& reg, const asmjit::x86::Gp& base, int32_t offset, DataStruct::Type type, bool swap)
        {
            using namespace asmjit;

            switch (type) {
            case DataStruct::Type::int8:
            case DataStruct::Type::uint8:
                c.mov(x86::byte_ptr(base, offset), reg.gp.r8());
                break;
            case DataStruct::Type::int16:
            case DataStruct::Type::uint16:
                if (swap) {
                    x86::Gp tmp = c.newUInt16();
                    c.mov(tmp, reg.gp.r16());
                    c.xchg(tmp.r8Lo(), tmp.r8Hi());
                    c.mov(x86::word_ptr(base, offset), tmp.r16());
                } else {
                    c.mov(x86::word_ptr(base, offset), reg.gp.r16());
                }
                break;
            case DataStruct::Type::int32:
            case DataStruct::Type::uint32:
                if (swap) {
                    x86::Gp tmp = c.newUInt32();
                    c.mov(tmp, reg.gp.r32());
                    c.bswap(tmp);
                    c.mov(x86::dword_ptr(base, offset), tmp.r32());
                } else {
                    c.mov(x86::dword_ptr(base, offset), reg.gp.r32());
                }
                break;
            case DataStruct::Type::int64:
            case DataStruct::Type::uint64:
                if (swap) {
                    x86::Gp tmp = c.newUInt64();
                    c.mov(tmp, reg.gp);
                    c.bswap(tmp);
                    c.mov(x86::qword_ptr(base, offset), tmp);
                } else {
                    c.mov(x86::qword_ptr(base, offset), reg.gp);
                }
                break;
            case DataStruct::Type::float16: {
                x86::Gp tmp2 = c.newUInt16();
                if (has_avx && has_f16c) {
                    x86::Xmm tmp3 = c.newXmm();
                    c.vcvtps2ph(tmp3, reg.xmm, 0);
                    c.vmovd(tmp2, tmp3);
                } else {
                    InvokeNode* node;
                    c.invoke(
                        &node,
                        imm((void*)math::float32_to_float16),
                        FuncSignature::build<uint16_t, float>(CallConvId::kHost)
                    );
                    node->setArg(0, reg.xmm);
                    node->setRet(0, tmp2);
                }
                if (swap)
                    c.xchg(tmp2.r8Lo(), tmp2.r8Hi());
                c.mov(x86::word_ptr(base, offset), tmp2.r16());
                break;
            }
            case DataStruct::Type::float32:
                if (swap) {
                    x86::Gp tmp = c.newUInt32();
                    movd(tmp, reg.xmm);
                    c.bswap(tmp);
                    c.mov(x86::dword_ptr(base, offset), tmp);
                } else {
                    movss(x86::dword_ptr(base, offset), reg.xmm);
                }
                break;
            case DataStruct::Type::float64:
                if (swap) {
                    x86::Gp tmp = c.newUInt64();
                    movq(tmp, reg.xmm);
                    c.bswap(tmp);
                    c.mov(x86::qword_ptr(base, offset), tmp);
                } else {
                    movsd(x86::qword_ptr(base, offset), reg.xmm);
                }
                break;
            }
        }

        /// Convert a register from one type to another.
        void cast(Register& reg, DataStruct::Type from, DataStruct::Type to)
        {
            using namespace asmjit;

            x86::Xmm dbl = c.newXmm();
            if (DataStruct::is_integer(from)) {
                if (from == DataStruct::Type::uint32 || from == DataStruct::Type::int64) {
                    cvtsi2sd(dbl, reg.gp.r64());
                } else if (from == DataStruct::Type::uint64) {
                    auto tmp = c.newUInt64();
                    c.mov(tmp, reg.gp.r64());
                    auto tmp2 = c.newUInt64Const(asmjit::ConstPoolScope::kGlobal, 0x7fffffffffffffffull);
                    c.and_(tmp, tmp2);
                    cvtsi2sd(dbl, tmp.r64());
                    c.test(reg.gp.r64(), reg.gp.r64());
                    Label done = c.newLabel();
                    c.jns(done);
                    addsd(dbl, c.newUInt64Const(asmjit::ConstPoolScope::kGlobal, 0x8000000000000000ull));
                    c.bind(done);
                } else {
                    cvtsi2sd(dbl, reg.gp.r32());
                }
            } else {
                if (from == DataStruct::Type::float64)
                    movsd(dbl, reg.xmm);
                else
                    c.cvtss2sd(dbl, reg.xmm);
            }
            if (DataStruct::is_integer(to)) {
                if (to == DataStruct::Type::uint32 || to == DataStruct::Type::int64) {
                    reg.gp = c.newInt64();
                    cvtsd2si(reg.gp, dbl);
                } else if (to == DataStruct::Type::uint64) {
                    reg.gp = c.newInt64();
                    cvtsd2si(reg.gp.r64(), dbl);

                    x86::Xmm large_thresh = c.newXmm();
                    movsd(large_thresh, const_(9.223372036854776e18 /* 2^63 - 1 */));

                    x86::Xmm tmp = c.newXmm();
                    subsd(tmp, dbl, large_thresh);

                    x86::Gp tmp2 = c.newInt64();
                    cvtsd2si(tmp2, tmp);

                    x86::Gp large_result = c.newInt64();
                    c.mov(large_result, Imm(0x7fffffffffffffffull));
                    c.add(large_result, tmp2);

                    ucomisd(dbl, large_thresh);
                    c.cmovnb(reg.gp.r64(), large_result);
                } else {
                    reg.gp = c.newInt32();
                    cvtsd2si(reg.gp, dbl);
                }
            } else {
                reg.xmm = c.newXmm();
                if (to == DataStruct::Type::float64)
                    movsd(reg.xmm, dbl);
                else
                    cvtsd2ss(reg.xmm, dbl);
            }
        }

        /// Forward/inverse gamma correction using the sRGB profile
        asmjit::x86::Xmm gamma(asmjit::x86::Xmm x, bool to_srgb)
        {
            using namespace asmjit;

            x86::Xmm a = c.newXmm();
            x86::Xmm b = c.newXmm();

            movsd(a, const_(to_srgb ? 12.92 : (1.0 / 12.92)));
            ucomisd(x, const_(to_srgb ? 0.0031308 : 0.04045));

            Label low_value = c.newLabel();
            c.jb(low_value);

            x86::Xmm y;
            if (to_srgb) {
                y = c.newXmm();
                sqrtsd(y, x);
            } else {
                y = x;
            }

            // Rational polynomial fit, rel.err = 8*10^-15
            double to_srgb_coeffs[2][11] = {
                {
                    -0.0031151377052754843,
                    0.5838023820686707,
                    8.450947414259522,
                    27.901125077137042,
                    32.44669922192121,
                    15.374469584296442,
                    3.0477578489880823,
                    0.2263810267005674,
                    0.002531335520959116,
                    -0.00021805827098915798,
                    -3.7113872202050023e-6,
                },
                {
                    1.,
                    10.723011300050162,
                    29.70548706952188,
                    30.50364355650628,
                    13.297981743005433,
                    2.575446652731678,
                    0.21749170309546628,
                    0.007244514696840552,
                    0.00007045228641004039,
                    -8.387527630781522e-9,
                    2.2380622409188757e-11,
                },
            };

            // Rational polynomial fit, rel.err = 1.5*10^-15
            double from_srgb_coeffs[2][10] = {
                {
                    -342.62884098034357,
                    -3483.4445569178347,
                    -9735.250875334352,
                    -10782.158977031822,
                    -5548.704065887224,
                    -1446.951694673217,
                    -200.19589605282445,
                    -14.786385491859248,
                    -0.5489744177844188,
                    -0.008042950896814532,
                },
                {
                    1.,
                    -84.8098437770271,
                    -1884.7738197074218,
                    -8059.219012060384,
                    -11916.470977597566,
                    -7349.477378676199,
                    -2013.8039726540235,
                    -237.47722999429413,
                    -9.646075249097724,
                    -2.2132610916769585e-8,
                },
            };

            size_t ncoeffs
                = to_srgb ? std::extent_v<decltype(to_srgb_coeffs), 1> : std::extent_v<decltype(from_srgb_coeffs), 1>;

            for (size_t i = 0; i < ncoeffs; ++i) {
                for (int j = 0; j < 2; ++j) {
                    x86::Xmm& v = (j == 0) ? a : b;
                    x86::Mem coeff = const_(to_srgb ? to_srgb_coeffs[j][i] : from_srgb_coeffs[j][i]);
                    if (i == 0) {
                        movsd(v, coeff);
                    } else {
                        fmadd213sd(v, y, coeff);
                    }
                }
            }

            divsd(a, b);
            c.bind(low_value);
            mulsd(a, x);

            return a;
        }
    };

    static asmjit::JitRuntime& runtime()
    {
        static asmjit::JitRuntime instance;
        return instance;
    }
};

#if SGL_ARM64

/// Conversion program running just-in-time compiled ARM code.
struct ARMProgram : public Program {
    using ConvertFunc = void (*)(const void* src, void* dst, size_t count);

    ConvertFunc func;

    void execute(const void* src, void* dst, size_t count) const override { func(src, dst, count); }

    static std::unique_ptr<Program> compile(const DataStruct& src_struct, const DataStruct& dst_struct)
    {
        // TODO: check cpu features, but for some reason these are all false
        bool supported = true;
        // supported &= runtime().cpuFeatures().arm().hasSVE();
        // supported &= runtime().cpuFeatures().arm().hasSVE_F32MM();
        // supported &= runtime().cpuFeatures().arm().hasSVE_F64MM();
        // supported &= runtime().cpuFeatures().arm().hasFRINT();
        // supported &= runtime().cpuFeatures().arm().hasFP16CONV();
        if (!supported) {
            log_warn_once("DataStruct converter cannot use AArch64 JIT: unsupported CPU features");
            return nullptr;
        }

        asmjit::CodeHolder code;
        code.init(runtime().environment(), runtime().cpuFeatures());

#if SGL_LOG_JIT_ASSEMBLY
        log_info(
            "Compiling Aarch64 program for converting from {} to {}",
            src_struct.to_string(),
            dst_struct.to_string()
        );
        asmjit::StringLogger logger;
        logger.setFlags(asmjit::FormatFlags::kMachineCode);
        code.setLogger(&logger);
#endif
        asmjit::a64::Compiler c(&code);

        Builder builder(c);
        builder.build(src_struct, dst_struct);

        asmjit::Error err = c.finalize();
        if (err != asmjit::kErrorOk) {
            SGL_THROW("AsmJit failed: {}", asmjit::DebugUtils::errorAsString(err));
        }

#if SGL_LOG_JIT_ASSEMBLY
        log_info(logger.content().data());
#endif

        ConvertFunc func;
        runtime().add(&func, &code);

        auto program = std::make_unique<X86Program>();
        program->func = func;
        return program;
    }

private:
    /// Helper class to build X86 code.
    struct Builder {
        asmjit::a64::Compiler& c;

        // Each register can hold either a 64-bit integer or a 64-bit floating point value.
        struct Register {
            uint32_t index;
            asmjit::a64::Gp gp;
            asmjit::a64::Vec vec;
        };

        std::map<uint32_t, Register> registers;

        Builder(asmjit::a64::Compiler& c)
            : c(c)
        {
        }

        Register& get_register(uint32_t reg)
        {
            auto it = registers.find(reg);
            if (it != registers.end())
                return it->second;
            auto [it2, inserted] = registers.emplace(reg, Register{.index = reg});
            return it2->second;
        }

        /// Load a constant value.
        asmjit::a64::Mem const_(double value) { return c.newDoubleConst(asmjit::ConstPoolScope::kGlobal, value); }

        void build(const DataStruct& src_struct, const DataStruct& dst_struct)
        {
            std::vector<Op> ops = generate_code(src_struct, dst_struct);

            auto comment = [this](std::string text)
            {
                text = "### " + text;
                c.comment(text.c_str());
            };

            auto node
                = c.addFunc(asmjit::FuncSignature::build<void, const void*, void*, size_t>(asmjit::CallConvId::kHost));
            auto src = c.newIntPtr("src");
            auto dst = c.newIntPtr("dst");
            auto count = c.newInt64("count");
            auto idx = c.newInt64("idx");

            node->setArg(0, src);
            node->setArg(1, dst);
            node->setArg(2, count);

            asmjit::Label loop_start = c.newLabel();
            asmjit::Label loop_end = c.newLabel();

            c.cmp(count, asmjit::a64::xzr);
            c.b_eq(loop_end);
            c.mov(idx, asmjit::a64::xzr);

            c.bind(loop_start);

            for (const Op& op : ops) {
                using namespace asmjit;

                switch (op.type) {
                case Op::Type::load_mem: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format(
                        "load_mem (reg={}, type={}, offset={}, swap={})",
                        reg.index,
                        op.load_mem.type,
                        op.load_mem.offset,
                        op.load_mem.swap
                    ));
                    load(reg, src, static_cast<int32_t>(op.load_mem.offset), op.load_mem.type, op.load_mem.swap);
                    break;
                }
                case Op::Type::load_imm: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("load_imm (reg={}, value={})", reg.index, op.load_imm.value));
                    reg.vec = c.newVecD();
                    c.ldr(reg.vec, const_(op.load_imm.value));
                    break;
                }
                case Op::Type::save_mem: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format(
                        "save_mem (reg={}, type={}, offset={}, swap={})",
                        reg.index,
                        op.save_mem.type,
                        op.save_mem.offset,
                        op.save_mem.swap
                    ));
                    save(reg, dst, static_cast<int32_t>(op.save_mem.offset), op.save_mem.type, op.save_mem.swap);
                    break;
                }
                case Op::Type::cast: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("cast (reg={}, from={}, to={})", reg.index, op.cast.from, op.cast.to));
                    cast(reg, op.cast.from, op.cast.to);
                    break;
                }
                case Op::Type::linear_to_srgb: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("linear_to_srgb (reg={})", op.reg));
                    reg.vec = gamma(reg.vec, true);
                    break;
                }
                case Op::Type::srgb_to_linear: {
                    Register& reg = get_register(op.reg);
                    comment(fmt::format("srgb_to_linear (reg={})", op.reg));
                    reg.vec = gamma(reg.vec, false);
                    break;
                }
                case Op::Type::multiply: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("multiply (reg={}, value={})", reg.index, op.multiply.value));
                    auto tmp = c.newVecD();
                    c.ldr(tmp, const_(op.multiply.value));
                    c.fmul(reg.vec, reg.vec, tmp);
                    break;
                }
                case Op::Type::multiply_add: {
                    const Register& reg1 = get_register(op.reg);
                    const Register& reg2 = get_register(op.multiply_add.reg);
                    comment(fmt::format(
                        "multiply_add (reg1={}, reg2={}, factor={})",
                        reg1.index,
                        reg2.index,
                        op.multiply_add.factor
                    ));
                    auto tmp = c.newVecD();
                    c.ldr(tmp, const_(op.multiply_add.factor));
                    c.fmadd(reg1.vec, reg2.vec, tmp, reg1.vec);
                    break;
                }
                case Op::Type::round: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("round (reg={})", reg.index));
                    c.frinta(reg.vec.d(), reg.vec.d());
                    break;
                }
                case Op::Type::clamp: {
                    const Register& reg = get_register(op.reg);
                    comment(fmt::format("clamp (reg={}, min={}, max={})", reg.index, op.clamp.min, op.clamp.max));
                    auto tmin = c.newVecD();
                    c.ldr(tmin, const_(op.clamp.min));
                    c.fmax(reg.vec, reg.vec, tmin);
                    auto tmax = c.newVecD();
                    c.ldr(tmax, const_(op.clamp.max));
                    c.fmin(reg.vec, reg.vec, tmax);
                    break;
                }
                }
            }

            c.add(idx, idx, asmjit::Imm(1));
            c.add(src, src, asmjit::Imm(src_struct.size()));
            c.add(dst, dst, asmjit::Imm(dst_struct.size()));
            c.cmp(idx, count);
            c.b_ne(loop_start);

            c.bind(loop_end);

            c.ret();
            c.endFunc();
        };

        /// Load a value from memory (base + offset) to a register.
        /// Optionally swaps the endianess of the value.
        /// Integer values are stored in a general purpose register.
        /// Floating point values are stored in a XMM register.
        /// Half precision floating point values are converted to single precision,
        /// either using the F16C instruction set or a software implementation.
        void load(Register& reg, const asmjit::a64::Gp& base, int32_t offset, DataStruct::Type type, bool swap)
        {
            using namespace asmjit;

            if (DataStruct::is_integer(type))
                reg.gp = c.newGpx();
            else
                reg.vec = c.newVecD();

            switch (type) {
            case DataStruct::Type::int8:
                c.ldrsb(reg.gp.x(), a64::ptr(base, offset));
                break;
            case DataStruct::Type::int16:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.ldrh(tmp, a64::ptr(base, offset));
                    c.rev16(tmp, tmp);
                    c.sxth(reg.gp.x(), tmp);
                } else {
                    c.ldrsh(reg.gp.x(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::int32:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.ldr(tmp, a64::ptr(base, offset));
                    c.rev32(tmp, tmp);
                    c.sxtw(reg.gp.x(), tmp.w());
                } else {
                    c.ldrsw(reg.gp.x(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::int64:
                c.ldr(reg.gp.x(), a64::ptr(base, offset));
                if (swap)
                    c.rev64(reg.gp.x(), reg.gp.x());
                break;
            case DataStruct::Type::uint8:
                c.ldrb(reg.gp.w(), a64::ptr(base, offset));
                break;
            case DataStruct::Type::uint16:
                c.ldrh(reg.gp.w(), a64::ptr(base, offset));
                if (swap)
                    c.rev16(reg.gp.w(), reg.gp.w());
                break;
            case DataStruct::Type::uint32:
                c.ldr(reg.gp.w(), a64::ptr(base, offset));
                if (swap)
                    c.rev32(reg.gp.w(), reg.gp.w());
                break;
            case DataStruct::Type::uint64:
                c.ldr(reg.gp.x(), a64::ptr(base, offset));
                if (swap)
                    c.rev64(reg.gp.x(), reg.gp.x());
                break;
            case DataStruct::Type::float16: {
                if (swap) {
                    auto tmp = c.newGpw();
                    c.ldrh(tmp, a64::ptr(base, offset));
                    c.rev16(tmp, tmp);
                    c.fmov(reg.vec.h(), tmp);
                } else {
                    c.ldr(reg.vec.h(), a64::ptr(base, offset));
                }
                break;
            }
            case DataStruct::Type::float32:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.ldr(tmp, a64::ptr(base, offset));
                    c.rev32(tmp, tmp);
                    c.fmov(reg.vec.s(), tmp);
                } else {
                    c.ldr(reg.vec.s(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::float64:
                if (swap) {
                    auto tmp = c.newGpx();
                    c.ldr(tmp, a64::ptr(base, offset));
                    c.rev64(tmp, tmp);
                    c.fmov(reg.vec.d(), tmp);
                } else {
                    c.ldr(reg.vec.d(), a64::ptr(base, offset));
                }
                break;
            }
        }

        /// Save a value from a register to memory (base + offset).
        /// Optionally swaps the endianess of the value.
        void save(const Register& reg, const asmjit::a64::Gp& base, int32_t offset, DataStruct::Type type, bool swap)
        {
            using namespace asmjit;

            switch (type) {
            case DataStruct::Type::int8:
            case DataStruct::Type::uint8:
                c.strb(reg.gp.w(), a64::ptr(base, offset));
                break;
            case DataStruct::Type::int16:
            case DataStruct::Type::uint16:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.mov(tmp, reg.gp.w());
                    c.rev16(tmp, tmp);
                    c.strh(tmp, a64::ptr(base, offset));
                } else {
                    c.strh(reg.gp.w(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::int32:
            case DataStruct::Type::uint32:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.mov(tmp, reg.gp.w());
                    c.rev32(tmp, tmp);
                    c.str(tmp, a64::ptr(base, offset));
                } else {
                    c.str(reg.gp.w(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::int64:
            case DataStruct::Type::uint64:
                if (swap) {
                    auto tmp = c.newGpx();
                    c.mov(tmp, reg.gp.x());
                    c.rev64(tmp, tmp);
                    c.str(tmp, a64::ptr(base, offset));
                } else {
                    c.str(reg.gp, a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::float16: {
                if (swap) {
                    auto tmp = c.newGpw();
                    c.fmov(tmp, reg.vec.h());
                    c.rev16(tmp, tmp);
                    c.strh(tmp, a64::ptr(base, offset));
                } else {
                    c.str(reg.vec.h(), a64::ptr(base, offset));
                }
                break;
            }
            case DataStruct::Type::float32:
                if (swap) {
                    auto tmp = c.newGpw();
                    c.fmov(tmp, reg.vec.s());
                    c.rev32(tmp, tmp);
                    c.str(tmp, a64::ptr(base, offset));
                } else {
                    c.str(reg.vec.s(), a64::ptr(base, offset));
                }
                break;
            case DataStruct::Type::float64:
                if (swap) {
                    auto tmp = c.newGpx();
                    c.fmov(tmp, reg.vec.d());
                    c.rev64(tmp, tmp);
                    c.str(tmp, a64::ptr(base, offset));
                } else {
                    c.str(reg.vec.d(), a64::ptr(base, offset));
                }
                break;
            }
        }

        /// Convert a register from one type to another.
        void cast(Register& reg, DataStruct::Type from, DataStruct::Type to)
        {
            using namespace asmjit;

            a64::Vec dbl = c.newVecD();
            if (DataStruct::is_integer(from)) {
                if (DataStruct::is_signed(from)) {
                    c.scvtf(dbl, reg.gp.x());
                } else {
                    c.ucvtf(dbl, reg.gp.x());
                }
            } else {
                if (from == DataStruct::Type::float16) {
                    c.fcvt(dbl, reg.vec.h());
                } else if (from == DataStruct::Type::float32) {
                    c.fcvt(dbl, reg.vec.s());
                } else {
                    c.fmov(dbl, reg.vec.d());
                }
            }
            if (DataStruct::is_integer(to)) {
                reg.gp = c.newGpx();
                if (DataStruct::is_signed(to)) {
                    c.fcvtzs(reg.gp.x(), dbl);
                } else {
                    c.fcvtzu(reg.gp.x(), dbl);
                }
            } else {
                reg.vec = c.newVecD();
                if (to == DataStruct::Type::float16) {
                    c.fcvt(reg.vec.h(), dbl);
                } else if (to == DataStruct::Type::float32) {
                    c.fcvt(reg.vec.s(), dbl);
                } else {
                    c.fmov(reg.vec.d(), dbl);
                }
            }
        }

        /// Forward/inverse gamma correction using the sRGB profile
        asmjit::a64::Vec gamma(asmjit::a64::Vec x, bool to_srgb)
        {
            using namespace asmjit;

            a64::Vec a = c.newVecD();
            a64::Vec b = c.newVecD();

            c.ldr(a, const_(to_srgb ? 12.92 : (1.0 / 12.92)));
            a64::Vec tmp = c.newVecD();
            c.ldr(tmp, const_(to_srgb ? 0.0031308 : 0.04045));
            c.fcmp(x, tmp);

            Label low_value = c.newLabel();
            // c.jb(low_value);
            c.b_lt(low_value);

            a64::Vec y;
            if (to_srgb) {
                y = c.newVecD();
                c.fsqrt(y, x);
            } else {
                y = x;
            }

            // Rational polynomial fit, rel.err = 8*10^-15
            double to_srgb_coeffs[2][11] = {
                {
                    -0.0031151377052754843,
                    0.5838023820686707,
                    8.450947414259522,
                    27.901125077137042,
                    32.44669922192121,
                    15.374469584296442,
                    3.0477578489880823,
                    0.2263810267005674,
                    0.002531335520959116,
                    -0.00021805827098915798,
                    -3.7113872202050023e-6,
                },
                {
                    1.,
                    10.723011300050162,
                    29.70548706952188,
                    30.50364355650628,
                    13.297981743005433,
                    2.575446652731678,
                    0.21749170309546628,
                    0.007244514696840552,
                    0.00007045228641004039,
                    -8.387527630781522e-9,
                    2.2380622409188757e-11,
                },
            };

            // Rational polynomial fit, rel.err = 1.5*10^-15
            double from_srgb_coeffs[2][10] = {
                {
                    -342.62884098034357,
                    -3483.4445569178347,
                    -9735.250875334352,
                    -10782.158977031822,
                    -5548.704065887224,
                    -1446.951694673217,
                    -200.19589605282445,
                    -14.786385491859248,
                    -0.5489744177844188,
                    -0.008042950896814532,
                },
                {
                    1.,
                    -84.8098437770271,
                    -1884.7738197074218,
                    -8059.219012060384,
                    -11916.470977597566,
                    -7349.477378676199,
                    -2013.8039726540235,
                    -237.47722999429413,
                    -9.646075249097724,
                    -2.2132610916769585e-8,
                },
            };

            size_t ncoeffs
                = to_srgb ? std::extent_v<decltype(to_srgb_coeffs), 1> : std::extent_v<decltype(from_srgb_coeffs), 1>;

            for (size_t i = 0; i < ncoeffs; ++i) {
                for (int j = 0; j < 2; ++j) {
                    a64::Vec& v = (j == 0) ? a : b;
                    a64::Mem coeff = const_(to_srgb ? to_srgb_coeffs[j][i] : from_srgb_coeffs[j][i]);
                    if (i == 0) {
                        c.ldr(v, coeff);
                    } else {
                        auto tmp = c.newVecD();
                        c.ldr(tmp, coeff);
                        c.fmadd(v, v, y, tmp);
                    }
                }
            }

            c.fdiv(a, a, b);
            c.bind(low_value);
            c.fmul(a, a, x);

            return a;
        }
    };

    static asmjit::JitRuntime& runtime()
    {
        static asmjit::JitRuntime instance;
        return instance;
    }
};

#endif // SGL_ARM64

#endif // SGL_HAS_ASMJIT


class ProgramCache {
public:
    const Program* get_program(const DataStruct& src_struct, const DataStruct& dst_struct)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        auto key = std::make_pair(src_struct, dst_struct);
        auto it = m_programs.find(key);
        if (it != m_programs.end())
            return it->second.get();
        auto [it2, inserted] = m_programs.emplace(key, compile_program(src_struct, dst_struct));
        return it2->second.get();
    }

    static ProgramCache& get()
    {
        static ProgramCache instance;
        return instance;
    }

private:
    std::unique_ptr<Program> compile_program(const DataStruct& src_struct, const DataStruct& dst_struct)
    {
        std::unique_ptr<Program> program;

#if SGL_HAS_ASMJIT
#if SGL_X86_64
        program = X86Program::compile(src_struct, dst_struct);
#elif SGL_ARM64
        program = ARMProgram::compile(src_struct, dst_struct);
#endif
#endif // SGL_HAS_ASMJIT
        if (!program)
            program = VMProgram::compile(src_struct, dst_struct);

        return program;
    }

    std::mutex m_mutex;
    std::unordered_map<
        std::pair<DataStruct, DataStruct>,
        std::unique_ptr<Program>,
        hasher<std::pair<DataStruct, DataStruct>>,
        comparator<std::pair<DataStruct, DataStruct>>>
        m_programs;
};


DataStructConverter::DataStructConverter(const DataStruct* src, const DataStruct* dst)
    : m_src(new DataStruct(*src))
    , m_dst(new DataStruct(*dst))
{
}

void DataStructConverter::convert(const void* src, void* dst, size_t count) const
{
    // Direct copy if source and destination struct are the same.
    if (*m_src == *m_dst) {
        std::memcpy(dst, src, m_src->size() * count);
        return;
    }

    const Program* program = ProgramCache::get().get_program(*m_src, *m_dst);
    SGL_CHECK(program, "Failed to compile conversion program.");
    program->execute(src, dst, count);
}

std::string DataStructConverter::to_string() const
{
    return fmt::format(
        "DataStructConverter(\n"
        "  src = {},\n"
        "  dst = {}\n"
        ")",
        string::indent(m_src->to_string()),
        string::indent(m_dst->to_string())
    );
}

} // namespace sgl
