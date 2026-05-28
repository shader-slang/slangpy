// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <slang.h>
#include <slang-com-ptr.h>

#define SGL_CATCH_INTERNAL_SLANG_ERROR(expr)                                                                           \
    try {                                                                                                              \
        expr;                                                                                                          \
    } catch (...) {                                                                                                    \
        const char* slang_error = slang::getLastInternalErrorMessage();                                                \
        throw sgl::SlangCompileError(fmt::format("Internal slang error: {}", slang_error));                            \
    }

namespace sgl {

/// Implementation of slang's ISlangBlob interface to access an unowned blob of data.
class UnownedSlangBlob : public ISlangBlob {
public:
    UnownedSlangBlob(const void* data, size_t size)
        : m_data(data)
        , m_size(size)
    {
    }

    virtual SLANG_NO_THROW const void* SLANG_MCALL getBufferPointer() override { return m_data; }
    virtual SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return m_size; }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** outObject) override
    {
        if (uuid == SLANG_UUID_ISlangBlob) {
            *outObject = static_cast<ISlangBlob*>(this);
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override
    {
        // Do not perform any reference counting.
        return 2;
    }

    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        // Do not perform any reference counting.
        return 2;
    }

private:
    const void* m_data;
    size_t m_size;
};

} // namespace sgl
