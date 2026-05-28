// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/object.h"

using namespace sgl;

TEST_SUITE_BEGIN("object");

class DummyObject : public Object {
    SGL_OBJECT(DummyObject)
public:
    DummyObject() { get_count()++; }
    ~DummyObject() { get_count()--; }

    static uint32_t& get_count()
    {
        static uint32_t s_count = 0;
        return s_count;
    }
};

TEST_CASE("ref")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    ref<DummyObject> r1;
    ref<DummyObject> r2;

    CHECK_EQ(r1, r1);
    CHECK_EQ(r1, r2);
    CHECK_EQ(r1, nullptr);
    CHECK_FALSE(r1 != r1);
    CHECK_FALSE(r1 != r2);
    CHECK_FALSE(r1 != nullptr);
    CHECK_FALSE(bool(r1));
    CHECK_EQ(r1.get(), nullptr);

    r1 = make_ref<DummyObject>();
    CHECK_EQ(DummyObject::get_count(), 1);
    CHECK_EQ(r1->ref_count(), 1);

    CHECK_EQ(r1, r1);
    CHECK_FALSE(r1 == r2);
    CHECK_FALSE(r1 == nullptr);
    CHECK_FALSE(r1 != r1);
    CHECK_NE(r1, r2);
    CHECK_NE(r1, nullptr);
    CHECK(bool(r1));
    CHECK_NE(r1.get(), nullptr);

    r2 = r1;
    CHECK_EQ(DummyObject::get_count(), 1);
    CHECK_EQ(r1->ref_count(), 2);
    CHECK_EQ(r1, r2);
    CHECK_FALSE(r1 != r2);

    r2 = nullptr;
    CHECK_EQ(DummyObject::get_count(), 1);
    CHECK_EQ(r1->ref_count(), 1);

    r1 = nullptr;
    CHECK_EQ(DummyObject::get_count(), 0);
}

class DummyBuffer;

class DummyDevice : public Object {
    SGL_OBJECT(DummyDevice)
public:
    ref<DummyBuffer> buffer;

    DummyDevice() { get_count()++; }
    ~DummyDevice() { get_count()--; }

    static uint32_t& get_count()
    {
        static uint32_t s_count = 0;
        return s_count;
    }
};

class DummyBuffer : public Object {
    SGL_OBJECT(DummyBuffer)
public:
    breakable_ref<DummyDevice> device;

    DummyBuffer(ref<DummyDevice> device)
        : device(std::move(device))
    {
        get_count()++;
    }
    ~DummyBuffer() { get_count()--; }

    static uint32_t& get_count()
    {
        static uint32_t s_count = 0;
        return s_count;
    }
};

TEST_CASE("breakable_ref")
{
    REQUIRE_EQ(DummyDevice::get_count(), 0);
    REQUIRE_EQ(DummyBuffer::get_count(), 0);

    {
        ref<DummyDevice> device = make_ref<DummyDevice>();

        // Create a buffer that has a reference to the device -> cyclic reference
        device->buffer = make_ref<DummyBuffer>(device);

        CHECK_EQ(DummyDevice::get_count(), 1);
        CHECK_EQ(DummyBuffer::get_count(), 1);

        DummyBuffer* bufferPtr = device->buffer.get();

        // Release the device
        device = nullptr;

        // Device is not released as there is still a reference from the buffer
        CHECK_EQ(DummyDevice::get_count(), 1);
        CHECK_EQ(DummyBuffer::get_count(), 1);

        // Break the cycle
        bufferPtr->device.break_strong_reference();

        CHECK_EQ(DummyDevice::get_count(), 0);
        CHECK_EQ(DummyBuffer::get_count(), 0);
    }

    {
        ref<DummyDevice> device = make_ref<DummyDevice>();

        // Create a buffer that has a reference to the device -> cyclic reference
        device->buffer = make_ref<DummyBuffer>(device);
        // Immediately break the cycle
        device->buffer->device.break_strong_reference();

        CHECK_EQ(DummyDevice::get_count(), 1);
        CHECK_EQ(DummyBuffer::get_count(), 1);

        // Release the device
        device = nullptr;

        // Device is released as there is no strong reference from the buffer
        CHECK_EQ(DummyDevice::get_count(), 0);
        CHECK_EQ(DummyBuffer::get_count(), 0);
    }
}

TEST_SUITE_END();
