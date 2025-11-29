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

class DerivedDummyObject : public DummyObject {
    SGL_OBJECT(DerivedDummyObject)
};

TEST_CASE("weak_ref")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    weak_ref<DummyObject> w1;
    CHECK(w1.expired());
    CHECK_EQ(w1.lock(), nullptr);

    {
        ref<DummyObject> r1 = make_ref<DummyObject>();
        CHECK_EQ(DummyObject::get_count(), 1);

        w1 = r1;
        CHECK_FALSE(w1.expired());

        ref<DummyObject> r2 = w1.lock();
        CHECK(r2);
        CHECK_EQ(r2, r1);
        // r1 + r2 = 2 references
        CHECK_EQ(r1->ref_count(), 2);
    }

    // Object should be gone now
    CHECK_EQ(DummyObject::get_count(), 0);
    CHECK(w1.expired());
    CHECK_EQ(w1.lock(), nullptr);

    // Test copy/move/assignment
    {
        ref<DummyObject> r1 = make_ref<DummyObject>();
        weak_ref<DummyObject> w2 = r1;
        weak_ref<DummyObject> w3 = w2; // Copy ctor
        CHECK_FALSE(w3.expired());
        CHECK_EQ(w3.lock(), r1);

        weak_ref<DummyObject> w4;
        w4 = w3; // Copy assign
        CHECK_FALSE(w4.expired());
        CHECK_EQ(w4.lock(), r1);

        weak_ref<DummyObject> w5 = std::move(w2); // Move ctor
        CHECK_FALSE(w5.expired());
        CHECK_EQ(w5.lock(), r1);

        weak_ref<DummyObject> w6;
        w6 = std::move(w3); // Move assign
        CHECK_FALSE(w6.expired());
        CHECK_EQ(w6.lock(), r1);
    }
    CHECK_EQ(DummyObject::get_count(), 0);

    // Test inheritance
    {
        ref<DerivedDummyObject> d1 = make_ref<DerivedDummyObject>();
        weak_ref<DummyObject> w_base = d1; // Construct from derived ref
        CHECK_FALSE(w_base.expired());
        CHECK_EQ(w_base.lock(), d1);

        weak_ref<DerivedDummyObject> w_derived = d1;
        weak_ref<DummyObject> w_base2 = w_derived; // Construct from derived weak_ref
        CHECK_FALSE(w_base2.expired());
        CHECK_EQ(w_base2.lock(), d1);

        w_base = w_derived; // Assign from derived weak_ref
        CHECK_EQ(w_base.lock(), d1);
    }
    CHECK_EQ(DummyObject::get_count(), 0);

    // Test comparison
    {
        ref<DummyObject> r1 = make_ref<DummyObject>();
        ref<DummyObject> r2 = make_ref<DummyObject>();
        weak_ref<DummyObject> w_cmp1 = r1;
        weak_ref<DummyObject> w_cmp1_copy = w_cmp1;
        weak_ref<DummyObject> w_cmp2 = r2;
        weak_ref<DummyObject> w_null;

        CHECK(w_cmp1 == w_cmp1_copy);
        CHECK(w_cmp1 != w_cmp2);
        CHECK(w_cmp1 != w_null);
        CHECK(w_null == nullptr);
        CHECK(w_cmp1 != nullptr);
    }
    CHECK_EQ(DummyObject::get_count(), 0);
}

TEST_SUITE_END();
