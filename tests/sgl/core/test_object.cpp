// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/object.h"

#include <array>
#include <atomic>
#include <mutex>
#include <thread>

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

class OrderedDestructionObject : public Object {
    SGL_OBJECT(OrderedDestructionObject)
public:
    explicit OrderedDestructionObject(std::atomic<uint32_t>* destroyed_value)
        : destroyed_value(destroyed_value)
    {
    }

    ~OrderedDestructionObject() { destroyed_value->store(value, std::memory_order_relaxed); }

    uint32_t value{0};
    std::atomic<uint32_t>* destroyed_value;
};

TEST_CASE("cross-thread final release observes prior object access")
{
    std::atomic<uint32_t> destroyed_value{0};
    std::atomic<bool> release_worker{false};
    ref<OrderedDestructionObject> object = make_ref<OrderedDestructionObject>(&destroyed_value);

    std::thread worker(
        [held_object = ref<OrderedDestructionObject>(object), &release_worker]() mutable
        {
            while (!release_worker.load(std::memory_order_relaxed))
                std::this_thread::yield();
            held_object = nullptr;
        }
    );

    object->value = 42;
    object = nullptr;
    release_worker.store(true, std::memory_order_relaxed);
    worker.join();

    CHECK(destroyed_value.load(std::memory_order_relaxed) == 42);
}

TEST_CASE("weak_ref")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    weak_ref<DummyObject> empty;
    CHECK(empty.expired());
    CHECK_FALSE(empty.lock());

    ref<DummyObject> strong = make_ref<DummyObject>();
    weak_ref<DummyObject> weak = strong;
    weak_ref<Object> base_weak = weak;
    weak_ref<const Object> const_weak = weak;

    CHECK_FALSE(weak.expired());
    CHECK_EQ(weak.lock().get(), strong.get());
    CHECK_EQ(base_weak.lock().get(), strong.get());
    CHECK_EQ(const_weak.lock().get(), strong.get());

    weak_ref<DummyObject> copied = weak;
    weak_ref<DummyObject> moved = std::move(copied);
    CHECK(copied.expired());
    CHECK_EQ(moved.lock().get(), strong.get());

    strong = nullptr;
    CHECK_EQ(DummyObject::get_count(), 0);
    CHECK(weak.expired());
    CHECK_FALSE(weak.lock());
    CHECK_FALSE(base_weak.lock());
    CHECK_FALSE(const_weak.lock());
    CHECK_FALSE(moved.lock());
}

TEST_CASE("weak_ref concurrent final release")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    for (uint32_t iteration = 0; iteration < 200; ++iteration) {
        ref<DummyObject> strong = make_ref<DummyObject>();
        weak_ref<DummyObject> weak = strong;
        std::atomic<bool> ready{false};
        std::atomic<bool> go{false};

        std::thread releaser(
            [strong = std::move(strong), &ready, &go]() mutable
            {
                ready.store(true, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                    std::this_thread::yield();
                strong = nullptr;
            }
        );

        while (!ready.load(std::memory_order_acquire))
            std::this_thread::yield();
        go.store(true, std::memory_order_release);

        ref<DummyObject> locked = weak.lock();
        releaser.join();
        locked = nullptr;

        REQUIRE(weak.expired());
        REQUIRE_FALSE(weak.lock());
        REQUIRE_EQ(DummyObject::get_count(), 0);
    }
}

TEST_CASE("weak_ref concurrent initialization")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    static constexpr size_t thread_count = 16;
    ref<DummyObject> strong = make_ref<DummyObject>();
    std::array<ref<DummyObject>, thread_count> owners;
    std::array<weak_ref<DummyObject>, thread_count> weak_refs;
    std::array<std::thread, thread_count> threads;
    std::atomic<size_t> ready{0};
    std::atomic<bool> go{false};

    for (ref<DummyObject>& owner : owners)
        owner = strong;

    for (size_t index = 0; index < thread_count; ++index) {
        threads[index] = std::thread(
            [&, index]
            {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                    std::this_thread::yield();
                weak_refs[index] = owners[index];
            }
        );
    }

    while (ready.load(std::memory_order_acquire) != thread_count)
        std::this_thread::yield();
    go.store(true, std::memory_order_release);

    for (std::thread& thread : threads)
        thread.join();
    for (weak_ref<DummyObject>& weak : weak_refs)
        REQUIRE_EQ(weak.lock().get(), strong.get());

    for (ref<DummyObject>& owner : owners)
        owner = nullptr;
    strong = nullptr;

    REQUIRE_EQ(DummyObject::get_count(), 0);
    for (weak_ref<DummyObject>& weak : weak_refs) {
        REQUIRE(weak.expired());
        REQUIRE_FALSE(weak.lock());
    }
}

#if SGL_ENABLE_REF_TRACKING
class TrackedDummyObject : public Object {
    SGL_OBJECT(TrackedDummyObject)
public:
    TrackedDummyObject() { set_enable_ref_tracking(true); }
};

TEST_CASE("weak_ref reference tracking")
{
    ref<TrackedDummyObject> strong = make_ref<TrackedDummyObject>();
    weak_ref<TrackedDummyObject> weak = strong;

    {
        ref<TrackedDummyObject> locked = weak.lock();
        REQUIRE_EQ(locked.get(), strong.get());
    }

    strong = nullptr;
    REQUIRE(weak.expired());
    REQUIRE_FALSE(weak.lock());
}
#endif

namespace {

struct FakePythonObject {
    std::atomic<Py_ssize_t_> ref_count{1};
    Object* object;
    std::atomic<bool> deallocating{false};
    std::atomic<uint32_t> invalid_inc_refs{0};
};

std::recursive_mutex& fake_python_gil()
{
    static std::recursive_mutex mutex;
    return mutex;
}

FakePythonObject* fake_python_object(PyObject* object)
{
    return reinterpret_cast<FakePythonObject*>(object);
}

void fake_python_inc_ref(PyObject* object) noexcept
{
    std::lock_guard lock(fake_python_gil());
    FakePythonObject* fake = fake_python_object(object);
    if (fake->deallocating.load(std::memory_order_relaxed) || fake->ref_count.load(std::memory_order_relaxed) <= 0) {
        fake->invalid_inc_refs.fetch_add(1, std::memory_order_relaxed);
    }
    fake->ref_count.fetch_add(1, std::memory_order_relaxed);
}

void fake_python_dec_ref(PyObject* object) noexcept
{
    std::lock_guard lock(fake_python_gil());
    FakePythonObject* fake = fake_python_object(object);
    if (fake->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (!fake->deallocating.exchange(true, std::memory_order_relaxed)) {
            delete fake->object;
            fake->deallocating.store(false, std::memory_order_relaxed);
        }
    }
}

Py_ssize_t_ fake_python_ref_count(PyObject* object) noexcept
{
    std::lock_guard lock(fake_python_gil());
    return fake_python_object(object)->ref_count.load(std::memory_order_relaxed);
}

void fake_python_run_with_gil(void (*callback)(void*) noexcept, void* context) noexcept
{
    std::lock_guard lock(fake_python_gil());
    callback(context);
}

void init_fake_python()
{
    object_init_py(fake_python_inc_ref, fake_python_dec_ref, fake_python_ref_count, fake_python_run_with_gil);
}

} // namespace

TEST_CASE("weak_ref preserves Python reference counting")
{
    REQUIRE_EQ(DummyObject::get_count(), 0);

    init_fake_python();

    ref<DummyObject> strong = make_ref<DummyObject>();
    weak_ref<DummyObject> weak = strong;
    FakePythonObject python_object{1, strong.get()};

    {
        std::lock_guard lock(fake_python_gil());
        strong->set_self_py(reinterpret_cast<PyObject*>(&python_object));
    }
    CHECK_EQ(python_object.ref_count.load(), 2);

    strong = nullptr;
    CHECK_EQ(python_object.ref_count.load(), 1);
    CHECK_EQ(DummyObject::get_count(), 1);

    {
        ref<DummyObject> locked = weak.lock();
        REQUIRE(locked);
        CHECK_EQ(python_object.ref_count.load(), 2);
    }
    CHECK_EQ(python_object.ref_count.load(), 1);

    fake_python_dec_ref(reinterpret_cast<PyObject*>(&python_object));
    CHECK_EQ(python_object.ref_count.load(), 0);
    CHECK_EQ(DummyObject::get_count(), 0);
    CHECK(weak.expired());
    CHECK_FALSE(weak.lock());
    CHECK_EQ(python_object.invalid_inc_refs.load(), 0);
}

class DestructorLocksWeak : public Object {
    SGL_OBJECT(DestructorLocksWeak)
public:
    ~DestructorLocksWeak() override
    {
        if (weak && lock_succeeded)
            *lock_succeeded = bool(weak->lock());
    }

    weak_ref<DestructorLocksWeak>* weak{nullptr};
    bool* lock_succeeded{nullptr};
};

TEST_CASE("weak_ref rejects Python promotion during destruction")
{
    init_fake_python();

    ref<DestructorLocksWeak> strong = make_ref<DestructorLocksWeak>();
    weak_ref<DestructorLocksWeak> weak = strong;
    bool lock_succeeded = true;
    strong->weak = &weak;
    strong->lock_succeeded = &lock_succeeded;
    FakePythonObject python_object{1, strong.get()};

    {
        std::lock_guard lock(fake_python_gil());
        strong->set_self_py(reinterpret_cast<PyObject*>(&python_object));
    }
    strong = nullptr;
    REQUIRE_EQ(python_object.ref_count.load(), 1);

    fake_python_dec_ref(reinterpret_cast<PyObject*>(&python_object));

    CHECK_FALSE(lock_succeeded);
    CHECK_EQ(python_object.invalid_inc_refs.load(), 0);
    CHECK_EQ(python_object.ref_count.load(), 0);
    CHECK(weak.expired());
    CHECK_FALSE(weak.lock());
}

TEST_CASE("weak_ref concurrent Python ownership transfer")
{
    init_fake_python();
    REQUIRE_EQ(DummyObject::get_count(), 0);

    for (uint32_t iteration = 0; iteration < 500; ++iteration) {
        ref<DummyObject> strong = make_ref<DummyObject>();
        weak_ref<DummyObject> weak = strong;
        FakePythonObject python_object{1, strong.get()};
        std::atomic<bool> ready{false};
        std::atomic<bool> go{false};

        std::thread transition(
            [&]
            {
                ready.store(true, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                    std::this_thread::yield();

                std::lock_guard lock(fake_python_gil());
                strong->set_self_py(reinterpret_cast<PyObject*>(&python_object));
                strong = nullptr;
            }
        );

        while (!ready.load(std::memory_order_acquire))
            std::this_thread::yield();
        go.store(true, std::memory_order_release);

        ref<DummyObject> locked = weak.lock();
        transition.join();
        locked = nullptr;

        REQUIRE_EQ(python_object.ref_count.load(), 1);
        fake_python_dec_ref(reinterpret_cast<PyObject*>(&python_object));
        REQUIRE_EQ(python_object.invalid_inc_refs.load(), 0);
        REQUIRE_EQ(DummyObject::get_count(), 0);
        REQUIRE(weak.expired());
    }
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
