// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/block_allocator.h"

#include <thread>
#include <vector>
#include <atomic>
#include <set>

using namespace sgl;

// Test struct for block allocator
struct TestObject {
    int value;
    double data;
    std::thread::id thread;
    char padding[128]; // Make it bigger to test alignment

    TestObject()
        : value(0)
        , data(0.0)
        , thread()
    {
    }

    TestObject(int v, double d, std::thread::id t = {})
        : value(v)
        , data(d)
        , thread(t)
    {
    }
};

TEST_SUITE_BEGIN("block_allocator");

TEST_CASE("basic_allocation")
{
    BlockAllocator<TestObject> allocator(4); // Small page size for testing

    TestObject* obj = allocator.allocate();
    REQUIRE(obj != nullptr);

    // Check alignment
    REQUIRE(reinterpret_cast<uintptr_t>(obj) % alignof(TestObject) == 0);

    // Use placement new to construct
    new (obj) TestObject(42, 3.14);
    CHECK_EQ(obj->value, 42);
    CHECK_EQ(obj->data, 3.14);

    // Destruct and deallocate
    obj->~TestObject();
    allocator.free(obj);
}

TEST_CASE("multiple_allocations")
{
    BlockAllocator<TestObject> allocator(4);
    std::vector<TestObject*> objects;

    // Allocate more than one page worth
    for (int i = 0; i < 10; ++i) {
        TestObject* obj = allocator.allocate();
        REQUIRE(obj != nullptr);
        new (obj) TestObject(i, i * 1.5);
        objects.push_back(obj);
    }

    // Verify values
    for (int i = 0; i < 10; ++i) {
        CHECK_EQ(objects[i]->value, i);
        CHECK_EQ(objects[i]->data, i * 1.5);
    }

    // Deallocate all
    for (auto obj : objects) {
        obj->~TestObject();
        allocator.free(obj);
    }
}

TEST_CASE("reuse_after_free")
{
    BlockAllocator<TestObject> allocator(4);

    // Allocate and free
    TestObject* obj1 = allocator.allocate();
    REQUIRE(obj1 != nullptr);
    new (obj1) TestObject(1, 1.0);
    obj1->~TestObject();
    allocator.free(obj1);

    // Allocate again - should reuse the same block
    TestObject* obj2 = allocator.allocate();
    REQUIRE(obj2 != nullptr);
    REQUIRE(obj2 == obj1); // Same memory location

    new (obj2) TestObject(2, 2.0);
    CHECK_EQ(obj2->value, 2);
    CHECK_EQ(obj2->data, 2.0);

    obj2->~TestObject();
    allocator.free(obj2);
}

TEST_CASE("free_nullptr")
{
    BlockAllocator<TestObject> allocator(4);
    // Should not crash
    allocator.free(nullptr);
}

TEST_CASE("owns")
{
    BlockAllocator<TestObject> allocator(4);

    TestObject* obj = allocator.allocate();
    REQUIRE(obj != nullptr);

    CHECK(allocator.owns(obj));
    CHECK_FALSE(allocator.owns(nullptr));

    // Stack object should not be owned
    TestObject stack_obj;
    CHECK_FALSE(allocator.owns(&stack_obj));

    // Heap object from regular new should not be owned
    TestObject* heap_obj = new TestObject();
    CHECK_FALSE(allocator.owns(heap_obj));
    delete heap_obj;

    allocator.free(obj);
}

TEST_CASE("page_allocation")
{
    BlockAllocator<TestObject> allocator(4);
    std::vector<TestObject*> objects;

    // Check initial state
    CHECK_EQ(allocator.num_pages(), 0);

    // First allocation should create a page
    objects.push_back(allocator.allocate());
    CHECK_EQ(allocator.num_pages(), 1);

    // Fill the first page
    for (int i = 1; i < 4; ++i) {
        objects.push_back(allocator.allocate());
    }
    CHECK_EQ(allocator.num_pages(), 1);

    // Next allocation should create a new page
    objects.push_back(allocator.allocate());
    CHECK_EQ(allocator.num_pages(), 2);

    // Cleanup
    for (auto obj : objects) {
        allocator.free(obj);
    }
}

TEST_CASE("reset")
{
    BlockAllocator<TestObject> allocator(4);
    std::vector<TestObject*> objects;
    std::set<TestObject*> all_ptrs;

    // Allocate some objects
    for (int i = 0; i < 8; ++i) {
        TestObject* obj = allocator.allocate();
        objects.push_back(obj);
        all_ptrs.insert(obj);
    }

    // Reset the allocator
    allocator.reset();

    // Allocate again - should reuse all blocks
    for (int i = 0; i < 8; ++i) {
        TestObject* obj = allocator.allocate();
        CHECK(all_ptrs.count(obj) == 1);
    }
}

TEST_CASE("multithreaded_allocation")
{
    BlockAllocator<TestObject> allocator(16);
    const int num_threads = 4;
    const int allocations_per_thread = 100;

    std::atomic<int> total_allocations{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(
            [&allocator, &total_allocations]()
            {
                std::vector<TestObject*> local_objects;
                std::thread::id tid = std::this_thread::get_id();

                for (int i = 0; i < allocations_per_thread; ++i) {
                    TestObject* obj = allocator.allocate();
                    if (obj) {
                        new (obj) TestObject(i, i * 2.0, tid);
                        local_objects.push_back(obj);
                        total_allocations++;
                    }
                }

                // Verify and deallocate
                for (size_t i = 0; i < local_objects.size(); ++i) {
                    TestObject* obj = local_objects[i];
                    CHECK_EQ(obj->value, static_cast<int>(i));
                    CHECK_EQ(obj->data, i * 2.0);
                    CHECK_EQ(obj->thread, tid);
                    obj->~TestObject();
                    allocator.free(obj);
                }
            }
        );
    }

    for (auto& t : threads) {
        t.join();
    }

    CHECK_EQ(total_allocations.load(), num_threads * allocations_per_thread);
}

// Test for the macros
class TestMacroClass {
    SGL_DECLARE_BLOCK_ALLOCATED(TestMacroClass)

public:
    int value;
    double data;

    TestMacroClass(int v = 0, double d = 0.0)
        : value(v)
        , data(d)
    {
    }

    static BlockAllocator<TestMacroClass>& get_allocator() { return s_allocator; }
};

SGL_IMPLEMENT_BLOCK_ALLOCATED(TestMacroClass, 16)

TEST_CASE("macro_basic")
{
    // Regular new/delete should use the block allocator
    TestMacroClass* obj = new TestMacroClass(42, 3.14);
    REQUIRE(obj != nullptr);
    CHECK_EQ(obj->value, 42);
    CHECK_EQ(obj->data, 3.14);

    // Verify it's owned by the allocator
    CHECK(TestMacroClass::get_allocator().owns(obj));

    delete obj;
}

TEST_CASE("macro_multiple")
{
    std::vector<TestMacroClass*> objects;

    // Allocate multiple objects
    for (int i = 0; i < 32; ++i) {
        TestMacroClass* obj = new TestMacroClass(i, i * 1.5);
        REQUIRE(obj != nullptr);
        objects.push_back(obj);
    }

    // Verify values
    for (int i = 0; i < 32; ++i) {
        CHECK_EQ(objects[i]->value, i);
        CHECK_EQ(objects[i]->data, i * 1.5);
    }

    // Delete all
    for (auto obj : objects) {
        delete obj;
    }
}

TEST_SUITE_END();
