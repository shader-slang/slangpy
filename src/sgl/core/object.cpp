// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "object.h"

#if SGL_ENABLE_OBJECT_TRACKING
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include <set>
#include <mutex>
#endif

namespace sgl {

static void (*object_inc_ref_py)(PyObject*) noexcept = nullptr;
static void (*object_dec_ref_py)(PyObject*) noexcept = nullptr;
static Py_ssize_t_ (*object_ref_cnt_py)(PyObject*) noexcept = nullptr;

#if SGL_ENABLE_OBJECT_TRACKING
static std::mutex s_tracked_objects_mutex;
static std::set<const Object*> s_tracked_objects;
#endif

Object::Object()
{
#if SGL_ENABLE_OBJECT_TRACKING
    std::lock_guard<std::mutex> lock(s_tracked_objects_mutex);
    s_tracked_objects.insert(this);
#endif
}

Object::~Object()
{
#if SGL_ENABLE_OBJECT_TRACKING
    {
        std::lock_guard<std::mutex> lock(s_tracked_objects_mutex);
        s_tracked_objects.erase(this);
    }
#endif
    WeakAuxiliary* aux = m_weak_aux.load(std::memory_order_acquire);
    if (aux) {
        {
            std::lock_guard<std::mutex> lock(aux->mutex);
            aux->object = nullptr;
        }
        if (aux->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete aux;
        }
    }
}

WeakAuxiliary* Object::get_weak_aux() const
{
    WeakAuxiliary* aux = m_weak_aux.load(std::memory_order_acquire);
    if (!aux) {
        WeakAuxiliary* new_aux = new WeakAuxiliary();
        new_aux->object = const_cast<Object*>(this);
        WeakAuxiliary* expected = nullptr;
        if (m_weak_aux
                .compare_exchange_strong(expected, new_aux, std::memory_order_release, std::memory_order_acquire)) {
            aux = new_aux;
        } else {
            delete new_aux;
            aux = expected;
        }
    }
    return aux;
}


void Object::inc_ref() const noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);

    while (true) {
        if (value & 1) {
            if (!m_state.compare_exchange_weak(value, value + 2, std::memory_order_relaxed, std::memory_order_relaxed))
                continue;
        } else {
            object_inc_ref_py((PyObject*)value);
        }

        break;
    }
}

void Object::dec_ref(bool dealloc) const noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);

    while (true) {
        if (value & 1) {
            if (value == 1) {
                fprintf(stderr, "Object::dec_ref(%p): reference count underflow!", this);
                abort();
            } else if (value == 3) {
                // The reference count is 1 (value 3 means ref count 1 because of the bit shift).
                // We are about to delete the object.
                // However, there might be a weak reference trying to lock the object concurrently.
                WeakAuxiliary* aux = m_weak_aux.load(std::memory_order_acquire);
                if (aux) {
                    // Acquire the mutex to synchronize with weak_ref::lock().
                    std::lock_guard<std::mutex> lock(aux->mutex);

                    // Check if the reference count has changed while we were waiting for the mutex.
                    // If weak_ref::lock() succeeded, it would have incremented the reference count.
                    if (m_state.load(std::memory_order_relaxed) != 3) {
                        value = m_state.load(std::memory_order_relaxed);
                        continue;
                    }

                    // If we are here, it means no weak reference was able to lock the object,
                    // and we hold the lock, so no new weak reference can lock it now.
                    // We can safely mark the object as destroyed in the auxiliary structure.
                    aux->object = nullptr;
                }
                // If aux is null, it is impossible for another thread to concurrently create it.
                // Creating the auxiliary object (via get_weak_aux()) requires holding a strong reference
                // to the object. If we are here (ref count == 1, which is ours), no other thread
                // holds a strong reference, so no other thread can call get_weak_aux().

                if (dealloc) {
                    delete this;
                } else {
                    m_state.store(1, std::memory_order_relaxed);
                }
            } else {
                if (!m_state
                         .compare_exchange_weak(value, value - 2, std::memory_order_relaxed, std::memory_order_relaxed))
                    continue;
            }
        } else {
            object_dec_ref_py((PyObject*)value);
        }
        break;
    }
}

uint64_t Object::ref_count() const
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);
    if (value & 1)
        return value >> 1;
    else
        return 0;
}

void Object::set_self_py(PyObject* o) noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);
    if (value & 1) {
        value >>= 1;
        for (uintptr_t i = 0; i < value; ++i)
            object_inc_ref_py(o);

        m_state.store((uintptr_t)o);
    } else {
        fprintf(stderr, "Object::set_self_py(%p): a Python object was already present!", this);
        abort();
    }
}

std::string Object::to_string() const
{
    return fmt::format("{}({})", class_name(), fmt::ptr(this));
}

PyObject* Object::self_py() const noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);
    if (value & 1)
        return nullptr;
    else
        return (PyObject*)value;
}

#if SGL_ENABLE_OBJECT_TRACKING

void Object::report_live_objects()
{
    std::lock_guard<std::mutex> lock(s_tracked_objects_mutex);
    if (!s_tracked_objects.empty()) {
        fmt::println("Found {} live objects!", s_tracked_objects.size());
        for (const Object* object : s_tracked_objects) {
            uint64_t ref_count = object->ref_count();
            PyObject* self_py = object->self_py();
            if (self_py)
                ref_count = object_ref_cnt_py(self_py);
            fmt::println(
                "Live object: {} self_py={} ref_count={} class_name=\"{}\"",
                fmt::ptr(object),
                self_py ? fmt::ptr(self_py) : "null",
                ref_count,
                object->class_name()
            );
            object->report_refs();
        }
    }
}

void Object::report_refs() const
{
#if SGL_ENABLE_REF_TRACKING
    std::lock_guard<std::mutex> lock(m_ref_trackers_mutex);
    for (const auto& it : m_ref_trackers) {
        fmt::println(
            "ref={} count={}\n{}\n",
            it.first,
            it.second.count,
            platform::format_stacktrace(it.second.stack_trace)
        );
    }
#endif
}

#endif // SGL_ENABLE_OBJECT_TRACKING

#if SGL_ENABLE_REF_TRACKING

void Object::inc_ref_tracked(uint64_t ref_id) const
{
    if (m_enable_ref_tracking) {
        std::lock_guard<std::mutex> lock(m_ref_trackers_mutex);
        auto it = m_ref_trackers.find(ref_id);
        if (it != m_ref_trackers.end()) {
            it->second.count++;
        } else {
            m_ref_trackers.emplace(ref_id, RefTracker{1, platform::backtrace()});
        }
    }

    inc_ref();
}

void Object::dec_ref_tracked(uint64_t ref_id, bool dealloc) const noexcept
{
    if (m_enable_ref_tracking) {
        std::lock_guard<std::mutex> lock(m_ref_trackers_mutex);
        auto it = m_ref_trackers.find(ref_id);
        SGL_ASSERT(it != m_ref_trackers.end());
        if (--it->second.count == 0) {
            m_ref_trackers.erase(it);
        }
    }

    dec_ref(dealloc);
}

void Object::set_enable_ref_tracking(bool enable)
{
    m_enable_ref_tracking = enable;
}

#endif // SGL_ENABLE_REF_TRACKING

void object_init_py(
    void (*object_inc_ref_py_)(PyObject*) noexcept,
    void (*object_dec_ref_py_)(PyObject*) noexcept,
    Py_ssize_t_ (*object_ref_cnt_py_)(PyObject*) noexcept
)
{
    object_inc_ref_py = object_inc_ref_py_;
    object_dec_ref_py = object_dec_ref_py_;
    object_ref_cnt_py = object_ref_cnt_py_;
}

} // namespace sgl
