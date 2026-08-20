// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "object.h"

#include <mutex>
#include <thread>

#if SGL_ENABLE_OBJECT_TRACKING
#include "sgl/core/error.h"
#include "sgl/core/logger.h"
#include <set>
#endif

namespace sgl {

static void (*object_inc_ref_py)(PyObject*) noexcept = nullptr;
static void (*object_dec_ref_py)(PyObject*) noexcept = nullptr;
static Py_ssize_t_ (*object_ref_cnt_py)(PyObject*) noexcept = nullptr;
static void (*object_run_with_gil_py)(void (*)(void*) noexcept, void*) noexcept = nullptr;

namespace detail {

    enum class WeakStateMode : uint8_t {
        native,
        python,
        expired,
    };

    class WeakState {
    public:
        explicit WeakState(Object* object)
            : m_object(object)
        {
        }

        void inc_ref() noexcept { m_ref_count.fetch_add(1, std::memory_order_relaxed); }

        void dec_ref() noexcept
        {
            if (m_ref_count.fetch_sub(1, std::memory_order_release) == 1) {
                std::atomic_thread_fence(std::memory_order_acquire);
                delete this;
            }
        }

        bool lock(uint64_t ref_id) noexcept
        {
            while (true) {
                switch (m_mode.load(std::memory_order_acquire)) {
                case WeakStateMode::native: {
                    std::unique_lock lock(m_mutex);
                    if (m_mode.load(std::memory_order_relaxed) != WeakStateMode::native)
                        continue;

                    Object* object = m_object;
                    uintptr_t state = object->m_state.load(std::memory_order_acquire);
                    if (state == 0) {
                        lock.unlock();
                        std::this_thread::yield();
                        continue;
                    }
                    if ((state & 1) == 0) {
                        m_mode.store(WeakStateMode::python, std::memory_order_release);
                        continue;
                    }
                    if (state == 1)
                        return false;

                    uintptr_t expected = state;
                    if (!object->m_state.compare_exchange_strong(
                            expected,
                            state + 2,
                            std::memory_order_relaxed,
                            std::memory_order_relaxed
                        ))
                        continue;

#if SGL_ENABLE_REF_TRACKING
                    object->track_ref(ref_id);
#else
                    SGL_UNUSED(ref_id);
#endif
                    return true;
                }
                case WeakStateMode::python: {
                    PythonLockContext context{this, ref_id};
                    if (!object_run_with_gil_py) {
                        fprintf(stderr, "WeakState::lock(): Python reference counting handlers are not installed!");
                        abort();
                    }
                    object_run_with_gil_py(lock_python, &context);
                    if (context.retry)
                        continue;
                    return context.result;
                }
                case WeakStateMode::expired:
                    return false;
                }
            }
        }

        bool try_release_native(Object* object, bool dealloc) noexcept
        {
            std::lock_guard lock(m_mutex);
            if (m_mode.load(std::memory_order_relaxed) != WeakStateMode::native || m_object != object)
                return false;

            uintptr_t expected = 3;
            if (!object->m_state
                     .compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_relaxed))
                return false;

            if (dealloc) {
                m_object = nullptr;
                m_mode.store(WeakStateMode::expired, std::memory_order_release);
            }
            return true;
        }

        void set_python(Object* object) noexcept
        {
            std::lock_guard lock(m_mutex);
            if (m_object == object && m_mode.load(std::memory_order_relaxed) == WeakStateMode::native)
                m_mode.store(WeakStateMode::python, std::memory_order_release);
        }

        void expire(Object* object) noexcept
        {
            std::lock_guard lock(m_mutex);
            if (m_object == object) {
                m_object = nullptr;
                m_mode.store(WeakStateMode::expired, std::memory_order_release);
            }
        }

        bool expired() const noexcept { return m_mode.load(std::memory_order_acquire) == WeakStateMode::expired; }

    private:
        struct PythonLockContext {
            WeakState* state;
            uint64_t ref_id;
            bool result{false};
            bool retry{false};
        };

        static void acquire_strong_ref(Object* object, uint64_t ref_id) noexcept
        {
#if SGL_ENABLE_REF_TRACKING
            object->inc_ref_tracked(ref_id);
#else
            SGL_UNUSED(ref_id);
            object->inc_ref();
#endif
        }

        static void lock_python(void* data) noexcept
        {
            auto& context = *static_cast<PythonLockContext*>(data);
            WeakState* state = context.state;
            std::lock_guard lock(state->m_mutex);

            if (state->m_mode.load(std::memory_order_relaxed) == WeakStateMode::expired)
                return;
            if (state->m_mode.load(std::memory_order_relaxed) != WeakStateMode::python) {
                context.retry = true;
                return;
            }

            Object* object = state->m_object;
            uintptr_t object_state = object->m_state.load(std::memory_order_acquire);
            if (object_state == 0) {
                context.retry = true;
                return;
            }
            if (object_state & 1) {
                state->m_mode.store(WeakStateMode::native, std::memory_order_release);
                context.retry = true;
                return;
            }

            if (!object_ref_cnt_py) {
                fprintf(stderr, "WeakState::lock_python(): Python reference counting handlers are not installed!");
                abort();
            }
            if (object_ref_cnt_py((PyObject*)object_state) <= 0) {
                state->m_object = nullptr;
                state->m_mode.store(WeakStateMode::expired, std::memory_order_release);
                return;
            }

            acquire_strong_ref(object, context.ref_id);
            context.result = true;
        }

        std::atomic<uint32_t> m_ref_count{1};
        mutable std::mutex m_mutex;
        std::atomic<WeakStateMode> m_mode{WeakStateMode::native};
        Object* m_object;
    };

    void weak_state_inc_ref(WeakState* state) noexcept
    {
        state->inc_ref();
    }

    void weak_state_dec_ref(WeakState* state) noexcept
    {
        state->dec_ref();
    }

    bool weak_state_lock(WeakState* state, uint64_t ref_id) noexcept
    {
        return state->lock(ref_id);
    }

    bool weak_state_expired(const WeakState* state) noexcept
    {
        return state->expired();
    }

} // namespace detail

#if SGL_ENABLE_OBJECT_TRACKING
static std::mutex s_tracked_objects_mutex;
static std::set<const Object*> s_tracked_objects;
#endif

#if SGL_ENABLE_OBJECT_TRACKING
Object::Object()
{
    std::lock_guard<std::mutex> lock(s_tracked_objects_mutex);
    s_tracked_objects.insert(this);
}
#endif

Object::~Object()
{
    detail::WeakState* weak_state = m_weak_state.exchange(nullptr, std::memory_order_acq_rel);
    if (weak_state) {
        weak_state->expire(this);
        weak_state->dec_ref();
    }

#if SGL_ENABLE_OBJECT_TRACKING
    std::lock_guard<std::mutex> lock(s_tracked_objects_mutex);
    s_tracked_objects.erase(this);
#endif
}

detail::WeakState* Object::acquire_weak_state() const
{
    detail::WeakState* state = m_weak_state.load(std::memory_order_acquire);
    if (!state) {
        auto* candidate = new detail::WeakState(const_cast<Object*>(this));
        if (!m_weak_state
                 .compare_exchange_strong(state, candidate, std::memory_order_release, std::memory_order_acquire)) {
            delete candidate;
        } else {
            state = candidate;
        }
    }

    uintptr_t object_state = m_state.load(std::memory_order_acquire);
    if (object_state != 0 && (object_state & 1) == 0)
        state->set_python(const_cast<Object*>(this));

    state->inc_ref();
    return state;
}


void Object::inc_ref() const noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);

    while (true) {
        if (value == 0) {
            std::this_thread::yield();
            value = m_state.load(std::memory_order_acquire);
            continue;
        }
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
    uintptr_t value = m_state.load(std::memory_order_acquire);

    while (true) {
        if (value == 0) {
            std::this_thread::yield();
            value = m_state.load(std::memory_order_acquire);
            continue;
        }
        if (value & 1) {
            if (value == 1) {
                fprintf(stderr, "Object::dec_ref(%p): reference count underflow!", this);
                abort();
            } else if (value == 3) {
                detail::WeakState* weak_state = m_weak_state.load(std::memory_order_acquire);
                if (weak_state) {
                    if (!weak_state->try_release_native(const_cast<Object*>(this), dealloc)) {
                        value = m_state.load(std::memory_order_relaxed);
                        continue;
                    }
                } else {
                    uintptr_t expected = 3;
                    if (!m_state.compare_exchange_strong(
                            expected,
                            1,
                            std::memory_order_acq_rel,
                            std::memory_order_relaxed
                        )) {
                        value = expected;
                        continue;
                    }
                }

                if (dealloc)
                    delete this;
            } else {
                if (!m_state
                         .compare_exchange_weak(value, value - 2, std::memory_order_acq_rel, std::memory_order_acquire))
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
    while (value == 0) {
        std::this_thread::yield();
        value = m_state.load(std::memory_order_acquire);
    }
    if (value & 1)
        return value >> 1;
    else
        return 0;
}

void Object::set_self_py(PyObject* o) noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);
    while (true) {
        if (value == 0) {
            fprintf(stderr, "Object::set_self_py(%p): Python ownership transfer is already in progress!", this);
            abort();
        }
        if ((value & 1) == 0) {
            fprintf(stderr, "Object::set_self_py(%p): a Python object was already present!", this);
            abort();
        }
        if (m_state.compare_exchange_weak(value, 0, std::memory_order_acq_rel, std::memory_order_relaxed))
            break;
    }

    uintptr_t ref_count = value >> 1;
    for (uintptr_t i = 0; i < ref_count; ++i)
        object_inc_ref_py(o);

    m_state.store((uintptr_t)o, std::memory_order_release);

    detail::WeakState* weak_state = m_weak_state.load(std::memory_order_acquire);
    if (weak_state)
        weak_state->set_python(this);
}

std::string Object::to_string() const
{
    return fmt::format("{}({})", class_name(), fmt::ptr(this));
}

PyObject* Object::self_py() const noexcept
{
    uintptr_t value = m_state.load(std::memory_order_relaxed);
    while (value == 0) {
        std::this_thread::yield();
        value = m_state.load(std::memory_order_acquire);
    }
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

void Object::track_ref(uint64_t ref_id) const
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
}

void Object::inc_ref_tracked(uint64_t ref_id) const
{
    track_ref(ref_id);

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
    Py_ssize_t_ (*object_ref_cnt_py_)(PyObject*) noexcept,
    void (*object_run_with_gil_py_)(void (*callback)(void*) noexcept, void* context) noexcept
)
{
    object_inc_ref_py = object_inc_ref_py_;
    object_dec_ref_py = object_dec_ref_py_;
    object_ref_cnt_py = object_ref_cnt_py_;
    object_run_with_gil_py = object_run_with_gil_py_;
}

} // namespace sgl
