// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/core/macros.h"
#include "sgl/core/format.h"

#include <atomic>
#include <type_traits>
#include <utility>
#include <string>
#include <cstdint>

extern "C" {
struct _object;
typedef _object PyObject;
typedef int64_t Py_ssize_t_;
static_assert(sizeof(Py_ssize_t_) == sizeof(size_t));
};

/// Enable/disable object lifetime tracking.
/// When enabled, each object derived from Object will have its
/// lifetime tracked. This is useful for debugging memory leaks.
#ifndef SGL_ENABLE_OBJECT_TRACKING
#define SGL_ENABLE_OBJECT_TRACKING 0
#endif

/// Enable/disable reference tracking.
/// When enabled, all references to an object that has reference tracking
/// enabled using set_enable_ref_tracking() are tracked. Each time the reference
/// count is increased, the current stack trace is stored. This helps identify
/// owners of objects that are not properly releasing their references.
/// Optionally all references can be tracked by setting SGL_TRACK_ALL_REFS.
#ifndef SGL_ENABLE_REF_TRACKING
#define SGL_ENABLE_REF_TRACKING 0
#endif

#if SGL_ENABLE_REF_TRACKING
#if !SGL_ENABLE_OBJECT_TRACKING
#error "SGL_ENABLE_REF_TRACKING requires SGL_ENABLE_OBJECT_TRACKING"
#endif
#include "platform.h"
#include <map>
#include <mutex>
static constexpr bool SGL_TRACK_ALL_REFS{false};
#endif


namespace sgl {

/**
 * \brief Object base class with intrusive reference counting
 *
 * The Object class provides a convenient foundation of a class hierarchy that
 * will ease lifetime and ownership-related issues whenever Python bindings are
 * involved.
 *
 * Internally, its constructor sets the `m_state` field to `1`, which indicates
 * that the instance is owned by C++. Bits 2..63 of this field are used to
 * store the actual reference count value. The `inc_ref()` and `dec_ref()`
 * functions can be used to increment or decrement this reference count. When
 * `dec_ref()` removes the last reference, the instance will be deallocated
 * using a `delete` expression handled using a polymorphic destructor.
 *
 * When a subclass of `Object` is constructed to Python or returned from C++ to
 * Python, nanobind will invoke `Object::set_self_py()`, which hands ownership
 * over to Python/nanobind. Any remaining references will be moved from the
 * `m_state` field to the Python reference count. In this mode, `inc_ref()` and
 * `dec_ref()` wrap Python reference counting primitives (`Py_INCREF()` /
 * `Py_DECREF()`) which must be made available by calling the function
 * `object_init_py` once during module initialization. Note that the `m_state`
 * field is also used to store a pointer to the `PyObject *`. Python instance
 * pointers are always aligned (i.e. bit 1 is zero), which disambiguates
 * between the two possible configurations.
 *
 * Within C++, the RAII helper class `ref` (defined below) can be used to keep
 * instances alive. This removes the need to call the `inc_ref()` / `dec_ref()`
 * functions explicitly.
 *
 * ```
 * {
 *    ref<MyClass> inst = new MyClass();
 *    inst->my_function();
 *    ...
 * } // end of scope, 'inst' automatically deleted if no longer referenced
 * ```
 *
 * A separate optional file ``object_py.h`` provides a nanobind type caster
 * to bind functions taking/returning values of type `ref<T>`.
 */
class SGL_API Object {
public:
#if SGL_ENABLE_OBJECT_TRACKING
    /// Default constructor.
    Object();
    /// Destructor.
    virtual ~Object();
#else
    /// Default constructor.
    Object() = default;
    /// Destructor.
    virtual ~Object() = default;
#endif

    /// Copy constructor.
    /// Note: We don't copy the reference counter, so that the new object
    /// starts with a reference count of 0.
    Object(const Object&) { }

    /// Copy assignment.
    /// Note: We don't copy the reference counter, but leave the reference
    /// counts of the two objects unchanged. This results in the same semantics
    /// that we would get if we used `std::shared_ptr` where the reference
    /// counter is stored in a separate place from the object.
    Object& operator=(const Object&) { return *this; }

    /// Make the object non-movable.
    Object(Object&&) = delete;
    Object& operator=(Object&&) = delete;

    /// Return the name of the class.
    /// Note: This reports the actual class name if \c SGL_OBJECT() is used.
    virtual const char* class_name() const { return "Object"; }

    /// Increase the object's reference count.
    void inc_ref() const noexcept;

    /// Decrease the object's reference count and potentially deallocate it.
    void dec_ref(bool dealloc = true) const noexcept;

    /// Return current reference count.
    uint64_t ref_count() const;

    /// Return the Python object associated with this instance (or NULL)
    PyObject* self_py() const noexcept;

    /// Set the Python object associated with this instance
    void set_self_py(PyObject* self) noexcept;

    /// Return a string representation of this object.
    /// This is used for debugging purposes.
    virtual std::string to_string() const;

#if SGL_ENABLE_OBJECT_TRACKING
    /// Reports current set of live objects.
    static void report_live_objects();

    /// Report references of this object.
    void report_refs() const;
#endif

#if SGL_ENABLE_REF_TRACKING
    void inc_ref_tracked(uint64_t ref_id) const;
    void dec_ref_tracked(uint64_t ref_id, bool dealloc = true) const noexcept;

    /// Enable/disable reference tracking of this object.
    void set_enable_ref_tracking(bool enable);
#endif

private:
    mutable std::atomic<uintptr_t> m_state{1};

#if SGL_ENABLE_REF_TRACKING
    struct RefTracker {
        uint32_t count{1};
        platform::StackTrace stack_trace;
        // RefTracker(StackTrace stack_trace_)
        //     : count(1)
        //     , stack_trace(std::move(stack_trace))
        // {
        // }
    };
    mutable std::map<uint64_t, RefTracker> m_ref_trackers;
    mutable std::mutex m_ref_trackers_mutex;
    bool m_enable_ref_tracking = SGL_TRACK_ALL_REFS;
#endif
};

/// Macro to declare the object class name.
#define SGL_OBJECT(class_)                                                                                             \
public:                                                                                                                \
    const char* class_name() const override                                                                            \
    {                                                                                                                  \
        return #class_;                                                                                                \
    }

/**
 * \brief Install Python reference counting handlers
 *
 * The `Object` class is designed so that the dependency on Python is
 * *optional*: the code compiles in ordinary C++ projects, in which case the
 * Python reference counting functionality will simply not be used.
 *
 * Python binding code must invoke `object_init_py` and provide functions that
 * can be used to increase/decrease/get the Python reference count of an instance
 * (i.e., `Py_INCREF` / `Py_DECREF` / `Py_REFCNT`).
 */
SGL_API void object_init_py(
    void (*object_inc_ref_py)(PyObject*) noexcept,
    void (*object_dec_ref_py)(PyObject*) noexcept,
    Py_ssize_t_ (*object_ref_cnt_py)(PyObject*) noexcept
);


#if SGL_ENABLE_REF_TRACKING
namespace detail {
    inline uint64_t next_ref_id()
    {
        static std::atomic<uint64_t> s_next_id = 0;
        return s_next_id.fetch_add(1);
    }
} // namespace detail
#endif


/**
 * \brief Reference counting helper.
 *
 * The \a ref template is a simple wrapper to store a pointer to an object. It
 * takes care of increasing and decreasing the object's reference count as
 * needed. When the last reference goes out of scope, the associated object
 * will be deallocated.
 *
 * This class follows similar semantics to the ``std::shared_ptr`` class from
 * the STL. In particular, we avoid implicit conversion to and from raw
 * pointers.
 */
template<typename T>
class ref {
public:
    /// Default constructor (nullptr).
    ref() { }

    /// Construct a reference from a nullptr.
    ref(std::nullptr_t) { }

    /// Construct a reference from a convertible pointer.
    template<typename T2 = T>
    explicit ref(T2* ptr)
        : m_ptr(ptr)
    {
        static_assert(
            std::is_base_of_v<Object, T2>,
            "Cannot create reference to object not inheriting from Object class."
        );
        static_assert(
            std::is_convertible_v<T2*, T*>,
            "Cannot create reference to object from unconvertible pointer type."
        );
        if (m_ptr)
            inc_ref(reinterpret_cast<const Object*>(m_ptr));
    }

    /// Copy constructor.
    ref(const ref& r)
        : m_ptr(r.m_ptr)
    {
        if (m_ptr)
            inc_ref(reinterpret_cast<const Object*>(m_ptr));
    }

    /// Construct a reference from a convertible reference.
    template<typename T2 = T>
    ref(const ref<T2>& r)
        : m_ptr(r.m_ptr)
    {
        static_assert(
            std::is_base_of_v<Object, T>,
            "Cannot create reference to object not inheriting from Object class."
        );
        static_assert(
            std::is_convertible_v<T2*, T*>,
            "Cannot create reference to object from unconvertible reference."
        );
        if (m_ptr)
            inc_ref(reinterpret_cast<const Object*>(m_ptr));
    }

    /// Construct a reference from a pointer.
    explicit ref(T* ptr)
        : m_ptr(ptr)
    {
        if (m_ptr)
            inc_ref(reinterpret_cast<const Object*>(m_ptr));
    }

    /// Move constructor.
    ref(ref&& r) noexcept
        : m_ptr(r.m_ptr)
#if SGL_ENABLE_REF_TRACKING
        , m_ref_id(r.m_ref_id)
#endif
    {
        r.m_ptr = nullptr;
#if SGL_ENABLE_REF_TRACKING
        r.m_ref_id = uint64_t(-1);
#endif
    }

    /// Construct a reference by moving from a convertible reference.
    template<typename T2>
    explicit ref(ref<T2>&& r) noexcept
        : m_ptr(r.m_ptr)
#if SGL_ENABLE_REF_TRACKING
        , m_ref_id(r.m_ref_id)
#endif
    {
        static_assert(
            std::is_base_of_v<Object, T>,
            "Cannot create reference to object not inheriting from Object class."
        );
        static_assert(
            std::is_convertible_v<T2*, T*>,
            "Cannot create reference to object from unconvertible reference."
        );
        r.m_ptr = nullptr;
#if SGL_ENABLE_REF_TRACKING
        r.m_ref_id = uint64_t(-1);
#endif
    }

    /// Destructor.
    ~ref()
    {
        if (m_ptr)
            dec_ref(reinterpret_cast<const Object*>(m_ptr));
    }

    /// Assign another reference into the current one.
    ref& operator=(const ref& r) noexcept
    {
        if (r != *this) {
            if (r.m_ptr)
                inc_ref(reinterpret_cast<const Object*>(r.m_ptr));
            T* prev_ptr = m_ptr;
            m_ptr = r.m_ptr;
            if (prev_ptr)
                dec_ref(reinterpret_cast<const Object*>(prev_ptr));
        }
        return *this;
    }

    /// Assign another convertible reference into the current one.
    template<typename T2>
    ref& operator=(const ref<T2>& r) noexcept
    {
        static_assert(
            std::is_convertible_v<T2*, T*>,
            "Cannot assign reference to object from unconvertible reference."
        );
        if (r != *this) {
            if (r.m_ptr)
                inc_ref(reinterpret_cast<const Object*>(r.m_ptr));
            T* prev_ptr = m_ptr;
            m_ptr = r.m_ptr;
            if (prev_ptr)
                dec_ref(reinterpret_cast<const Object*>(prev_ptr));
        }
        return *this;
    }

    /// Move another reference into the current one.
    ref& operator=(ref&& r) noexcept
    {
        if (static_cast<void*>(&r) != this) {
            if (m_ptr)
                dec_ref(reinterpret_cast<const Object*>(m_ptr));
            m_ptr = r.m_ptr;
            r.m_ptr = nullptr;
#if SGL_ENABLE_REF_TRACKING
            m_ref_id = r.m_ref_id;
            r.m_ref_id = uint64_t(-1);
#endif
        }
        return *this;
    }

    /// Move another convertible reference into the current one.
    template<typename T2>
    ref& operator=(ref<T2>&& r) noexcept
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot move reference to object from unconvertible reference.");
        if (static_cast<void*>(&r) != this) {
            if (m_ptr)
                dec_ref(reinterpret_cast<const Object*>(m_ptr));
            m_ptr = r.m_ptr;
            r.m_ptr = nullptr;
#if SGL_ENABLE_REF_TRACKING
            m_ref_id = r.m_ref_id;
            r.m_ref_id = uint64_t(-1);
#endif
        }
        return *this;
    }

    /// Overwrite this reference with a pointer to another object
    template<typename T2 = T>
    void reset(T2* ptr = nullptr) noexcept
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot assign reference to object from unconvertible pointer.");
        if (ptr != m_ptr) {
            if (ptr)
                inc_ref(reinterpret_cast<const Object*>(ptr));
            T* prevPtr = m_ptr;
            m_ptr = ptr;
            if (prevPtr)
                dec_ref(reinterpret_cast<const Object*>(prevPtr));
        }
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator==(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>,
            "Cannot compare references of non-convertible types."
        );
        return m_ptr == r.m_ptr;
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator!=(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>,
            "Cannot compare references of non-convertible types."
        );
        return m_ptr != r.m_ptr;
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator<(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>,
            "Cannot compare references of non-convertible types."
        );
        return m_ptr < r.m_ptr;
    }

    /// Compare this reference to a pointer.
    template<typename T2 = T>
    bool operator==(const T2* ptr) const
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot compare reference to pointer of non-convertible types.");
        return m_ptr == ptr;
    }

    /// Compare this reference to a pointer.
    template<typename T2 = T>
    bool operator!=(const T2* ptr) const
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot compare reference to pointer of non-convertible types.");
        return m_ptr != ptr;
    }

    /// Compare this reference to a null pointer.
    bool operator==(std::nullptr_t) const { return m_ptr == nullptr; }

    /// Compare this reference to a null pointer.
    bool operator!=(std::nullptr_t) const { return m_ptr != nullptr; }

    /// Compare this reference to a null pointer.
    bool operator<(std::nullptr_t) const { return m_ptr < nullptr; }

    /// Access the object referenced by this reference.
    T* operator->() const { return m_ptr; }

    /// Return a C++ reference to the referenced object.
    T& operator*() const { return *m_ptr; }

    /// Return a pointer to the referenced object.
    T* get() const { return m_ptr; }

    /// Return a pointer to the referenced object.
    operator T*() const { return m_ptr; }

    /// Check if the object is defined
    explicit operator bool() const { return m_ptr != nullptr; }

    /// Swap this reference with another reference.
    void swap(ref& r) noexcept
    {
        std::swap(m_ptr, r.m_ptr);
#if SGL_ENABLE_REF_TRACKING
        std::swap(m_ref_id, r.m_ref_id);
#endif
    }

private:
    inline void inc_ref(const Object* object)
    {
#if SGL_ENABLE_REF_TRACKING
        object->inc_ref_tracked(m_ref_id);
#else
        object->inc_ref();
#endif
    }

    inline void dec_ref(const Object* object)
    {
#if SGL_ENABLE_REF_TRACKING
        object->dec_ref_tracked(m_ref_id);
#else
        object->dec_ref(true);
#endif
    }

    T* m_ptr{nullptr};
#if SGL_ENABLE_REF_TRACKING
    uint64_t m_ref_id{detail::next_ref_id()};
#endif

    template<typename T2>
    friend class ref;
};

template<class T, class... Args>
ref<T> make_ref(Args&&... args)
{
    return ref<T>(new T(std::forward<Args>(args)...));
}

template<class T, class U>
ref<T> static_ref_cast(const ref<U>& r) noexcept
{
    return ref<T>(static_cast<T*>(r.get()));
}

template<class T, class U>
ref<T> dynamic_ref_cast(const ref<U>& r) noexcept
{
    return ref<T>(dynamic_cast<T*>(r.get()));
}

template<typename T>
struct is_ref : std::false_type { };

template<typename T>
struct is_ref<ref<T>> : std::true_type { };

template<typename T>
struct is_ref<const ref<T>> : std::true_type { };

template<typename T>
inline constexpr bool is_ref_v = is_ref<T>::value;

static_assert(is_ref_v<Object> == false);
static_assert(is_ref_v<ref<Object>> == true);

template<class T>
struct remove_ref {
    using type = T;
};

template<class T>
struct remove_ref<ref<T>> {
    using type = T;
};

template<class T>
struct remove_ref<const ref<T>> {
    using type = T;
};

static_assert(std::is_same_v<remove_ref<Object>::type, Object> == true);
static_assert(std::is_same_v<remove_ref<ref<Object>>::type, Object> == true);
static_assert(std::is_same_v<remove_ref<const ref<Object>>::type, Object> == true);


/**
 * \brief Breakable reference counting helper for avoding reference cycles.
 *
 * This helper represents a strong reference (ref<T>) that can be broken.
 * This is accomplished by storing both a strong reference and a raw pointer.
 * When the strong reference is broken, we access the referenced object through
 * the raw pointer.
 *
 * This helper can be used in scenarios where some object holds nested objects
 * that themselves hold a reference to the parent object. In such cases, the
 * nested objects should hold a breakable reference to the parent object.
 * When the nested objects are created, we can immediately break the strong
 * reference to the parent object. This allows the parent object to be destroyed
 * when all of the external references to it are released.
 *
 * This helper can be used in place of a \a ref, but it cannot be reassigned.
 */
template<typename T>
class breakable_ref {
public:
    breakable_ref(const ref<T>& r)
        : m_strong_ref(r)
        , m_weak_ref(m_strong_ref.get())
    {
    }
    breakable_ref(ref<T>&& r)
        : m_strong_ref(std::forward<ref<T>>(r))
        , m_weak_ref(m_strong_ref.get())
    {
    }

    breakable_ref() = delete;
    breakable_ref& operator=(const ref<T>&) = delete;
    breakable_ref& operator=(ref<T>&&) = delete;

    T* get() const { return m_weak_ref; }
    T* operator->() const { return get(); }
    T& operator*() const { return *get(); }
    operator ref<T>() const { return ref<T>(get()); }
    operator T*() const { return get(); }
    explicit operator bool() const { return get() != nullptr; }

    void break_strong_reference() { m_strong_ref.reset(); }

private:
    ref<T> m_strong_ref;
    T* m_weak_ref = nullptr;
};

/**
 * \brief Helper class to protect objects from being deleted before the constructor has finished.
 *
 * When objects are constructed, they have a reference count of zero.
 * If during the constructor, a reference to the object is taken and later released,
 * the object will be deleted before the constructor has finished.
 *
 * This helper class will take the initial reference to the object during construction,
 * and avoid any other references to the object from deallocating it before the constructor
 * has finished.
 *
 * \tparam T Object type.
 */
template<typename T>
class ConstructorRefGuard {
public:
    SGL_NON_COPYABLE_AND_MOVABLE(ConstructorRefGuard);

    explicit ConstructorRefGuard(T* obj)
        : m_obj(obj)
    {
        m_obj->inc_ref();
    }

    ~ConstructorRefGuard() { m_obj->dec_ref(false); }

private:
    T* m_obj;
};


} // namespace sgl

template<typename T>
struct fmt::formatter<sgl::ref<T>> : formatter<const void*> {
    template<typename FormatContext>
    auto format(const sgl::ref<T>& ref, FormatContext& ctx) const
    {
        return formatter<const void*>::format(ref.get(), ctx);
    }
};

template<typename T>
struct fmt::formatter<sgl::breakable_ref<T>> : formatter<const void*> {
    template<typename FormatContext>
    auto format(const sgl::breakable_ref<T>& ref, FormatContext& ctx) const
    {
        return formatter<const void*>::format(ref.get(), ctx);
    }
};

namespace std {
template<typename T>
void swap(::sgl::ref<T>& x, ::sgl::ref<T>& y) noexcept
{
    return x.swap(y);
}

template<typename T>
struct less<::sgl::ref<T>> {
    bool operator()(const ::sgl::ref<T>& a, const ::sgl::ref<T>& b) const { return a.get() < b.get(); }
};

template<typename T>
struct hash<::sgl::ref<T>> {
    constexpr size_t operator()(const ::sgl::ref<T>& r) const { return std::hash<T*>()(r.get()); }
};


} // namespace std
