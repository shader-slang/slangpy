// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/plugin.h"

#include <functional>
#include <memory>
#include <string>

using namespace sgl;

TEST_SUITE_BEGIN("plugin");

/// Base class for plugin type A.
class PluginBaseA {
public:
    struct PluginInfo {
        std::string desc;
    };

    using PluginCreate = std::function<std::shared_ptr<PluginBaseA>(const std::string&)>;

    SGL_PLUGIN_BASE_CLASS(PluginBaseA);

    virtual ~PluginBaseA() { }
    virtual const std::string& get_text() const = 0;
};

/// First plugin of type A.
class PluginA1 : public PluginBaseA {
public:
    SGL_PLUGIN_CLASS(PluginA1, "PluginA1", "This is PluginA1");

    static std::shared_ptr<PluginA1> create(const std::string& text) { return std::make_shared<PluginA1>(text); }

    PluginA1(const std::string& text)
        : mText(text)
    {
    }

    const std::string& get_text() const { return mText; }

private:
    std::string mText;
};

/// Second plugin of type A.
class PluginA2 : public PluginBaseA {
public:
    SGL_PLUGIN_CLASS(PluginA2, "PluginA2", "This is PluginA2");

    static std::shared_ptr<PluginA2> create(const std::string& text) { return std::make_shared<PluginA2>(text); }

    PluginA2(const std::string& text)
        : mText(text + "\n" + text)
    {
    }

    const std::string& get_text() const { return mText; }

private:
    std::string mText;
};

/// Base class for plugin type B.
/// Here we use a different create function that returns pointers.
class PluginBaseB {
public:
    struct PluginInfo {
        std::string name;
        std::vector<int> sequence;
    };

    using PluginCreate = PluginBaseB* (*)();

    SGL_PLUGIN_BASE_CLASS(PluginBaseB);

    virtual ~PluginBaseB() { }
};

/// First plugin of type B.
class PluginB1 : public PluginBaseB {
public:
    SGL_PLUGIN_CLASS(PluginB1, "PluginB1", PluginInfo({"This is PluginB1", {1, 2, 4, 8}}));

    static PluginBaseB* create() { return new PluginB1(); }
};

/// Second plugin of type B.
class PluginB2 : public PluginBaseB {
public:
    SGL_PLUGIN_CLASS(PluginB2, "PluginB2", PluginInfo({"This is PluginB2", {2, 4, 8, 16}}));

    static PluginBaseB* create() { return new PluginB2(); }
};

TEST_CASE("PluginManager")
{
    PluginManager pm;

    // No classes of first type are registered yet.
    CHECK_FALSE(pm.has_class<PluginBaseA>("PluginA1"));
    CHECK_FALSE(pm.has_class<PluginBaseA>("PluginA2"));

    // Register plugins of first type.
    // Note: This is typically done within the plugin library in an exported registerPlugin function.
    {
        PluginRegistry registry(pm, 0);
        registry.register_class<PluginBaseA, PluginA1>();
        registry.register_class<PluginBaseA, PluginA2>();
    }

    // Check for registered classes of first type.
    CHECK(pm.has_class<PluginBaseA>("PluginA1"));
    CHECK(pm.has_class<PluginBaseA>("PluginA2"));

    // Check infos of first type.
    {
        bool has_plugin_a1{false};
        bool has_plugin_a2{false};
        size_t count = 0;
        for (const auto& [name, info] : pm.get_infos<PluginBaseA>()) {
            if (name == "PluginA1")
                has_plugin_a1 = info.desc == "This is PluginA1";
            else if (name == "PluginA2")
                has_plugin_a2 = info.desc == "This is PluginA2";
            count++;
        }
        CHECK(has_plugin_a1);
        CHECK(has_plugin_a2);
        CHECK_EQ(count, 2);
    }

    // Create plugins of first type.
    {
        auto plugin_a1 = pm.create_class<PluginBaseA>("PluginA1", "Hello world");
        auto plugin_a2 = pm.create_class<PluginBaseA>("PluginA2", "Hello world again");
        auto plugin_a3 = pm.create_class<PluginBaseA>("PluginA3", ""); // does not exist

        CHECK_NE(plugin_a1, nullptr);
        CHECK_NE(plugin_a2, nullptr);
        CHECK_EQ(plugin_a3, nullptr);

        CHECK_EQ(plugin_a1->get_plugin_type(), "PluginA1");
        CHECK_EQ(plugin_a1->get_plugin_info().desc, "This is PluginA1");
        CHECK_EQ(plugin_a1->get_text(), "Hello world");

        CHECK_EQ(plugin_a2->get_plugin_type(), "PluginA2");
        CHECK_EQ(plugin_a2->get_plugin_info().desc, "This is PluginA2");
        CHECK_EQ(plugin_a2->get_text(), "Hello world again\nHello world again");
    }

    // No classes of second type are registered yet.
    CHECK_FALSE(pm.has_class<PluginBaseB>("PluginB1"));
    CHECK_FALSE(pm.has_class<PluginBaseB>("PluginB2"));

    // Register plugins of second type.
    // Note: This is typically done within the plugin library in an exported registerPlugin function.
    {
        PluginRegistry registry(pm, 0);
        registry.register_class<PluginBaseB, PluginB1>();
        registry.register_class<PluginBaseB, PluginB2>();
    }

    // Check for registered classes of second type.
    CHECK(pm.has_class<PluginBaseB>("PluginB1"));
    CHECK(pm.has_class<PluginBaseB>("PluginB2"));

    // Check infos of second type.
    {
        bool has_plugin_b1{false};
        bool has_plugin_b2{false};
        size_t count = 0;
        for (const auto& [name, info] : pm.get_infos<PluginBaseB>()) {
            if (name == "PluginB1")
                has_plugin_b1 = info.sequence == std::vector<int>{1, 2, 4, 8};
            else if (name == "PluginB2")
                has_plugin_b2 = info.sequence == std::vector<int>{2, 4, 8, 16};
            count++;
        }
        CHECK(has_plugin_b1);
        CHECK(has_plugin_b2);
        CHECK_EQ(count, 2);
    }

    // Create plugins of second type.
    {
        auto plugin_b1 = pm.create_class<PluginBaseB>("PluginB1");
        auto plugin_b2 = pm.create_class<PluginBaseB>("PluginB2");
        auto plugin_b3 = pm.create_class<PluginBaseB>("PluginB3"); // does not exist

        CHECK_NE(plugin_b1, nullptr);
        CHECK_NE(plugin_b2, nullptr);
        CHECK_EQ(plugin_b3, nullptr);

        CHECK_EQ(plugin_b1->get_plugin_type(), "PluginB1");
        CHECK_EQ(plugin_b1->get_plugin_info().sequence, std::vector<int>({1, 2, 4, 8}));

        CHECK_EQ(plugin_b2->get_plugin_type(), "PluginB2");
        CHECK_EQ(plugin_b2->get_plugin_info().sequence, std::vector<int>({2, 4, 8, 16}));

        delete plugin_b1;
        delete plugin_b2;
        delete plugin_b3;
    }
}

TEST_SUITE_END();
