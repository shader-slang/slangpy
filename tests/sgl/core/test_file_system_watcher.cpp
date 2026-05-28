// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"
#include "sgl/core/file_system_watcher.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace sgl;

TEST_SUITE_BEGIN("file_system_watcher");

TEST_CASE("FileSystemWatcher")
{
    SUBCASE("add_and_remove_watch")
    {
        auto path = testing::get_case_temp_directory() / "add_and_remove_watch";
        std::filesystem::create_directories(path);
        ref<FileSystemWatcher> watcher = make_ref<FileSystemWatcher>();
        watcher->add_watch({.directory = path});
        watcher->remove_watch(path);
    }

    SUBCASE("add_and_remove_watch_by_id")
    {
        auto path = testing::get_case_temp_directory() / "add_and_remove_watch";
        std::filesystem::create_directories(path);
        ref<FileSystemWatcher> watcher = make_ref<FileSystemWatcher>();
        auto id = watcher->add_watch({.directory = path});
        watcher->remove_watch(id);
    }

    SUBCASE("fail_missing_directory")
    {
        ref<FileSystemWatcher> watcher = make_ref<FileSystemWatcher>();
        CHECK_THROWS(watcher->add_watch({.directory = "./i_am_missing"}));
    }

    SUBCASE("fail_duplicate_watch")
    {
        ref<FileSystemWatcher> watcher = make_ref<FileSystemWatcher>();
        auto path = testing::get_case_temp_directory();
        watcher->add_watch({.directory = path});
        CHECK_THROWS(watcher->add_watch({.directory = path}));
    }

    SUBCASE("file_watches")
    {
        using namespace std::chrono;
        using namespace std::chrono_literals;
        using namespace std::filesystem;

        // Events array + watcher event handler to add to them.
        std::vector<FileSystemWatchEvent> events;
        auto handler = [&events](std::span<FileSystemWatchEvent> event_received)
        {
            for (auto x : event_received) {
                if (x.absolute_path.has_extension())
                    events.push_back(x);
            }
        };

        // Setup temp directory.
        auto path = testing::get_case_temp_directory() / "file_watches";
        path = absolute(path);
        create_directories(path);

        // Also create a subdirectory below it for later on in the test.
        auto subpath = path / "subdir0" / "subdir1";
        create_directories(subpath);

        // Record start time + configure file watcher.
        auto now = system_clock::now();
        ref<FileSystemWatcher> watcher = make_ref<FileSystemWatcher>();
        watcher->set_on_change(handler);
        watcher->add_watch({.directory = path});

        // Set a short delay on watcher outputs so the tests don't take ages to run.
        watcher->set_delay(10);

        auto check = [&](std::string filename, FileSystemWatcherChange event_type)
        {
            bool done = false;
            // Exact wait time is unpredictable, so poll watcher for up to 1s in short intervals
            // Expecting less than 100ms really, but operating systems are unpredictable!
            for (int it = 0; it < 100 && !done; it++) {
                std::this_thread::sleep_for(10ms);
                watcher->update();

                // File system watcher can return spurious modification events, so
                // we're just requiring the desired event to appear, but not treating
                // excess events as an error.
                for (auto ev : events) {
                    if (ev.change == event_type && ev.path == filename) {
                        // Verify abs path.
                        CHECK_EQ(ev.absolute_path, path / filename);

                        // Rough check for sane timing.
                        auto duration = duration_cast<seconds>(ev.time - now).count();
                        CHECK_GE(duration, 0);
                        CHECK_LT(duration, 10);

                        // Mark done and exit.
                        done = true;
                        break;
                    }
                }
            }
            CHECK(done);
            events.clear();
        };

        auto check_none = [&]()
        {
            // Verifying no events arrive just requires minimum wait. Chosen
            // 1s to avoid making test system take too long.
            for (int it = 0; it < 100; it++) {
                std::this_thread::sleep_for(10ms);
                watcher->update();
                if (events.size() > 0)
                    break;
            }

            // Check nothing received
            CHECK_EQ(events.size(), 0);
        };

        // Create a file.
        std::ofstream outfile(path / "testfile.txt");
        check("testfile.txt", FileSystemWatcherChange::added);

        // Write some data and close file (only way to guaruntee it flushes).
        outfile << "hello";
        outfile.close();
        check("testfile.txt", FileSystemWatcherChange::modified);

        // Delete the file and repeat tests again checking for 'deleted' event
        remove(path / "testfile.txt");
        check("testfile.txt", FileSystemWatcherChange::removed);

        // Create a file in the subdirectory which should not be detected,
        // as the watch is not recursive
        std::ofstream outsubfile(subpath / "subfile.txt");
        check_none();

        // Clean up
        watcher->remove_watch(path);
    }
}

TEST_SUITE_END();
