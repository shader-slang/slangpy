// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "tev.h"

#include "sgl/core/config.h"
#include "sgl/core/bitmap.h"
#include "sgl/core/format.h"
#include "sgl/core/thread.h"

#include "sgl/device/resource.h"

#include <tevclient.h>

#include <atomic>
#include <semaphore>

namespace sgl::tev {

class ClientPool {
public:
    ClientPool() { tevclient::initialize(); }

    ~ClientPool()
    {
        m_clients.clear();
        tevclient::shutdown();
    }

    std::unique_ptr<tevclient::Client> acquire_client(std::string host, uint16_t port)
    {
        std::unique_ptr<tevclient::Client> client;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = std::find_if(
                m_clients.begin(),
                m_clients.end(),
                [&](const auto& c)
                {
                    return c->getHostname() == host && c->getPort() == port;
                }
            );
            if (it != m_clients.end()) {
                client = std::move(*it);
                m_clients.erase(it);
            }
        }

        if (!client)
            client = std::make_unique<tevclient::Client>(host.c_str(), port);

        return client;
    }

    void release_client(std::unique_ptr<tevclient::Client> client)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_clients.push_back(std::move(client));
    }

    static ClientPool& get()
    {
        static ClientPool instance;
        return instance;
    }

private:
    std::mutex m_mutex;
    std::vector<std::unique_ptr<tevclient::Client>> m_clients;
};

bool show(const Bitmap* bitmap, std::string name, std::string host, uint16_t port, uint32_t max_retries)
{
    SGL_CHECK_NOT_NULL(bitmap);

    ref<Bitmap> converted;
    if (bitmap->component_type() != Bitmap::ComponentType::float32) {
        log_warn_once(
            "tev only supports 32-bit floating point images. Converting {} data to float.",
            bitmap->component_type()
        );
        converted = bitmap->convert(bitmap->pixel_format(), Bitmap::ComponentType::float32, false);
        bitmap = converted;
    }

    static std::atomic<uint32_t> image_counter{0};
    if (name.empty())
        name = fmt::format("image_{}", image_counter++);

    const auto& pixel_struct = *bitmap->pixel_struct();
    std::vector<const char*> channel_names(pixel_struct.field_count());
    std::vector<uint64_t> channel_offsets(pixel_struct.field_count());
    std::vector<uint64_t> channel_strides(pixel_struct.field_count());
    for (size_t i = 0; i < channel_names.size(); ++i) {
        channel_names[i] = pixel_struct[i].name.c_str();
        channel_offsets[i] = pixel_struct[i].offset / sizeof(float);
        channel_strides[i] = pixel_struct.size() / sizeof(float);
    }

    bool connected = false;

    for (uint32_t attempt = 1; attempt <= max_retries; ++attempt) {
        std::unique_ptr<tevclient::Client> client = ClientPool::get().acquire_client(host, port);

        if (!client->isConnected()) {
            if (client->connect() != tevclient::Error::Ok) {
                log_warn(
                    "Failed to connect to tev (attempt {}/{}): {}",
                    attempt,
                    max_retries,
                    client->lastErrorString()
                );
                continue;
            }
        }

        connected = true;

        if (client->createImage(
                name.c_str(),
                bitmap->width(),
                bitmap->height(),
                bitmap->channel_count(),
                channel_names.data()
            )
            != tevclient::Error::Ok) {
            log_warn(
                "Failed to create image in tev (attempt {}/{}): {}",
                attempt,
                max_retries,
                client->lastErrorString()
            );
            continue;
        }

        if (client->updateImage(
                name.c_str(),
                0,
                0,
                bitmap->width(),
                bitmap->height(),
                bitmap->channel_count(),
                channel_names.data(),
                channel_offsets.data(),
                channel_strides.data(),
                (const float*)bitmap->data(),
                bitmap->buffer_size() / 4
            )
            != tevclient::Error::Ok) {
            log_warn(
                "Failed to update image in tev (attempt {}/{}): {}",
                attempt,
                max_retries,
                client->lastErrorString()
            );
            continue;
        }

        ClientPool::get().release_client(std::move(client));
    }

    if (!connected) {
        log_warn(
            "Failed to connect to tev after {} attempts. Make sure the tev image viewer (https://github.com/Tom94/tev) "
            "is running.",
            max_retries
        );
    }

    return true;
}

bool show(const Texture* texture, std::string name, std::string host, uint16_t port, uint32_t max_retries)
{
    SGL_CHECK_NOT_NULL(texture);

    ref<Bitmap> bitmap = texture->to_bitmap();
    return show(bitmap, name, host, port, max_retries);
}

void show_async(const Bitmap* bitmap, std::string name, std::string host, uint16_t port, uint32_t max_retries)
{
    SGL_CHECK_NOT_NULL(bitmap);

    bitmap->inc_ref();

    thread::global_task_group().do_async(
        [=]()
        {
            static std::counting_semaphore semaphore{8};
            semaphore.acquire();
            show(bitmap, name, host, port, max_retries);
            semaphore.release();
            bitmap->dec_ref();
        }
    );
}

void show_async(const Texture* texture, std::string name, std::string host, uint16_t port, uint32_t max_retries)
{
    SGL_CHECK_NOT_NULL(texture);

    ref<Bitmap> bitmap = texture->to_bitmap();
    return show_async(bitmap, name, host, port, max_retries);
}

} // namespace sgl::tev
