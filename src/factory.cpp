
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vsag/factory.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <string>

#include "safe_thread_pool.h"
#include "vsag/engine.h"
#include "vsag/options.h"
#include "io/direct_io_object.h"

namespace vsag {

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& origin_name,
                     const std::string& parameters,
                     Allocator* allocator) {
    std::shared_ptr<Resource> resource{nullptr};
    if (allocator == nullptr) {
        resource = std::make_shared<Resource>(Engine::CreateDefaultAllocator(), nullptr);
    } else {
        resource = std::make_shared<Resource>(allocator, nullptr);
    }
    Engine e(resource.get());
    return e.CreateIndex(origin_name, parameters);
}

class LocalFileReader : public Reader {
public:
    explicit LocalFileReader(const std::string& filename,
                             int64_t base_offset = 0,
                             int64_t size = 0,
                             std::shared_ptr<SafeThreadPool> pool = nullptr)
        : filename_(filename),
          file_(std::ifstream(filename, std::ios::binary)),
          base_offset_(base_offset),
          size_(size),
          pool_(std::move(pool)) {
    }

    ~LocalFileReader() override {
        file_.close();
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(static_cast<int64_t>(base_offset_ + offset), std::ios::beg);
        file_.read((char*)dest, static_cast<int64_t>(len));
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        if (not pool_) {
            pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        }
        pool_->GeneralEnqueue([this,  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
                               offset,
                               len,
                               dest,
                               callback]() {
            this->Read(offset, len, dest);
            callback(IOErrorCode::IO_SUCCESS, "success");
        });
    }

    uint64_t
    Size() const override {
        return size_;
    }

private:
    const std::string filename_;
    std::ifstream file_;
    int64_t base_offset_;
    uint64_t size_;
    std::mutex mutex_;
    std::shared_ptr<SafeThreadPool> pool_;
};


class LocalFileAsyncReader : public Reader {
public:
    explicit LocalFileAsyncReader(const std::string& filename,
                                  int64_t base_offset = 0,
                                  int64_t size = 0,
                                  std::shared_ptr<SafeThreadPool> pool = nullptr)
        : base_offset_(base_offset), size_(size), pool_(pool), filename_(filename) {
        rfd_ = open(filename.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
    }

    ~LocalFileAsyncReader() {
        close(rfd_);
        // std::cout << "#total_io_count: " << total_io_count_
        //           << ", #total_io_size: " << total_io_size_ / (1024 * 1024) << "MB" << std::endl;
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        offset += base_offset_;
        DirectIOObject obj(len, offset);
        auto ret = pread64(this->rfd_, obj.align_data, obj.size, obj.offset);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("pread64 error {}", ret));
        }
        memcpy(dest, obj.data, len);
        auto ptr = const_cast<uint8_t*>(obj.data);
        constexpr auto ALIGN_BIT = DirectIOObject::ALIGN_BIT;
        free(reinterpret_cast<void*>((reinterpret_cast<uint64_t>(ptr) >> ALIGN_BIT) << ALIGN_BIT));
    }

    // only async read call this, so we can summary
    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        total_io_count_ += 1;
        total_io_size_ += len;
        if (not pool_) {
            pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        }
        pool_->GeneralEnqueue([this,  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
                               offset,
                               len,
                               dest,
                               callback]() {
            this->Read(offset, len, dest);
            callback(IOErrorCode::IO_SUCCESS, "success");
        });
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return size_;
    }

private:
    int64_t base_offset_;
    int64_t size_;
    std::shared_ptr<SafeThreadPool> pool_;
    int rfd_{-1};
    int64_t total_io_count_{0};
    int64_t total_io_size_{0};
    std::string filename_;
};

std::shared_ptr<Reader>
Factory::CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size) {
    return std::make_shared<LocalFileAsyncReader>(filename, base_offset, size);
}

}  // namespace vsag
