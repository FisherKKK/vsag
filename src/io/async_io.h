
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

#pragma once

#include <iostream>
#include <utility>

#include "async_io_parameter.h"
#include "basic_io.h"
#include "direct_io_object.h"
#include "index/index_common_param.h"
#include "io_context.h"

namespace vsag {
class AsyncIO : public BasicIO<AsyncIO> {
public:
    AsyncIO(std::string filename, Allocator* allocator)
        : BasicIO<AsyncIO>(allocator), filepath_(std::move(filename)) {
        this->rfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644);
        this->wfd_ = open(filepath_.c_str(), O_CREAT | O_RDWR, 0644);
        std::cout << "Using async io" << std::endl;
    }

    explicit AsyncIO(const AsyncIOParameterPtr& io_param, const IndexCommonParam& common_param)
        : AsyncIO(io_param->path_, common_param.allocator_.get()){};

    explicit AsyncIO(const IOParamPtr& param, const IndexCommonParam& common_param)
        : AsyncIO(std::dynamic_pointer_cast<AsyncIOParameter>(param), common_param){};

    ~AsyncIO() override {
#ifdef DEBUG_IO
        std::cout << "Prepare time: " << io_info.prepare << ", wait time: " << io_info.wait
                  << ", copy time: " << io_info.copy << ", overall time: " << io_info.overall
                  << ", call number: " << io_info.call_time << std::endl;
#endif
    }

#ifdef DEBUG_IO
    using Clock = std::chrono::high_resolution_clock;
    mutable struct {
        double prepare{0};
        double wait{0};
        double copy{0};
        double overall{0};
        int64_t call_time{0};
        std::mutex mutex;
    } io_info;
#endif

public:
    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
        auto ret = pwrite64(this->wfd_, data, size, offset);
        if (ret != size) {
            throw std::runtime_error(fmt::format("write bytes {} less than {}", ret, size));
        }
        if (size + offset > this->size_) {
            this->size_ = size + offset;
        }
        fsync(wfd_);
    }

    inline bool
    ReadImpl(uint64_t size, uint64_t offset, uint8_t* data) const {
        bool need_release = true;
        auto ptr = DirectReadImpl(size, offset, need_release);
        memcpy(data, ptr, size);
        this->ReleaseImpl(ptr);
        return true;
    }

    [[nodiscard]] inline const uint8_t*
    DirectReadImpl(uint64_t size, uint64_t offset, bool& need_release) const {
        need_release = true;
        if (size == 0) {
            return nullptr;
        }
        DirectIOObject obj(size, offset);
        auto ret = pread64(this->rfd_, obj.align_data, obj.size, obj.offset);
        if (ret < 0) {
            throw std::runtime_error(fmt::format("pread64 error {}", ret));
        }
        return obj.data;
    }

    inline void
    ReleaseImpl(const uint8_t* data) const {
        auto ptr = const_cast<uint8_t*>(data);
        constexpr auto ALIGN_BIT = DirectIOObject::ALIGN_BIT;
        free(reinterpret_cast<void*>((reinterpret_cast<uint64_t>(ptr) >> ALIGN_BIT) << ALIGN_BIT));
    }

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        auto context = io_context_pool->TakeOne();
        uint8_t* cur_data = datas;
        int64_t all_count = count;
        while (all_count > 0) {
#ifdef DEBUG_IO
            auto point1 = Clock::now();
#endif
            count = std::min(IOContext::DEFAULT_REQUEST_COUNT, all_count);
            auto* cb = context->cb_;
            std::vector<DirectIOObject> objs(count);
            for (int64_t i = 0; i < count; ++i) {
                objs[i].Set(sizes[i], offsets[i]);
                auto& obj = objs[i];
                io_prep_pread(cb[i], rfd_, obj.align_data, obj.size, obj.offset);
                cb[i]->data = &(objs[i]);
            }

            int submitted = io_submit(context->ctx_, count, cb);
            if (submitted < 0) {
                io_context_pool->ReturnOne(context);
                for (auto& obj : objs) {
                    obj.Release();
                }
                throw std::runtime_error("io submit failed");
            }

#ifdef DEBUG_IO
            auto point2 = Clock::now();
#endif

            struct timespec timeout = {1, 0};
            auto num_events = io_getevents(context->ctx_, count, count, context->events_, &timeout);
            if (num_events != count) {
                io_context_pool->ReturnOne(context);
                for (auto& obj : objs) {
                    obj.Release();
                }
                throw std::runtime_error("io async read failed");
            }

#ifdef DEBUG_IO
            auto point3 = Clock::now();
#endif

            for (int64_t i = 0; i < count; ++i) {
                memcpy(cur_data, objs[i].data, sizes[i]);
                cur_data += sizes[i];
                this->ReleaseImpl(objs[i].data);
            }

#ifdef DEBUG_IO
            auto point4 = Clock::now();
#endif

#ifdef DEBUG_IO
            {
                std::unique_lock<std::mutex> lk(io_info.mutex);
                auto prepare_time =
                    std::chrono::duration<double, std::milli>(point2 - point1).count();
                auto wait_time = std::chrono::duration<double, std::milli>(point3 - point2).count();
                auto copy_time = std::chrono::duration<double, std::milli>(point4 - point3).count();
                auto overall_time =
                    std::chrono::duration<double, std::milli>(point4 - point1).count();
                io_info.prepare += prepare_time;
                io_info.wait += wait_time;
                io_info.copy += copy_time;
                io_info.overall += overall_time;
                io_info.call_time += 1;
            }
#endif

            sizes += count;
            offsets += count;
            all_count -= count;
        }
        io_context_pool->ReturnOne(context);
        return true;
    }

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cache_line = 64){};

    static inline bool
    InMemoryImpl() {
        return false;
    }

public:
    static std::unique_ptr<IOContextPool> io_context_pool;

private:
    std::string filepath_{};

    int rfd_{-1};

    int wfd_{-1};
};
}  // namespace vsag
