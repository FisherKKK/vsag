
#pragma once

#include <cstdint>
#ifndef _WINDOWS

#include <malloc.h>
#include <cstdio>
#include <mutex>
#include <thread>
#include "tsl/robin_map.h"
#include "utils.h"
#include <functional>
#include "vsag/readerset.h"
typedef std::function<void(vsag::IOErrorCode code, const std::string& message)> CallBack;


typedef std::vector<std::tuple<uint64_t, float*, void*>> batch_cal_request;
typedef std::vector<std::tuple<uint64_t, uint64_t, void*>> batch_request;

typedef std::function<void(void *query, uint64_t *ids, float *dists, uint32_t size)> reader_multi_cal_function;
typedef std::function<void(batch_request, bool, CallBack)> reader_function;
typedef std::function<void(batch_cal_request, bool, CallBack)> reader_cal_function;

struct AlignedRead
{
    uint64_t offset; // where to read from
    uint64_t len;    // how much to read
    void *buf;       // where to read into

#if USE_ALIFLASH == 1
    uint64_t neighbor_id; // neighbor id
    void *query;
    float *dist;

    AlignedRead() : offset(0), len(0), buf(nullptr), query(nullptr) {}

    AlignedRead(uint64_t offset, uint64_t len, void *buf, uint64_t neighbor_id = 0, void *query = nullptr) :
        offset(offset), len(len), buf(buf), neighbor_id(neighbor_id), query(query) {}

#else
    AlignedRead() : offset(0), len(0), buf(nullptr)
    {
    }
    AlignedRead(uint64_t offset, uint64_t len, void *buf) : offset(offset), len(len), buf(buf)
    {
    }

#endif

};

class LocalFileReader
{
private:
    reader_function func_;
    reader_cal_function func_cal_;
    reader_multi_cal_function func_multi_cal_;
public:
    LocalFileReader(reader_function func): func_(func) {}
    LocalFileReader(reader_function func, reader_cal_function func_cal): func_(func), func_cal_(func_cal) {}
    LocalFileReader(reader_function func, reader_cal_function func_cal, reader_multi_cal_function func_multi_cal):
        func_(func), func_cal_(func_cal), func_multi_cal_(func_multi_cal) {}
    ~LocalFileReader() = default;

    // de-register thread-id for a context
    void deregister_thread() {}
    void deregister_all_threads() {}

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname) {}
    void close() {}

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<AlignedRead> &read_reqs, bool async = false, CallBack callBack = nullptr);

    void read_and_cal(std::vector<AlignedRead> &read_reqs, bool async = false, CallBack callBack = nullptr);

    void read_and_multi_cal(void *query, uint64_t *ids, float *dists, uint32_t size);
};



#endif
