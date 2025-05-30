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

#include "pnm_engine_def.h"
#include "pnmesdk_client_c.h"
#include <iostream>
#include <cstdint>
#include <memory>

// #define MULTICASE
// #define ALIFLASH_DEBUG_CAL

struct AliFlashClient {

  static std::shared_ptr<AliFlashClient> GetInstance(size_t dim) {
    static auto client = std::make_shared<AliFlashClient>(dim);
    return client;
  }

  AliFlashClient(size_t dim) {
      config_ = new pnmesdk_conf();
      context_ = new database_context();

      ret = pnmesdk_init(config_);

      context_->data_type = FP32;
      context_->vecdim = dim;
      vecdim_ = dim;
  }

  ~AliFlashClient() {
    std::cout << "pre uinit and context delete" << std::endl;
    pnmesdk_uninit(config_);
    delete context_;
    delete config_;
    std::cout << "uinit and context delete" << std::endl;

    // print the average cal number per batch
#ifdef ALIFLASH_DEBUG_CAL
    {
      std::cout << "Aliflash called number: " << cal_info.call_num
                << ", distance cal number: " << cal_info.cal_num
                << ", average batch cal number: " << cal_info.cal_num / (cal_info.call_num + 0.00001f)
                << std::endl;
    }
#endif


  }

  void open() {
#ifdef MULTICASE
      static bool init = false;
      if (init) return;
#endif
      database_id_ = pnmesdk_db_open(context_, database_id_, database_name_);
      if (database_id_ == 0) {
          std::cout << "open or create new database failed\n";
          exit(1);
      }
#ifdef MULTICASE
      init = true;
#endif
  }

  void upload(float *base_data, size_t vecsize) {
#ifdef MULTICASE
      static bool init = false;
      if (init) return;
#endif

      vecsize_ = vecsize;

      std::cout << "Upload ready, " << "vecsize: " << vecsize_
                << ", vecdim: " << vecdim_
                << std::endl;

      // block size
      size_t offset = BLOCK_SIZE / (vecdim_ * FP32);
      std::cout << "Offset size: " << std::dec << offset << std::endl;

      // fold upload
      size_t i;
      for (i = 0; i + offset <= vecsize_; i += offset) {
          std::cout << "Offset #" << i << std::endl;
          ret = pnmesdk_db_storage(context_,
                                  (char*)(base_data + i * vecdim_),
                                  offset * vecdim_ * FP32);
      }

      // tail
      if (i < vecsize_) {
          std::cout << "Deal with tail: " << vecsize_ - i << std::endl;
          ret = pnmesdk_db_storage(context_, (char*)(base_data + i * vecdim_), (vecsize_ - i) * vecdim_ * FP32);
      }

      if (ret) {
          std::cout << "pnmesdk_db_storage failed, [data upload failed] ret = "
                    << ret << std::endl;
          exit(1);
      }

      std::cout << "Upload succeed" << std::endl;
#ifdef MULTICASE
      init = true;
#endif
  }

  void cal_single(void *query, uint64_t single_id, float *dist, int hnsw_query_id) {
    calculate_config calc_config;
    calc_config.target_vector = query;
    calc_config.target_vector_size = vecdim_ * FP32;
    calc_config.ids_list = &single_id;
    calc_config.ids_size = 1;
    calc_config.result_list = dist;
    calc_config.hnsw_query_id = hnsw_query_id;
    ret = database_context_cal(context_, &calc_config);
    if (ret) {
        printf("current vector cal task execute failed, ret = %d\n", ret);
        exit(1);
    }

#ifdef ALIFLASH_DEBUG_CAL
    {
      std::unique_lock<std::mutex> lk(cal_info.mutex);
      cal_info.cal_num += 1;
      cal_info.call_num += 1;
    }
#endif

  }

  void cal_multi(void *query, uint64_t* ids, float *dist, uint32_t size, int hnsw_query_id) {
    calculate_config calc_config;
    calc_config.target_vector = query;
    calc_config.target_vector_size = vecdim_ * FP32;
    calc_config.ids_list = ids;
    calc_config.ids_size = size;
    calc_config.result_list = dist;
    calc_config.hnsw_query_id = hnsw_query_id;
    ret = database_context_cal(context_, &calc_config);
    if (ret) {
      printf("current vector cal task execute failed, ret = %d\n", ret);
      exit(1);
    }

#ifdef ALIFLASH_DEBUG_CAL
    // record call number and cal number
    {
      std::unique_lock<std::mutex> lk(cal_info.mutex);
      cal_info.cal_num += size;
      cal_info.call_num += 1;
    }
#endif

  }


  int begin_single() {
    std::unique_lock<std::mutex> lk(mutex_);
    return pnme_get_search_query_id();
  }

  void end_single(int query_id) {
    std::unique_lock<std::mutex> lk(mutex_);
    pnme_hnsw_search_end(query_id);
  }

#ifdef ALIFLASH_DEBUG_CAL
  struct {
    std::mutex mutex;
    size_t cal_num{0};
    size_t call_num{0};
  } cal_info;
#endif


  int ret = 0;
  database_context* context_ = nullptr;
  pnmesdk_conf *config_ = nullptr;
  uint64_t database_id_ = 0;
  char* database_name_ = "ant-vsag-hgraph";
  size_t vecsize_ = 0;
  size_t vecdim_ = 0;
  std::mutex mutex_;

  constexpr static uint64_t BLOCK_SIZE = 100000UL * 1024 * 4;
   // 1 * 1024 * 1024 * 1024; // bytes
};