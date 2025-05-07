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

#include <string>

namespace vsag {

/**
  * @brief Get the version based on git revision
  * 
  * @return the version text
  */
extern std::string
version();

/**
  * @brief Init the vsag library
  * 
  * @return true always
  */
extern bool
init();

}  // namespace vsag

#include "allocator.h"
#include "binaryset.h"
#include "bitset.h"
#include "constants.h"
#include "dataset.h"
#include "engine.h"
#include "errors.h"
#include "expected.hpp"
#include "factory.h"
#include "index.h"
#include "logger.h"
#include "options.h"
#include "readerset.h"
#include "utils.h"
#include "pnm_engine_def.h"
#include "pnmesdk_client_c.h"


struct AliFlashClient {
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
  }

  void open() {
      database_id_ = pnmesdk_db_open(context_, database_id_, database_name_);
      if (database_id_ == 0) {
          std::cout << "open or create new database failed\n";
          exit(1);
      }
  }

  void upload(float *base_data, size_t vecsize) {
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
          std::cout << "Deal with tail" << std::endl;
          ret = pnmesdk_db_storage(context_, (char*)(base_data + i * vecdim_), (vecsize_ - i) * vecdim_ * FP32);
      }

      if (ret) {
          std::cout << "pnmesdk_db_storage failed, [data upload failed] ret = "
                    << ret << std::endl;
          exit(1);
      }

      std::cout << "Upload succeed" << std::endl;
  }

  int ret = 0;
  database_context* context_ = nullptr;
  pnmesdk_conf *config_ = nullptr;
  uint64_t database_id_ = 0;
  char* database_name_ = "ant-vsag-diskann";
  size_t vecsize_ = 0;
  size_t vecdim_ = 0;

  constexpr static uint64_t BLOCK_SIZE = 1000000ul * 1024 * 4;
   // 1 * 1024 * 1024 * 1024; // bytes
};