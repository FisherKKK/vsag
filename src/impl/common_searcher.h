
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

#include "algorithm/hnswlib/algorithm_interface.h"

namespace vsag {
enum InnerSearchMode { KNN_SEARCH = 1, RANGE_SEARCH = 2 };

class InnerSearchParam {
public:
    int64_t topk{0};
    float radius{0.0f};
    InnerIdType ep{0};
    uint64_t ef{10};
    FilterPtr is_inner_id_allowed{nullptr};
    float skip_ratio{0.8F};
    InnerSearchMode search_mode{KNN_SEARCH};
    int range_search_limit_size{-1};
};

constexpr float THRESHOLD_ERROR = 2e-6;
}  // namespace vsag