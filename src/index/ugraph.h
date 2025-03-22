
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

#include <random>

#include "algorithm/inner_index_interface.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "impl/basic_searcher.h"
#include "impl/odescent_graph_parameter.h"
#include "quantization/fp32_quantizer.h"
#include "ugraph_zparameters.h"
#include "utils/visited_list.h"

namespace vsag {
class UGraph : public InnerIndexInterface {
public:
    using Edge = std::tuple<float, InnerIdType, InnerIdType>;
    using InnerIdBucket = Vector<std::unique_ptr<Vector<InnerIdType>>>;
    using InnerIdBucketPtr = std::shared_ptr<InnerIdBucket>;

    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    UGraph(const UgraphParameterPtr& param, const IndexCommonParam& common_param);

    UGraph(const ParamPtr& param, const IndexCommonParam& common_param)
        : UGraph(std::dynamic_pointer_cast<UGraphParameter>(param), common_param) {
    }

    ~UGraph() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return "ugraph";
    }

    void
    InitFeatures() override {
    }

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override {
        return {};
    }

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size) const override {
        return nullptr;
    }

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return num_elements_;
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override {
        return 0;
    }

private:
    void
    resize(uint64_t new_size);

public:
    mutable uint64_t io_total_count_ = 0;
    mutable uint64_t io_total_size_ = 0;  // in bytes
    mutable uint64_t query_count_ = 0;
    mutable uint64_t cmp_count_ = 0;  // dist cmp times

private:
private:
    IndexCommonParam common_param_;

    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    InnerIdType max_capacity_{100};
    int64_t num_elements_{0};
    uint64_t capacity_{0};
    std::string bucket_file_{"/tmp/buckets.bin"};

    uint64_t line_size_{0};
    uint64_t code_size_{0};

    std::default_random_engine generator_{2021};

    Vector<InnerIdType> core_ids_;
    FlattenInterfacePtr graph_flatten_codes_{nullptr};
    GraphInterfacePtr graph_{nullptr};

    InnerIdType entry_point_{0};

    FlattenDataCellParamPtr graph_flatten_codes_param_{nullptr};
    GraphInterfaceParamPtr graph_param_{nullptr};
    ODescentParameterPtr odescent_param_{nullptr};

    InnerIdBucketPtr buckets_{nullptr};
    Vector<int64_t> buckets_offset_;

    std::unique_ptr<VisitedListPool> pool_{nullptr};
    std::shared_ptr<SafeThreadPool> thread_pool_{nullptr};
    std::unique_ptr<BasicSearcher> searcher_{nullptr};

    std::unique_ptr<Quantizer<FP32Quantizer<>>> quantizer_;

    FlattenDataCellParamPtr low_precision_graph_flatten_codes_param_{nullptr};
    FlattenInterfacePtr low_precision_graph_flatten_codes_{nullptr};
    bool use_quantization_{false};

    std::shared_ptr<Reader> disk_reader_;
    std::function<void(uint64_t offset, uint64_t size, void* dest, CallBack)> batch_read_;
};

}  // namespace vsag
