
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

#include <memory>
#include <random>
#include <iostream>
#include <data_cell/flatten_datacell.h>

#include "algorithm/inner_index_interface.h"
#include "data_cell/graph_interface.h"
#include "impl/basic_searcher.h"
#include "index_common_param.h"
#include "io/async_io.h"
#include "pag_zparameters.h"
#include "parameter.h"
#include "quantization/fp32_quantizer.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {

class PAGraph : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

    using InnerIdBucket = Vector<std::unique_ptr<Vector<InnerIdType>>>;
    using InnerIdBucketPtr = std::shared_ptr<InnerIdBucket>;

    PAGraph(const PAGraphParameterPtr& pag_param, const IndexCommonParam& common_param);

    PAGraph(const ParamPtr& param, const IndexCommonParam& common_param)
        : PAGraph(std::dynamic_pointer_cast<PAGraphParameter>(param), common_param) {
    }

    ~PAGraph() override {
        std::cout << "#query number: " << query_count_
                  << ", #io count: " << io_total_count_
                  << ", #io size: " << io_total_size_ / (1024 * 1024)
                  << "MB" << ", #bucket vec cal number: " << cmp_count_ << std::endl;
    };

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    [[nodiscard]] std::string
    GetName() const override;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size) const override;
    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetNumElements() const override;

    void
    InitFeatures() override {
        return;
    }

    int64_t
    GetMemoryUsage() const override {
        return 0;
    }

private:
    Vector<InnerIdType>
    sample_graph_ids(int64_t num_element, int64_t num_sample);

    void
    get_radius();

    void
    aggregate_pag(const DatasetPtr& base);

    void
    select_partition_by_heuristic(MaxHeap& candidates);

    float
    match_score(float d, float r, size_t cur_size, size_t capacity);

    InnerIdType
    calculate_new_centroid(const Vector<InnerIdType>& member, const DatasetPtr& base);

    void
    resize(uint64_t new_size);

    void
    clear_statistic() {
        io_total_count_ = 0;
        io_total_size_ = 0;
        query_count_ = 0;
    }

    void
    parallelize_task(const std::function<void(int64_t, int64_t)>& task, int64_t total_num) {
        Vector<std::future<void>> futures(allocator_);
        for (int64_t i = 0; i < total_num; i += thread_block_size_) {
            int64_t end = std::min(i + thread_block_size_, total_num);
            futures.push_back(thread_pool_->GeneralEnqueue(task, i, end));
        }
        for (auto& future : futures) {
            future.get();
        }
    }

public:
    mutable uint64_t io_total_count_ = 0;
    mutable uint64_t io_total_size_ = 0;  // in bytes
    mutable uint64_t query_count_ = 0;
    mutable uint64_t cmp_count_ = 0;  // dist cmp times

private:
    IndexCommonParam common_param_;
    // PAGraphParameter pag_param_;
    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    InnerIdType max_capacity_{100};
    int64_t num_elements_{0};
    uint64_t num_iter_{1};
    uint64_t replicas_{4};
    uint64_t capacity_{0};
    float sample_rate_{.1f};
    float start_decay_rate_{.8f};
    uint64_t num_sample_{0};
    uint64_t ef_{200};
    std::string bucket_file_{"/tmp/buckets.bin"};

    uint64_t line_size_{0};
    uint64_t code_size_{0};

    float fine_radius_rate_{.5f};
    float coarse_radius_rate_{.5f};
    float upper_bound_radius_{.0f};

    std::default_random_engine generator_{2021};

    Vector<InnerIdType> graph_ids_;
    FlattenInterfacePtr graph_flatten_codes_{nullptr};
    GraphInterfacePtr graph_{nullptr};
    // LabelTablePtr graph_label_table_{nullptr};
    InnerIdType entry_point_{0};
    Vector<float> radii_;

    FlattenDataCellParamPtr graph_flatten_codes_param_;
    GraphInterfaceParamPtr graph_param_;
    ODescentParameterPtr odescent_param_;

    InnerIdBucketPtr buckets_{nullptr};

    std::unique_ptr<VisitedListPool> pool_{nullptr};

    std::shared_ptr<SafeThreadPool> thread_pool_{nullptr};

    std::unique_ptr<BasicSearcher> searcher_{nullptr};

    std::unique_ptr<BasicIO<AsyncIO>> io_;

    std::unique_ptr<Quantizer<FP32Quantizer<>>> quantizer_;


    FlattenDataCellParamPtr low_precision_graph_flatten_codes_param_;
    FlattenInterfacePtr low_precision_graph_flatten_codes_{nullptr};



    bool use_quantization_{false};
    const uint64_t resize_increase_count_bit_{10};
    const int64_t thread_block_size_{800};
    const float recal_threshold_{0.1f};
    const float alpha_{1.2f};

    // Vector<std::mutex> buckets_mutex_;
    // std::shared_mutex bucket_mutex_;
    // std::shared_mutex graph_mutex_;

    // FlattenInterfacePtr flatten_interface_ptr_{nullptr};
    // BucketInterfacePtr bucket_interface_ptr_{nullptr};
    // Vector<InnerIdType> inner_ids_;
    // LabelTable labels_;

    // UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> bucket_;
    // std::unique_ptr<BasicIO<AsyncIO>> io;
};

// class PGraphIndex : public Index {
//     PGraphIndex(const PGraphIndexParameter& param, const IndexCommonParam& common_param);
//
//     ~PGraphIndex() override;
//
//     tl::expected<std::vector<int64_t>, Error>
//     Build(const DatasetPtr& data) override {
//         SAFE_CALL(return this->pgraph_->Build(data));
//     }
//
//     tl::expected<std::vector<int64_t>, Error>
//     Add(const DatasetPtr& data) override {
//         SAFE_CALL(return this->pgraph_->Add(data));
//     }
//
//     tl::expected<DatasetPtr, Error>
//     KnnSearch(const DatasetPtr& query,
//               int64_t k,
//               const std::string& parameters,
//               BitsetPtr invalid = nullptr) const override {
//         std::function<bool(int64_t)> func = [&invalid](int64_t id) -> bool {
//             int64_t bit_index = id & ROW_ID_MASK;
//             return invalid->Test(bit_index);
//         };
//         if (invalid == nullptr) {
//             func = nullptr;
//         }
//         SAFE_CALL(return this->pgraph_->KnnSearch(query, k, parameters, func));
//     }
//
//     tl::expected<DatasetPtr, Error>
//     KnnSearch(const DatasetPtr& query,
//               int64_t k,
//               const std::string& parameters,
//               const std::function<bool(int64_t)>& filter) const override {
//         SAFE_CALL(return this->pgraph_->KnnSearch(query, k, parameters, filter));
//     }
//
// private:
//     std::unique_ptr<PGraph> pgraph_{nullptr};
//
//     std::shared_ptr<Allocator> allocator_{nullptr};
// };

}  // namespace vsag