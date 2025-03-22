
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
#include "ugraph.h"

#include "impl/odescent_graph_builder.h"
#include "utils/util_functions.h"
#include "vsag/factory.h"

// #define OMYDEBUG

namespace vsag {

namespace {
struct UnionSet {
    UnionSet(InnerIdType num_elements, Allocator* allocator)
        : num_elements_(num_elements),
          parents_(num_elements, allocator),
          rank_(num_elements, 0, allocator) {
        for (InnerIdType i = 0; i < num_elements; i++) {
            parents_[i] = i;
        }
    }

    InnerIdType
    Find(InnerIdType x) {
        if (parents_[x] == x)
            return x;
        return parents_[x] = Find(parents_[x]);
    }

    void
    Union(InnerIdType x, InnerIdType y) {
        InnerIdType root_x = Find(x);
        InnerIdType root_y = Find(y);
        if (root_x != root_y) {
            if (rank_[root_x] < rank_[root_y]) {
                parents_[root_x] = root_y;
            } else {
                parents_[root_y] = root_x;
                if (rank_[root_x] == rank_[root_y])
                    rank_[root_x] += 1;
            }
        }
    }

    InnerIdType num_elements_;
    Vector<InnerIdType> parents_;
    Vector<InnerIdType> rank_;
};

struct EdgeCmpFirst {
    constexpr bool
    operator()(std::tuple<float, InnerIdType, InnerIdType>& a,
               std::tuple<float, InnerIdType, InnerIdType>& b) const noexcept {
        return std::get<0>(a) > std::get<0>(b);
    }
};

using MinEdgeHeap = std::priority_queue<std::tuple<float, InnerIdType, InnerIdType>,
                                        Vector<std::tuple<float, InnerIdType, InnerIdType>>,
                                        EdgeCmpFirst>;
}  // namespace

UGraph::UGraph(const UgraphParameterPtr& ugraph_param, const IndexCommonParam& common_param)
    : InnerIndexInterface(ugraph_param, common_param),
      common_param_(common_param),
      dim_(common_param.dim_),
      metric_(common_param.metric_),
      core_ids_(common_param.allocator_.get()),
      graph_flatten_codes_param_(ugraph_param->graph_flatten_codes_param_),
      graph_param_(ugraph_param->graph_param_),
      odescent_param_(ugraph_param->odescent_param_),
      buckets_offset_(common_param.allocator_.get()),
      use_quantization_(ugraph_param->use_quantization_),
      low_precision_graph_flatten_codes_param_(
          ugraph_param->low_precision_graph_flatten_codes_param_),
      thread_pool_(common_param_.thread_pool_),
      bucket_file_(ugraph_param->bucket_file_) {
    code_size_ = dim_ * sizeof(float);
    //TODO: line_size?
    pool_ = std::make_unique<VisitedListPool>(
        1, common_param.allocator_.get(), max_capacity_, common_param_.allocator_.get());
    searcher_ = std::make_unique<BasicSearcher>(common_param_);
    quantizer_ = std::make_unique<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>>(
        common_param.dim_,
        common_param.allocator_.get());  // TODO: quantizer for bucket calculation

    int64_t file_size = 0;
    disk_reader_ = Factory::CreateLocalFileReader(bucket_file_, 0, file_size);
    batch_read_ =
        [&](uint64_t offset, uint64_t size, void* dest, const CallBack& callback) -> void {
        disk_reader_->AsyncRead(offset, size, dest, callback);
    };
}

std::vector<int64_t>
UGraph::Build(const DatasetPtr& base) {
    auto num_elements = base->GetNumElements();
    const auto* base_vecs = base->GetFloat32Vectors();
    const auto* base_ids = base->GetIds();
    num_elements_ += num_elements;

    label_table_->label_table_.resize(num_elements);
    memcpy(label_table_->label_table_.data(), base_ids, sizeof(LabelType) * num_elements);

    pool_ = std::make_unique<VisitedListPool>(1, allocator_, num_elements, allocator_);

    graph_flatten_codes_ =
        FlattenInterface::MakeInstance(graph_flatten_codes_param_, common_param_);
    graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);

    // Insert vector
    graph_flatten_codes_->BatchInsertVector(base_vecs, num_elements);

    ODescent graph_builder(
        odescent_param_, graph_flatten_codes_, allocator_, common_param_.thread_pool_.get());
    graph_builder.Build();
    graph_builder.SaveGraph(graph_);

    auto graph_size = graph_->TotalCount();
    MinEdgeHeap edges(allocator_);

    UnionSet us(graph_size, allocator_);
    auto edges_size = edges.size();

    Vector<float> radius(graph_size, std::numeric_limits<float>::max(), allocator_);

    for (auto i = 0; i < graph_size; i++) {
        Vector<InnerIdType> nns(allocator_);
        graph_->GetNeighbors(i, nns);
        // auto nn_size = nns.size() > 2 ? 2 : nns.size();
        //
        // for (int nni = 0; nni < nn_size; nni++) {
        //     us.Union(i, nns[nni]);
        // }
        Vector<float> distances(allocator_);
        distances.reserve(32);

        for (auto nn : nns) {
            auto d = graph_flatten_codes_->ComputePairVectors(i, nn);
            edges.emplace(d, i, nn);
            distances.emplace_back(d);
        }
        std::sort(distances.begin(), distances.end());
        radius[i] = distances[distances.size() * 0.2];
    }

    // UnionSet us(graph_size, allocator_);
    // auto edges_size = edges.size();

    // Compress half edges for performance test
    float last_dist = 0.f, history = 0.f, delta = 0.1f;
    int64_t count = 0, count_threshold = 128;


    // std::ofstream edges_writer("/tmp/edges.txt", std::ios_base::out);
    while (edges.size() > 0) {
        auto [d, src, dest] = edges.top();
        auto src_root = us.Find(src), dest_root = us.Find(dest);

        // edges_writer << d << " ";
        edges.pop();
        // Select which edge should be aggregate

        auto &src_rd = radius[src], &dest_rd = radius[dest];
        if (count < count_threshold || (d < dest_rd && d < src_rd)) {
            us.Union(src_root, dest_root);
            // src_rd = d;
            // dest_rd = d;
        }
        count += 1;
    }

    // Store the union graph id in union_core
    // [root1] [root2] [root3] ... [rootn]
    Vector<InnerIdType> union_core(allocator_);
    union_core.reserve(graph_size / 5);

    // Set for union_core unique record
    // [root1] -- 1, [root2] -- 2, [root3] -- 3, mapping
    UnorderedMap<InnerIdType, InnerIdType> unique_core(allocator_);

    // Buckets record the core[id1, id2, id3, ..., ]
    buckets_ = std::make_shared<InnerIdBucket>(allocator_);
    buckets_->reserve(graph_size / 5);

    // Sets for edge storage
    // Vector<std::shared_ptr<UnorderedSet<InnerIdType>>> linklist(allocator_);
    // linklist.reserve(graph_size / 5);

    // Keep graph
    for (InnerIdType i = 0; i < graph_size; i++) {
        auto root = us.Find(i);
        Vector<InnerIdType> nns(allocator_);
        graph_->GetNeighbors(i, nns);
        InnerIdType index = 0;

        if (unique_core.count(root)) {
            auto core = unique_core[root];
            index = core;
        } else {
            auto new_core = (InnerIdType)(union_core.size());
            unique_core.emplace(root, new_core);
            union_core.emplace_back(root);

            // Create neighbor set
            // linklist.emplace_back(std::make_shared<UnorderedSet<InnerIdType>>(allocator_));

            // Create bucket
            buckets_->emplace_back(std::make_unique<Vector<InnerIdType>>(allocator_));
            buckets_->back()->reserve(5);

            index = new_core;
        }

        // Merge the edges of graph
        // linklist[index]->insert(nns.begin(), nns.end());
        buckets_->at(index)->emplace_back(i);
    }

    // Construct core --> unique id
    auto compute_centroid = [&](Vector<InnerIdType>& partition) {
        auto partition_size = partition.size();

        Vector<float> sum(dim_, 0.f, allocator_);
        for (auto pid : partition) {
            const float* q = base_vecs + pid * dim_;
            for (int64_t i = 0; i < dim_; i++) {
                sum[i] += q[i];
            }
        }

        for (int64_t i = 0; i < dim_; i++) {
            sum[i] /= partition_size + 0.00001f;
        }

        float min_dist = std::numeric_limits<float>::max();
        InnerIdType min_pid = 0;

        for (auto pid : partition) {
            const float* q = base_vecs + pid * dim_;
            float d = quantizer_->Compute((uint8_t*)sum.data(), (uint8_t*)q);
            if (d < min_dist) {
                min_dist = d;
                min_pid = pid;
            }
        }
        return min_pid;
    };

    auto core_size = (InnerIdType)union_core.size();

    std::cout << "Compressed graph size: " << core_size << std::endl;
    int64_t cur_offset = 0;
    buckets_offset_.resize(core_size);
    for (InnerIdType ci = 0; ci < core_size; ci++) {
        auto& bucket = *(buckets_->at(ci));
        auto new_core = compute_centroid(bucket);
        union_core[ci] = new_core;
        buckets_offset_[ci] = cur_offset;
        assert(buckets_->size() >= 1);
        cur_offset += (static_cast<int64_t>(bucket.size()) - 1) * code_size_;
    }

    core_ids_.swap(union_core);

    // rebuild graph?
    graph_flatten_codes_.reset();
    graph_.reset();

    graph_flatten_codes_ =
        FlattenInterface::MakeInstance(graph_flatten_codes_param_, common_param_);
    graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);
    for (auto core : core_ids_) {
        const auto* core_vec = base_vecs + core * dim_;
        graph_flatten_codes_->InsertVector(core_vec);
    }

    ODescent refine_graph_builder(
        odescent_param_, graph_flatten_codes_, allocator_, common_param_.thread_pool_.get());
    refine_graph_builder.Build();
    refine_graph_builder.SaveGraph(graph_);

    // Save the bucket
    std::cout << "Saving bucket..." << std::endl;
    std::ofstream stream(bucket_file_, std::ios_base::out | std::ios_base::binary);
    // TODO: error check
    Vector<float> dummy_vec(dim_, 0.f, allocator_);
    for (int i = 0; i < core_ids_.size(); i++) {
        auto core_id = core_ids_[i];
        auto& members_ptr = buckets_->at(i);
        auto& members = *members_ptr;

        for (auto m_i = members.begin(); m_i != members.end(); ++m_i) {
            if (*m_i == core_id) {
                members.erase(m_i);
                break;
            }
        }

        // assert(members.size() <= capacity_);
        for (auto member : members) {
            const auto* vec = base_vecs + member * dim_;
            stream.write((char*)vec, sizeof(*vec) * dim_);
        }

        // for (int i = 0; i < capacity_ - members.size(); i++) {
        //     stream.write((char*)(dummy_vec.data()), sizeof(float) * dim_);
        // }
    }

    return {};
}

DatasetPtr
UGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    // TODO: make it type check
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // IO-related stat
    query_count_ += 1;

    auto search_param_json = JsonType::parse(parameters)["ugraph"];
    int nprobe = search_param_json["nprobe"];
    int ef_search = search_param_json["ef_search"];

    MaxHeap result(common_param_.allocator_.get());

    auto vl = pool_->TakeOne();

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_;
    search_param.topk = nprobe;
    search_param.ef = ef_search;

    FlattenInterfacePtr flatten_codes = graph_flatten_codes_;
    if (use_quantization_) {
        flatten_codes = low_precision_graph_flatten_codes_;
    }

    // Vector<AlignedRead> sorted_read_reqs(allocator_);
    auto graph_id_result =
        searcher_->Search(graph_, flatten_codes, vl, query->GetFloat32Vectors(), search_param);

    auto issue_all_size = 0;
    auto result_copy = graph_id_result;

    while (result_copy.size() > 0) {
        auto [_, gid] = result_copy.top();
        result_copy.pop();
        issue_all_size += buckets_->at(gid)->size() * code_size_;
    }

    Vector<uint8_t> issue_data(allocator_);
    issue_data.resize(issue_all_size);

    Vector<InnerIdType> issue_bucket_inner_ids(allocator_);
    issue_bucket_inner_ids.reserve(nprobe);

    uint64_t issue_count = 0, issue_data_size = 0;

    Deque<std::pair<uint8_t*, InnerIdType>> completion_queue(allocator_);

    std::mutex mutex;
    std::condition_variable cv;
    int64_t issue_data_off = 0;

    while (graph_id_result.size() > 0) {
        auto [centroid_dist, centroid_id] = graph_id_result.top();
        auto inner_id = core_ids_[centroid_id];
        graph_id_result.pop();
        if (!use_quantization_) {
            result.emplace(centroid_dist, inner_id);
            vl->Set(inner_id);
        }

        auto& bucket = buckets_->at(centroid_id);
        auto issue_size = bucket->size() * code_size_;
        // auto offset = line_size_ * centroid_id;
        auto offset = buckets_offset_[centroid_id];
        if (issue_size == 0)
            continue;

        auto dest = issue_data.data() + issue_data_off;
        issue_data_off += issue_size;

        auto callback = [&, dest, centroid_id](vsag::IOErrorCode code, const std::string& message) {
            std::unique_lock<std::mutex> lock(mutex);
            completion_queue.emplace_back(dest, centroid_id);
            cv.notify_all();
        };

        batch_read_(offset, issue_size, dest, callback);

        // sorted_read_reqs.emplace_back(offset, issue_size, issue_data.data() + issue_data_off);

        // issue_offsets.emplace_back(line_size_ * centroid_id);
        // issue_sizes.emplace_back(issue_size);
        issue_data_size += issue_size;
        // issue_bucket_inner_ids.insert(issue_bucket_inner_ids.end(), bucket->begin(), bucket->end());
        issue_count += 1;
    }

    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(query->GetFloat32Vectors());

    for (uint64_t i = 0; i < issue_count; i++) {
        std::pair<uint8_t*, InnerIdType> element;

        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !completion_queue.empty(); });
            element = completion_queue.front();
            completion_queue.pop_front();
        }

        const uint8_t* vecs = element.first;
        auto& bucket = *(buckets_->at(element.second));

        for (int j = 0; j < bucket.size(); j++) {
            const uint8_t* codes = vecs + j * code_size_;
            auto bucket_inner_id = bucket[j];
            if (vl->Get(bucket_inner_id))
                continue;
            vl->Set(bucket_inner_id);
            float d;
            computer->ComputeDist(codes, &d);
            result.emplace(d, bucket_inner_id);
        }
        cmp_count_ += bucket.size();
    }

    io_total_count_ += issue_count;
    io_total_size_ += issue_data_size;

    pool_->ReturnOne(vl);

    while (result.size() > k) result.pop();

    // transform inner id --> outer label
    auto [dataset_result, dists, ids] = CreateFastDataset(k, allocator_);
    auto count = static_cast<const int64_t>(result.size());
    for (int64_t j = count - 1; j >= 0; --j) {
        auto [dist, inner_id] = result.top();
        dists[j] = dist;
        ids[j] = label_table_->GetLabelById(inner_id);
        result.pop();
    }
    return std::move(dataset_result);
}

void
UGraph::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, dim_);
    StreamWriter::WriteObj(writer, metric_);
    StreamWriter::WriteObj(writer, max_capacity_);
    StreamWriter::WriteObj(writer, num_elements_);
    StreamWriter::WriteObj(writer, capacity_);
    StreamWriter::WriteString(writer, bucket_file_);
    StreamWriter::WriteObj(writer, line_size_);
    StreamWriter::WriteObj(writer, code_size_);
    StreamWriter::WriteObj(writer, entry_point_);
    StreamWriter::WriteObj(writer, use_quantization_);

    // Graph ids inner label table
    StreamWriter::WriteVector(writer, core_ids_);
    StreamWriter::WriteVector(writer, label_table_->label_table_);

    uint64_t label_size = label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, label_size);
    for (auto [key, value] : label_table_->label_remap_) {
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, value);
    }

    // Graph
    if (use_quantization_)
        low_precision_graph_flatten_codes_->Serialize(writer);
    else
        graph_flatten_codes_->Serialize(writer);
    graph_->Serialize(writer);

    // Bucket ids
    for (int i = 0; i < buckets_->size(); i++) {
        auto& bucket = *(buckets_->at(i));
        StreamWriter::WriteVector(writer, bucket);
    }

    StreamWriter::WriteVector(writer, buckets_offset_);
}

void
UGraph::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, dim_);
    StreamReader::ReadObj(reader, metric_);
    StreamReader::ReadObj(reader, max_capacity_);
    StreamReader::ReadObj(reader, num_elements_);
    StreamReader::ReadObj(reader, capacity_);
    bucket_file_ = StreamReader::ReadString(reader);
    StreamReader::ReadObj(reader, line_size_);
    StreamReader::ReadObj(reader, code_size_);
    StreamReader::ReadObj(reader, entry_point_);
    StreamReader::ReadObj(reader, use_quantization_);

    // Graph ids
    StreamReader::ReadVector(reader, core_ids_);
    StreamReader::ReadVector(reader, label_table_->label_table_);

    uint64_t label_size;
    StreamReader::ReadObj(reader, label_size);
    for (uint64_t i = 0; i < label_size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }

    // Graph
    if (use_quantization_) {
        low_precision_graph_flatten_codes_ =
            FlattenInterface::MakeInstance(low_precision_graph_flatten_codes_param_, common_param_);
        low_precision_graph_flatten_codes_->Deserialize(reader);
    } else {
        graph_flatten_codes_ =
            FlattenInterface::MakeInstance(graph_flatten_codes_param_, common_param_);
        graph_flatten_codes_->Deserialize(reader);
    }
    graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);
    graph_->Deserialize(reader);

    // Bucket ids
    buckets_ = std::make_shared<InnerIdBucket>(allocator_);
    for (int i = 0; i < core_ids_.size(); i++) {
        auto bucket = std::make_unique<Vector<InnerIdType>>(allocator_);
        StreamReader::ReadVector(reader, *bucket);
        buckets_->emplace_back(std::move(bucket));
    }

    StreamReader::ReadVector(reader, buckets_offset_);

#ifdef OMYDEBUG
    {
        int64_t bucket_num = buckets_->size();
        int64_t empty_num = 0;
        for (int64_t i = 0; i < bucket_num; i++) {
            auto& bucket = buckets_->at(i);
            if (bucket->empty()) {
                empty_num += 1;
            }
        }
        std::cout << "Bucket number: " << bucket_num << std::endl;
        std::cout << "Empty number: " << empty_num << std::endl;
    }
#endif

    std::cout << "Graph point number: " << core_ids_.size() << std::endl;
    resize(num_elements_);
}

void
UGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_;
    // new_size = next_multiple_of_power_of_two(new_size, resize_increase_count_bit_);
    if (cur_size < new_size) {
        // core_ids_.resize(new_size);
        // if (graph_)
        //     graph_->Resize(new_size);
        // buckets_->resize(new_size);
        pool_ = std::make_unique<VisitedListPool>(1, allocator_, new_size, allocator_);
        label_table_->label_table_.resize(new_size);
        this->max_capacity_ = new_size;
    }
}

static const std::string UGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "ugraph",
        "graph": {
            "io_params": {
                "type": "block_memory_io",
                "file_path": "./default_file_path"
            },
            "max_degree": 16,
            "init_capacity": 100
        },
        "odescent": {
            "max_degree": 16,
            "alpha": 1.2,
            "graph_iter_turn": 30,
            "neighbor_sample_rate": 0.3,
            "min_in_degree": 4,
            "build_block_size": 100
        },
        "base_codes": {
            "io_params": {
                "type": "block_memory_io",
                "file_path": "./default_file_path"
            },
            "codes_type": "flatten_codes",
            "quantization_params": {
                "type": "fp32",
                "use_quantization": "false"
            }
        },
        "quantization_codes": {
            "io_params": {
                "type": "block_memory_io",
                "file_path": "./default_file_path_qt"
            },
            "codes_type": "flatten_codes",
            "quantization_params": {
                "type": "sq8_uniform"
            }
        },
        "build_params": {
            "use_quantization": false,
            "build_thread_count": 100,
            "capacity": 48,
            "bucket_file": "/data/index/test_ugraph_sift/sift_buckets.bin"
        }
    })";

ParamPtr
UGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    std::string default_param_str = format_map(UGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto param_json = JsonType::parse(default_param_str);

    auto pagraph_param = std::make_shared<UGraphParameter>();
    pagraph_param->FromJson(param_json);
    return pagraph_param;
}

}  // namespace vsag
