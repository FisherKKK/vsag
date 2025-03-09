
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

#include "pag.h"

#include "quantization/fp32_quantizer.h"
#include "utils/util_functions.h"
#include "impl/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "lock_strategy.h"

namespace vsag {

PAGraph::PAGraph(const PAGraphParameterPtr& pag_param, const IndexCommonParam& common_param)
    : InnerIndexInterface(pag_param, common_param),
      common_param_(common_param),
      dim_(common_param.dim_),
      metric_(common_param.metric_),
      num_iter_(pag_param->num_iter_),
      replicas_(pag_param->replicas_),
      capacity_(pag_param->capacity_),
      sample_rate_(pag_param->sample_rate_),
      start_decay_rate_(pag_param->start_decay_rate_),
      ef_(pag_param->ef_),
      fine_radius_rate_(pag_param->fine_radius_rate_),
      coarse_radius_rate_(pag_param->coarse_radius_rate_),
      graph_ids_(common_param.allocator_.get()),
      radii_(common_param.allocator_.get()),
      graph_flatten_codes_param_(pag_param->graph_flatten_codes_param_),
      graph_param_(pag_param->graph_param_),
      odescent_param_(pag_param->odescent_param_) {
    line_size_ = capacity_ * sizeof(float); // TODO: for float
    code_size_ = dim_ * sizeof(float);
    pool_ = std::make_unique<VisitedListPool>(
    1, common_param.allocator_.get(), max_capacity_, common_param_.allocator_.get());
    searcher_ = std::make_unique<BasicSearcher>(common_param_);
    io_ = std::make_unique<AsyncIO>(bucket_file_, common_param.allocator_.get()); // TODO: for async io
    quantizer_ = std::make_unique<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>>(
        common_param.dim_, common_param.allocator_.get()); // TODO: quantizer for bucket calculation
}

std::vector<int64_t>
PAGraph::Build(const DatasetPtr& base) {
    auto num_elements = base->GetNumElements();
    const auto* base_vectors = base->GetFloat32Vectors();
    const auto* base_ids = base->GetIds();

    // Only point for inner id --> outer label mapping
    memcpy(this->label_table_->label_table_.data(), base_ids, sizeof(LabelType) * num_elements);
    num_sample_ = static_cast<uint64_t>(num_elements * sample_rate_);
    auto lower_num_sample = static_cast<int64_t>(num_sample_ * start_decay_rate_);

    pool_ = std::make_unique<VisitedListPool>(
        1, allocator_, max_capacity_, allocator_);

    // flatten_interface_ptr_->Train(base_vectors, num_elements);
    // flatten_interface_ptr_->BatchInsertVector(base_vectors, num_elements);

    //! All of the id current is the inner id, and in search stage will convert to outer label
    // Sample inner ids for inner index
    graph_ids_ = sample_graph_ids(num_elements, lower_num_sample);
    // auto cur_graph_ids_set = UnorderedSet<InnerIdType>(common_param_.allocator_.get());

    for (uint64_t iter = 0; iter < num_iter_; iter++) {
        // Keep the codes of graph
        graph_flatten_codes_ = FlattenInterface::MakeInstance(graph_flatten_codes_param_, common_param_);
        graph_flatten_codes_->SetMaxCapacity(num_sample_);
        // auto residual_flatten_codes = FlattenInterface::MakeInstance(pag_param_.flatten_data_cell_param, common_param_);

        // Keep the graph structure
        graph_ = GraphInterface::MakeInstance(graph_param_, common_param_);
        graph_->Resize(num_sample_);
        // Here we need the mapping from graph-inner id --> inner id,
        // and cur_graph_ids is the ideal mapping
        // graph_ids

        // auto &graph_inner_ids = cur_graph_ids;

        // UnorderedMap<InnerIdType, InnerIdType> graph_label(common_param_.allocator_);
        // memcpy(graph_labels.label_table_.data(), )


        // auto residual_labels = LabelTable(common_param_.allocator_.get());

        // Bucket which stores key(graph inner id) and value(inner ids)
        // graph_inner_id: [inner_id1, inner_id2, inner_id3]
        // meanwhile reserve the elements
        buckets_ = std::make_shared<InnerIdBucket>(allocator_);
        buckets_->reserve(num_sample_);
        radii_.reserve(num_sample_);
        for (auto _: graph_ids_) {
            buckets_->emplace_back(std::make_unique<Vector<InnerIdType>>(allocator_));
            buckets_->back()->reserve(capacity_);
        }

        //TODO: whether keep the sample sequence
        // for (auto graph_id: cur_inner_ids) {
        //     const auto* vec = base_vectors + graph_id * code_size;
        //     graph_flatten_codes->InsertVector(vec);
        // }

        // int64_t graph_count = 0, residual_count = 0;

        // Insert the vectors to  graph flatten code
        for (auto graph_id: graph_ids_) {
            const auto* vec = base_vectors + graph_id * code_size_;
            graph_flatten_codes_->InsertVector(vec);
        }


        // for (int64_t i = 0; i < num_elements; i++) {
        //     const auto* vec = base_vectors + i * code_size;
        //     if (cur_graph_ids_set.count(i)) {
        //         graph_flatten_codes->InsertVector(vec);
        //         graph_labels.Insert(graph_count, i);
        //         graph_count += 1;
        //     }
            // else {
            //     residual_flatten_codes->InsertVector(vec);
            //     graph_labels.Insert(residual_count, i);
            //     residual_count += 1;
            // }
        // }


        // bucket_.clear();
        // for (auto graph_id: cur_graph_ids) {
        //     bucket_.emplace(graph_id, std::make_unique<Vector<InnerIdType>>(common_param_.allocator_.get()));
        // }

        ODescent graph_builder(odescent_param_,
                               graph_flatten_codes_,
                               allocator_,
                               common_param_.thread_pool_.get());
        graph_builder.Build();
        graph_builder.SaveGraph(graph_);

        // Get radius of graph, radii is graph_inner_id driven
        get_radius();
        // Aggregate residual point to graph point
        aggregate_pag(base);

        if (iter == num_iter_ - 1)
            break;
        // Update the centroid
        Vector<InnerIdType> tmp_inner_ids(common_param_.allocator_.get());
        tmp_inner_ids.reserve(graph_ids_.size());

        for (int i = 0; i < buckets_->size(); i++) {
            auto& members = buckets_->at(i);
            if (members->size() > 0) {
                members->emplace_back(graph_ids_[i]);
                auto new_centroid = calculate_new_centroid(*(members.get()), base);
                tmp_inner_ids.emplace_back(new_centroid);
            }
        }
        tmp_inner_ids.swap(graph_ids_);
    }



    // Remap all graph_id to outer id

    // for (int i = 0; i < cur_graph_ids.size(); i++) {
    //     graph_label_table_->Insert(i, base_ids[graph_ids_[i]]);
    //         // labels_.GetLabelById(cur_graph_ids[i]));
    // }

    // Remap bucket inner id -> outer id
    // InnerIdBucket buckets;
    // for (auto& members_ptr: buckets) {
    //     auto members = *members_ptr;
    //     for (auto &inner_id: members) {
    //         inner_id = base_ids[inner_id]; // labels_.GetLabelById(inner_id);
    //     }
    // }

    // Keep bucket vectors
    // Now we only store vectors in secondary storage
    // offset can be calculated by graph_id * line_size
    // bucket size and bucket ids will keep in memory (TODO: can be optimized along with vector)
    std::ofstream stream(bucket_file_, std::ios_base::out | std::ios_base::binary);
    // TODO: error check
    Vector<float> dummy_vec(dim_, 0.f, allocator_);
    for (int i = 0; i < graph_ids_.size(); i++) {
        auto &members_ptr = buckets_->at(i);
        auto members = *members_ptr;
        for (auto member: members) {
            const auto *vec = base_vectors + member * dim_;
            stream.write((char*)vec, sizeof(*vec) * dim_);
        }

        for (int i = 0; i < capacity_ - members.size(); i++) {
            stream.write((char*)dummy_vec.data(), sizeof(float) * dim_);
        }
    }
}

DatasetPtr
PAGraph::KnnSearch(const DatasetPtr& query,
                   int64_t k,
                   const std::string& parameters,
                   const FilterPtr& filter) const {
    MaxHeap result(common_param_.allocator_.get());
    auto vl = pool_->TakeOne();

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_;
    search_param.ef = 200;

    auto graph_id_result = searcher_->Search(
        graph_, graph_flatten_codes_, vl, query->GetFloat32Vectors(), search_param);

    Vector<uint8_t> issue_data(allocator_);
    Vector<uint64_t> issue_sizes(allocator_);
    Vector<uint64_t> issue_offsets(allocator_);
    Vector<float> issue_dists(allocator_);
    Vector<InnerIdType> issue_bucket_inner_ids(allocator_);

    uint64_t issue_count = 0, issue_data_size = 0;

    while (graph_id_result.size() > 0) {
        auto [centroid_dist, centroid_id] = graph_id_result.top();
        auto inner_id = graph_ids_[centroid_id];
        graph_id_result.pop();
        result.emplace(centroid_dist, inner_id);
        vl->Set(inner_id);
        if (result.size() > k) result.pop();

        auto &bucket = buckets_->at(centroid_id);
        auto issue_size = bucket->size() * code_size_;
        if (issue_size == 0) continue;

        issue_offsets[issue_count] = line_size_ * centroid_id;
        issue_sizes[issue_count] = issue_size;
        issue_data_size += issue_size;
        issue_bucket_inner_ids.emplace(issue_bucket_inner_ids.end(), bucket->begin(), bucket->end());
        issue_count += 1;
    }

    // ByteBuffer codes(issue_data_size, allocator_);

    io_->MultiRead(issue_data.data(), issue_sizes.data(), issue_offsets.data(), issue_count);


    auto computer = quantizer_->FactoryComputer();
    computer->SetQuery(query->GetFloat32Vectors());

    for (int i = 0; i < issue_bucket_inner_ids.size(); i++) {
        auto bucket_inner_id = issue_bucket_inner_ids[i];
        auto *codes = issue_data.data() + i * code_size_;
        if (vl->Get(bucket_inner_id)) continue;
        vl->Set(bucket_inner_id);
        float d;
        computer->ComputeDist(codes, &d);
        result.emplace(d, bucket_inner_id);
        if (result.size() > k) result.pop();
    }

    pool_->ReturnOne(vl);

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
PAGraph::Serialize(StreamWriter& writer) const {
    // Basic info
    StreamWriter::WriteObj(writer, dim_);
    StreamWriter::WriteObj(writer, metric_);
    StreamWriter::WriteObj(writer, max_capacity_);
    StreamWriter::WriteObj(writer, num_elements_);
    StreamWriter::WriteObj(writer, num_iter_);
    StreamWriter::WriteObj(writer, replicas_);
    StreamWriter::WriteObj(writer, capacity_);
    StreamWriter::WriteObj(writer, sample_rate_);
    StreamWriter::WriteObj(writer, start_decay_rate_);
    StreamWriter::WriteObj(writer, num_sample_);
    StreamWriter::WriteObj(writer, ef_);
    StreamWriter::WriteString(writer, bucket_file_);
    StreamWriter::WriteObj(writer, line_size_);
    StreamWriter::WriteObj(writer, code_size_);
    StreamWriter::WriteObj(writer, fine_radius_rate_);
    StreamWriter::WriteObj(writer, coarse_radius_rate_);
    StreamWriter::WriteObj(writer, upper_bound_radius_);
    StreamWriter::WriteObj(writer, entry_point_);

    // Graph ids inner label table
    StreamWriter::WriteVector(writer, graph_ids_);
    StreamWriter::WriteVector(writer, label_table_->label_table_);


    uint64_t label_size = label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, label_size);
    for (auto [key, value]: label_table_->label_remap_) {
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, value);
    }

    // Graph
    graph_flatten_codes_->Serialize(writer);
    graph_->Serialize(writer);

    // Bucket ids
    for (int i = 0; i < buckets_->size(); i++) {
        auto &bucket = *(buckets_->at(i));
        StreamWriter::WriteVector(writer, bucket);
    }
}

void
PAGraph::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, dim_);
    StreamReader::ReadObj(reader, metric_);
    StreamReader::ReadObj(reader, max_capacity_);
    StreamReader::ReadObj(reader, num_elements_);
    StreamReader::ReadObj(reader, num_iter_);
    StreamReader::ReadObj(reader, replicas_);
    StreamReader::ReadObj(reader, capacity_);
    StreamReader::ReadObj(reader, sample_rate_);
    StreamReader::ReadObj(reader, start_decay_rate_);
    StreamReader::ReadObj(reader, num_sample_);
    StreamReader::ReadObj(reader, ef_);
    bucket_file_ = StreamReader::ReadString(reader);
    StreamReader::ReadObj(reader, line_size_);
    StreamReader::ReadObj(reader, code_size_);
    StreamReader::ReadObj(reader, fine_radius_rate_);
    StreamReader::ReadObj(reader, coarse_radius_rate_);
    StreamReader::ReadObj(reader, upper_bound_radius_);
    StreamReader::ReadObj(reader, entry_point_);

    // Graph ids
    StreamReader::ReadVector(reader, graph_ids_);
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
    graph_flatten_codes_->Deserialize(reader);
    graph_->Deserialize(reader);

    // Bucket ids
    for (int i = 0; i < graph_ids_.size(); i++) {
        auto bucket = std::make_unique<Vector<InnerIdType>>();
        StreamReader::ReadVector(reader, *bucket);
        buckets_->emplace_back(std::move(bucket));
    }
}

std::vector<int64_t>
PAGraph::Add(const DatasetPtr& base) {
    return {};
}

std::string
PAGraph::GetName() const {
    return "pagraph";
}


int64_t
PAGraph::GetNumElements() const {
    return num_elements_;
}


DatasetPtr
PAGraph::RangeSearch(const DatasetPtr& query,
                     float radius,
                     const std::string& parameters,
                     const FilterPtr& filter,
                     int64_t limited_size) const {
    return Dataset::Make();
}


Vector<InnerIdType>
PAGraph::sample_graph_ids(int64_t num_elements, int64_t num_sample) {
    Vector<InnerIdType> reservoir(common_param_.allocator_.get());
    reservoir.resize(num_sample);

    for (int64_t i = 0; i < num_sample; i++) reservoir[i] = i;

    for (int64_t i = num_sample; i < num_elements; i++) {
        std::uniform_int_distribution<InnerIdType> distribution(0, i);
        int64_t j = distribution(generator_);
        if (j < num_sample)
            reservoir[j] = i;
    }
    return std::move(reservoir);
}

void
PAGraph::get_radius() {
    auto graph_size = graph_->TotalCount();

    Vector<float> all_radius(graph_size, allocator_);
    all_radius.reserve(graph_size);


    for (InnerIdType v = 0; v < graph_size; v++) {
        Vector<float> distances(common_param_.allocator_.get());
        distances.reserve(32);

        Vector<InnerIdType> nns(common_param_.allocator_.get());
        graph_->GetNeighbors(v, nns);
        for (auto nn: nns) {
            float d = graph_flatten_codes_->ComputePairVectors(v, nn);
            distances.emplace_back(d);
        }

        size_t nth = std::min(static_cast<size_t>(nns.size() * fine_radius_rate_),
                              nns.size());
        std::nth_element(distances.begin(), distances.begin() + nth, distances.end());
        float r = *(distances.begin() + nth);
        radii_.emplace_back(r);
        all_radius.emplace_back(r);
    }

    // Calculate coarse-grained radius
    size_t coarse_nth = static_cast<size_t>(all_radius.size() * coarse_radius_rate_);
    std::nth_element(all_radius.begin(), all_radius.begin() + coarse_nth,
                    all_radius.end());
    upper_bound_radius_ = *(all_radius.begin() + coarse_nth);

    // Min-Max the radii
    for (auto &r: radii_) {
        r = std::min(r, upper_bound_radius_);
    }
}

void
PAGraph::aggregate_pag(const DatasetPtr &base) {
    auto total_count = base->GetNumElements();
    auto dim = base->GetDim();
    const auto *base_vecs = base->GetFloat32Vectors();

    UnorderedSet<InnerIdType> graph_ids_set(allocator_);
    graph_ids_set.insert(graph_ids_.begin(), graph_ids_.end());

    InnerSearchParam search_param;
    search_param.ep = entry_point_;
    search_param.ef = ef_;
    auto empty_mutex = std::make_shared<EmptyMutex>();

    for (InnerIdType i = 0; i < total_count; i++) {
        if (graph_ids_set.count(i)) continue;
        const float *q = base_vecs + i * dim; //TODO: only for float32
        // residual_flatten_ptr->GetCodesById(i, codes.data());

        auto vl = pool_->TakeOne();
        auto result = searcher_->Search(
            graph_, graph_flatten_codes_, vl, q, search_param);
        pool_->ReturnOne(vl);

        // copy the result, result is graph_inner_id
        auto result_copy = result;

        select_partition_by_heuristic(result);

        if (result.size() == 0) {
            graph_flatten_codes_->InsertVector(q);
            mutually_connect_new_element(i,
                                         result_copy,
                                         graph_,
                                         graph_flatten_codes_,
                                         empty_mutex,
                                         common_param_.allocator_.get());
            graph_ids_.emplace_back(i);
            buckets_->emplace_back(std::make_unique<Vector<InnerIdType>>(allocator_));
            buckets_->back()->reserve(capacity_);
            auto bound_position = static_cast<size_t>(result_copy.size() * fine_radius_rate_);
            while (result_copy.size() > bound_position) {
                result_copy.pop();
            }
            radii_.emplace_back(std::min(result_copy.top().first, upper_bound_radius_));
        } else {
            while (result.size() > 0) {
                auto partition = result.top().second;
                result.pop();
                if (result.size() > replicas_) continue;
                buckets_->at(partition)->emplace_back(i);
            }
        }
    }


}

void
PAGraph::select_partition_by_heuristic(MaxHeap& candidates) {
    MaxHeap closest_queue(allocator_);
    Vector<std::pair<float, InnerIdType>> return_list(allocator_);

    // Score filter
    while (candidates.size() > 0) {
        auto [dist, graph_id] = candidates.top();

        std::uniform_real_distribution<float> distribution(0.f, 1.f);
        auto probability = distribution(generator_);

        auto radius = radii_[graph_id];
        auto bucket_size = buckets_->at(graph_id)->size();
        auto score = match_score(dist, radius, bucket_size, capacity_);

        if (probability < score || (score != 0.f && graph_->TotalCount() > num_sample_ - 5))
            closest_queue.emplace(-dist, graph_id);

        candidates.pop();
    }

    // Density filter
    while (closest_queue.size() > 0) {
        auto [cur_dist, cur_graph_id] = closest_queue.top();
        cur_dist = -cur_dist;
        closest_queue.pop();

        bool good = true;

        for (auto [pre_dist, pre_graph_id]: return_list) {
            auto dist_partition = graph_flatten_codes_->ComputePairVectors(cur_graph_id, pre_graph_id);
            if (dist_partition < cur_dist) {
                good = false;
                break;
            }
        }

        if (good) {
            return_list.emplace_back(cur_dist, cur_graph_id);
        }

    }

    for (auto [dist, label]: return_list) {
        candidates.emplace(dist, label);
    }
}

float
PAGraph::match_score(float d, float r, size_t cur_bucket_size, size_t bucket_capacity) {
    if (cur_bucket_size >= bucket_capacity) return .0f;
    auto dr_ratio = d / (r + 0.000001f);
    auto bucket_ratio = cur_bucket_size / (float) (bucket_capacity + 0.000001f);

    float dr_factor;
    if (dr_ratio < 0.5f) {
        dr_factor = 1.f;
    } else if (dr_ratio < 1.0f) {
        dr_factor = 0.9f;
    } else if (dr_ratio < 2.0f) {
        dr_factor = 0.8f;
    } else if (dr_ratio < 3.0f) {
        dr_factor = 0.5f;
    } else {
        dr_factor = 0.4f;
    }

    float bucket_factor;
    if (bucket_ratio <= 0.0f) {
        bucket_factor = 1.0f;
    } else if (bucket_ratio <= 0.5f) {
        bucket_factor = 0.95;
    } else if (bucket_ratio <= 0.8f) {
        bucket_factor = 0.8;
    } else {
        bucket_factor = 0.4f;
    }

    return dr_factor * bucket_factor;
}

InnerIdType
PAGraph::calculate_new_centroid(const Vector<InnerIdType> &members, const DatasetPtr &base) {
    const float *base_vecs = base->GetFloat32Vectors();


    Vector<float> sum(common_param_.allocator_.get());
    sum.resize(0.f, dim_);
    for (auto mem: members) {
        const float *vec = base_vecs + dim_ * mem;
        for (int i = 0; i < dim_; i++) {
            sum[i] += vec[i];
        }
    }

    for (int i = 0; i < dim_; i++) {
        sum[i] /= members.size() + 0.00001f;
    }

    float min_dist = std::numeric_limits<float>::max();
    InnerIdType min_id = 0;

    for (int i = 0; i < members.size(); i++) {
        const float *vec = base_vecs + members[i] * dim_;
        auto d = quantizer_->Compute((uint8_t*)vec, (uint8_t*)sum.data()); // TODO: whether better ways
        if (d < min_dist) {
            min_dist = d;
            min_id = members[i];
        }
    }

    return min_id;
}







static uint64_t
next_multiple_of_power_of_two(uint64_t x, uint64_t n) {
    if (n > 63) {
        throw std::runtime_error(fmt::format("n is larger than 63, n is {}", n));
    }
    uint64_t y = 1 << n;
    auto result = (x + y - 1) & ~(y - 1);
    return result;
}

void
PAGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_;
    new_size = next_multiple_of_power_of_two(new_size, resize_increase_count_bit_);
    if (cur_size < new_size) {
        graph_ids_.resize(new_size);
        if (graph_)
            graph_->Resize(new_size);
        radii_.resize(new_size);
        buckets_->resize(new_size);
        pool_ = std::make_unique<VisitedListPool>(
            1, allocator_, new_size, allocator_);
        label_table_->label_table_.resize(new_size);
        this->max_capacity_ = new_size;
    }
}


static const std::string PAGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "pagraph",
        "graph": {
            "io_params": {
                "type": "block_memory_io",
                "file_path": "./default_file_path"
            },
            "max_degree": 32,
            "init_capacity": 100
        },
        "odescent": {
            "max_degree": 32,
            "alpha": 1,
            "graph_iter_turn": 10,
            "neighbor_sample_rate": 0.5,
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
                "use_quantization": false
            }
        },
        "build_params": {
            "build_thread_count": 100,
            "sample_rate": 0.2,
            "start_decay_rate": 0.8,
            "capacity": 32,
            "num_iter": 2,
            "replicas": 4,
            "fine_radius_rate": 0.5,
            "coarse_radius_rate": 0.5
        }
    })";


ParamPtr
PAGraph::MappingExternalParamAndCheck(const JsonType& external_param,
                                      const IndexCommonParam& common_param) {
    std::string default_param_str = format_map(PAGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto param_json = JsonType::parse(default_param_str);

    auto pagraph_param = std::make_shared<PAGraphParameter>();
    pagraph_param->FromJson(param_json);
    return pagraph_param;
}





}