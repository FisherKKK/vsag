
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

#include "flow_searcher.h"

#include <omp.h>

#include <iostream>
#include <limits>

#include "utils/linear_congruential_generator.h"

float CALL_NUMBER[256];
float PUSH_DOWN_NUMBER[256];
namespace vsag {

namespace {

struct SearchContext {
    /** 0-init (init and pass ep calculation)
     *  1-entry (load candidate_set, result set)
     *  2-search (best-first search)
     */
    explicit SearchContext(Allocator* allocator,
                           float radius,
                           float skip_ratio,
                           uint32_t prefetch_jump_visit_size,
                           InnerIdType ep,
                           uint64_t ef,
                           uint64_t code_size,
                           void* query,
                           const FilterPtr& is_id_allowed,
                           MaxHeap& top_candidates,
                           MaxHeap& candidate_set,
                           const VisitedListPtr& vl,
                           const MutexArrayPtr& mutex_array,
                           const GraphInterfacePtr& graph)
        : radius_(radius),
          skip_ratio_(skip_ratio),
          prefetch_jump_visit_size_(prefetch_jump_visit_size),
          ep_(ep),
          ef_(ef),
          code_size_(code_size),
          query_(query),
          is_id_allowed_(is_id_allowed),
          top_candidates_(top_candidates),
          candidate_set_(candidate_set),
          vl_(vl),
          mutex_array_(mutex_array),
          graph_(graph),
          neighbors_(graph->MaximumDegree() * LOOK_AHEAD, allocator),
          allocator_(allocator) {
    }

    uint8_t state_{0};
    float lower_bound_{std::numeric_limits<float>::max()};
    float radius_{0};
    float skip_ratio_{0.8f};
    uint32_t prefetch_jump_visit_size_{0};
    InnerIdType ep_{0};
    uint64_t ef_{10};
    uint64_t code_size_{0};
    void* query_{nullptr};
    const FilterPtr& is_id_allowed_;

    Vector<InnerIdType> neighbors_;
    MaxHeap& top_candidates_;
    MaxHeap& candidate_set_;
    const VisitedListPtr& vl_;
    const MutexArrayPtr& mutex_array_;
    const GraphInterfacePtr& graph_;
    Allocator* allocator_;
    static constexpr int LOOK_AHEAD = 2;
};

template <InnerSearchMode mode>
int
flow_search_fn(hnsw_search_opt* search_opt) {
    // unpack necessary var
    auto search_context = static_cast<SearchContext*>(search_opt->user_data);
    auto is_id_allowed = search_context->is_id_allowed_;
    auto ef = search_context->ef_;
    auto ep = search_context->ep_;
    auto code_size = search_context->code_size_;
    auto radius = search_context->radius_;
    auto skip_ratio = search_context->skip_ratio_;
    auto prefetch_jump_visit_size = search_context->prefetch_jump_visit_size_;
    auto query = search_context->query_;

    auto& top_candidates = search_context->top_candidates_;
    auto& candidate_set = search_context->candidate_set_;
    auto& lower_bound = search_context->lower_bound_;
    auto& neighbors = search_context->neighbors_;
    auto& vl = search_context->vl_;
    auto& mutex_array = search_context->mutex_array_;
    auto& graph = search_context->graph_;

    auto dist_line = search_opt->copt.result_list;
    auto ids_line = search_opt->copt.ids_list;
    auto& ids_size = search_opt->copt.ids_size;

    Allocator* allocator = search_context->allocator_;

    search_opt->query_vector = query;
    search_opt->query_vector_size = code_size;

    auto thread_num = omp_get_thread_num();

    // first enter, directly calculate the entry point distance
    if (search_context->state_ == 0) {
        search_context->state_ = 1;
        ids_size = 1;
        ids_line[0] = ep;
        // std::cout << "Entry point..." << std::endl;
#ifdef GET_ALIFLASH_INFO
        {
            CALL_NUMBER[thread_num] += 1;
            PUSH_DOWN_NUMBER[thread_num] += 1;
        }
#endif
        return 0;
    }

    // second enter, get the distance of entry point
    if (search_context->state_ == 1) {
        auto dist = dist_line[0];
        if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
            top_candidates.emplace(dist, ep);
            lower_bound = top_candidates.top().first;
        }
        candidate_set.emplace(-dist, ep);
        vl->Set(ep);
        search_context->state_ = 2;
        // std::cout << "Entry point into pq..." << std::endl;
    } else if (search_context->state_ == 2) {
        // 1. deal with last calculation's result, emplace to the candidate
        for (uint32_t i = 0; i < ids_size; i++) {
            auto dist = dist_line[i];
            if (top_candidates.size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= radius)) {
                candidate_set.emplace(-dist, ids_line[i]);
                //                flatten->Prefetch(candidate_set.top().second);
                if (not is_id_allowed ||
                    is_id_allowed->CheckValid(static_cast<int64_t>(ids_line[i]))) {
                    top_candidates.emplace(dist, ids_line[i]);
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }

                if (not top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
        // std::cout << "Process last result..." << std::endl;
    }

    // reset the ids size
    ids_size = 0;

    // 2. prepare for next calculation must larger than 0
    //! we should record the average cal times
    Vector<InnerIdType> look_ahead_ids(allocator);
    look_ahead_ids.reserve(SearchContext::LOOK_AHEAD);

    while (not candidate_set.empty() && ids_size < 1) {
        auto current_node_pair = candidate_set.top();
        look_ahead_ids.emplace_back(current_node_pair.second);
        if constexpr (mode == KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates.size() >= ef) {
                // complete search
                return 1;
            }
        }

        candidate_set.pop();

        if (not candidate_set.empty()) {
            graph->Prefetch(candidate_set.top().second, 0);
        }

        // look ahead the candidate
        for (int lk = 1; lk < SearchContext::LOOK_AHEAD && not candidate_set.empty(); lk++) {
            look_ahead_ids.emplace_back(candidate_set.top().second);
            candidate_set.pop();
        }

        LinearCongruentialGenerator generator;
        uint32_t count_no_visited = 0;

        Vector<InnerIdType> tmp_nbr(allocator);
        InnerIdType cur_nbr_size = 0;

        // clear the nbrs
        neighbors.clear();

        for (auto look_ahead_id : look_ahead_ids) {
            if (mutex_array != nullptr) {
                SharedLock lock(mutex_array, look_ahead_id);
                graph->GetNeighbors(look_ahead_id, tmp_nbr);
            } else {
                graph->GetNeighbors(look_ahead_id, tmp_nbr);
            }
            auto nbr_sz = tmp_nbr.size();
            memcpy(neighbors.data() + cur_nbr_size, tmp_nbr.data(), sizeof(InnerIdType) * nbr_sz);
            cur_nbr_size += nbr_sz;
        }

        float skip_threshold = (is_id_allowed != nullptr
                                    ? (is_id_allowed->ValidRatio() == 1.0F
                                           ? 0
                                           : (1 - ((1 - is_id_allowed->ValidRatio()) * skip_ratio)))
                                    : 0.0F);

        for (uint32_t i = 0; i < prefetch_jump_visit_size and cur_nbr_size > i; i++) {
            vl->Prefetch(neighbors[i]);
        }

        for (uint32_t i = 0; i < cur_nbr_size; i++) {
            if (i + prefetch_jump_visit_size < cur_nbr_size) {
                vl->Prefetch(neighbors[i + prefetch_jump_visit_size]);
            }
            if (not vl->Get(neighbors[i])) {
                if (not is_id_allowed || count_no_visited == 0 ||
                    generator.NextFloat() > skip_threshold ||
                    is_id_allowed->CheckValid(neighbors[i])) {
                    ids_line[count_no_visited] = neighbors[i];
                    // to_be_visited_rid[count_no_visited] = i;
                    // to_be_visited_id[count_no_visited] = neighbors[i];
                    count_no_visited++;
                }
                vl->Set(neighbors[i]);
            }
        }
        ids_size = count_no_visited;

        // std::cout << "Input nodes need to be calculated: " << ids_size << " ..." << std::endl;
    }

    if (ids_size > 0) {
#ifdef GET_ALIFLASH_INFO
        {
            CALL_NUMBER[thread_num] += 1;
            PUSH_DOWN_NUMBER[thread_num] += ids_size;
        }
#endif
        return 0;
    }

    // if empty, search complete
    return 1;
};

}  // namespace

FlowSearcher::FlowSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)) {
    client_ = AliFlashClient::GetInstance(common_param.dim_);
    std::cout << "Using flow searcher" << std::endl;
}

uint32_t
FlowSearcher::visit(const GraphInterfacePtr& graph,
                    const VisitedListPtr& vl,
                    const std::pair<float, uint64_t>& current_node_pair,
                    const FilterPtr& filter,
                    float skip_ratio,
                    Vector<InnerIdType>& to_be_visited_rid,
                    Vector<InnerIdType>& to_be_visited_id,
                    Vector<InnerIdType>& neighbors) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        SharedLock lock(this->mutex_array_, current_node_pair.second);
        graph->GetNeighbors(current_node_pair.second, neighbors);
    } else {
        graph->GetNeighbors(current_node_pair.second, neighbors);
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);

    for (uint32_t i = 0; i < prefetch_jump_visit_size_ and neighbors.size() > i; i++) {
        vl->Prefetch(neighbors[i]);
    }

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_jump_visit_size_ < neighbors.size()) {
            vl->Prefetch(neighbors[i + prefetch_jump_visit_size_]);
        }
        if (not vl->Get(neighbors[i])) {
            if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                filter->CheckValid(neighbors[i])) {
                to_be_visited_rid[count_no_visited] = i;
                to_be_visited_id[count_no_visited] = neighbors[i];
                count_no_visited++;
            }
            vl->Set(neighbors[i]);
        }
    }
    return count_no_visited;
}

MaxHeap
FlowSearcher::Search(const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     const VisitedListPtr& vl,
                     const float* query,
                     const InnerSearchParam& inner_search_param) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(graph, flatten, vl, query, inner_search_param);
    }
    return this->search_impl<RANGE_SEARCH>(graph, flatten, vl, query, inner_search_param);
}

MaxHeap
FlowSearcher::Search(const GraphInterfacePtr& graph,
                     const FlattenInterfacePtr& flatten,
                     const VisitedListPtr& vl,
                     const float* query,
                     const InnerSearchParam& inner_search_param,
                     IteratorFilterContext* iter_ctx) const {
    return this->search_impl<KNN_SEARCH>(graph, flatten, vl, query, inner_search_param, iter_ctx);
}

template <InnerSearchMode mode>
MaxHeap
FlowSearcher::search_impl(const GraphInterfacePtr& graph,
                          const FlattenInterfacePtr& flatten,
                          const VisitedListPtr& vl,
                          const float* query,
                          const InnerSearchParam& inner_search_param,
                          IteratorFilterContext* iter_ctx) const {
    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    float dist = 0.0F;
    uint64_t ids_cnt = 1;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph->MaximumDegree(), allocator_);
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), allocator_);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), allocator_);
    Vector<float> line_dists(graph->MaximumDegree(), allocator_);

    if (!iter_ctx->IsFirstUsed()) {
        if (iter_ctx->Empty()) {
            return top_candidates;
        }
        while (!iter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_ctx->GetTopID();
            float cur_dist = iter_ctx->GetTopDist();
            if (!vl->Get(cur_inner_id) && iter_ctx->CheckPoint(cur_inner_id)) {
                vl->Set(cur_inner_id);
                lower_bound = std::max(lower_bound, cur_dist);
                flatten->Query(&cur_dist, computer, &cur_inner_id, 1);
                top_candidates.emplace(cur_dist, cur_inner_id);
                candidate_set.emplace(cur_dist, cur_inner_id);
                if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                    if (cur_dist > inner_search_param.radius and not top_candidates.empty()) {
                        top_candidates.pop();
                    }
                }
            }
            iter_ctx->PopDiscard();
        }
    } else {
        flatten->Query(&dist, computer, &ep, 1);
        if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
            top_candidates.emplace(dist, ep);
            lower_bound = top_candidates.top().first;
        }
        candidate_set.emplace(-dist, ep);
        vl->Set(ep);
    }

    while (not candidate_set.empty()) {
        hops++;
        auto current_node_pair = candidate_set.top();

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates.size() == ef) {
                break;
            }
        }
        candidate_set.pop();

        if (not candidate_set.empty()) {
            graph->Prefetch(candidate_set.top().second, 0);
        }

        count_no_visited = visit(graph,
                                 vl,
                                 current_node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors);

        dist_cmp += count_no_visited;

        flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (top_candidates.size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                if (!iter_ctx->CheckPoint(to_be_visited_id[i])) {
                    continue;
                }
                candidate_set.emplace(-dist, to_be_visited_id[i]);
                flatten->Prefetch(candidate_set.top().second);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates.emplace(dist, to_be_visited_id[i]);
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates.size() > ef) {
                        if (iter_ctx->CheckPoint(top_candidates.top().second)) {
                            auto cur_node_pair = top_candidates.top();
                            iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
                        }
                        top_candidates.pop();
                    }
                }

                if (not top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates.size() > inner_search_param.topk) {
            auto cur_node_pair = top_candidates.top();
            if (iter_ctx->CheckPoint(cur_node_pair.second)) {
                iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
            }
            top_candidates.pop();
        }
    }

    return top_candidates;
}

template <InnerSearchMode mode>
MaxHeap
FlowSearcher::search_impl(const GraphInterfacePtr& graph,
                          const FlattenInterfacePtr& flatten,
                          const VisitedListPtr& vl,
                          const float* query,
                          const InnerSearchParam& inner_search_param) const {
    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto search_context = std::make_unique<SearchContext>(allocator_,
                                                          inner_search_param.radius,
                                                          inner_search_param.skip_ratio,
                                                          prefetch_jump_visit_size_,
                                                          inner_search_param.ep,
                                                          inner_search_param.ef,
                                                          flatten->code_size_,
                                                          (void*)query,
                                                          inner_search_param.is_inner_id_allowed,
                                                          top_candidates,
                                                          candidate_set,
                                                          vl,
                                                          mutex_array_,
                                                          graph);
    client_->context_search(flow_search_fn<KNN_SEARCH>, search_context.get());

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates.size() > inner_search_param.topk) {
            top_candidates.pop();
        }
    } else if constexpr (mode == RANGE_SEARCH) {
        if (inner_search_param.range_search_limit_size > 0) {
            while (top_candidates.size() > inner_search_param.range_search_limit_size) {
                top_candidates.pop();
            }
        }
        while (not top_candidates.empty() &&
               top_candidates.top().first > inner_search_param.radius + THRESHOLD_ERROR) {
            top_candidates.pop();
        }
    }

    return top_candidates;
}

}  // namespace vsag
