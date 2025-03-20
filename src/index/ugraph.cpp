
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

#include "ugraph.h"

#include <impl/odescent_graph_builder.h>

namespace vsag {

namespace {
struct UnionSet {

    UnionSet(InnerIdType num_elements, Allocator *allocator):
        num_elements_(num_elements), parents_(num_elements, allocator),
        rank_(num_elements, 0, allocator) {
        for (InnerIdType i = 0; i < num_elements; i++) {
            parents_[i] = i;
        }
    }


    InnerIdType Find(InnerIdType x) {
        if (parents_[x] == x) return x;
        return parents_[x] = Find(parents_[x]);
    }


    void Union(InnerIdType x, InnerIdType y) {
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

}


std::vector<int64_t>
UGraph::Build(const DatasetPtr& base) {
    ODescent graph_builder(odescent_param_, graph_flatten_codes_, allocator_, common_param_.thread_pool_.get());
    graph_builder.Build();
    graph_builder.SaveGraph(graph_);

    auto graph_size = graph_->TotalCount();
    MaxHeap<Edge> edges(allocator_);

    for (auto i = 0; i < graph_size; i++) {
        Vector<InnerIdType> nns(allocator_);
        graph_->GetNeighbors(i, nns);
        for (auto nn: nns) {
            auto d = graph_flatten_codes_->ComputePairVectors(i, nn);
            edges.emplace(-d, i, nn);
        }
    }


}


}
