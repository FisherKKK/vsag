
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

#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "impl/odescent_graph_parameter.h"

#include "ugraph_zparameters.h"
#include "algorithm/inner_index_interface.h"

namespace vsag {
class UGraph : public InnerIndexInterface  {
public:
    using Edge = std::tuple<float, InnerIdType, InnerIdType>;
    UGraph(const UgraphParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param) {

    }


    ~UGraph() override = default;

    [[nodiscard]] std::string
    GetName() const override {
        return "ugraph";
    }

    void
    InitFeatures() override {

    }

    std::vector<int64_t> Build(const DatasetPtr& base) override;

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
        int64_t limited_size) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    [[nodiscard]] int64_t
    GetNumElements() const override;

private:


private:
    IndexCommonParam common_param_;

    ODescentParameterPtr odescent_param_;

    GraphInterfacePtr graph_;
    FlattenInterfacePtr graph_flatten_codes_;




};


}




