
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
#include <data_cell/flatten_datacell_parameter.h>
#include <data_cell/graph_interface_parameter.h>
#include <impl/odescent_graph_parameter.h>

#include <memory>

#include "parameter.h"

namespace vsag {
class UGraphParameter : public Parameter {
public:
    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    GraphInterfaceParamPtr graph_param_{nullptr};
    FlattenDataCellParamPtr graph_flatten_codes_param_{nullptr};
    ODescentParameterPtr odescent_param_{nullptr};
    bool use_quantization_{false};
    FlattenDataCellParamPtr low_precision_graph_flatten_codes_param_{nullptr};
    std::string bucket_file_{"/tmp/buckets.bin"};
};

using UgraphParameterPtr = std::shared_ptr<UGraphParameter>;
}  // namespace vsag