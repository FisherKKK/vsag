
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

#include "ugraph_zparameters.h"

#include "inner_string_params.h"

namespace vsag {
void
UGraphParameter::FromJson(const JsonType& json) {
    // Build parameter
    const auto& build_param_json = json[BUILD_PARAMS_KEY];
    this->use_quantization_ = build_param_json["use_quantization"];

    // Graph flatten codes parameter
    const auto& graph_codes_json = json["base_codes"];
    this->graph_flatten_codes_param_ = std::make_shared<FlattenDataCellParameter>();
    this->graph_flatten_codes_param_->FromJson(graph_codes_json);

    // Graph parameter
    const auto& graph_json = json["graph"];
    this->graph_param_ = GraphInterfaceParameter::GetGraphParameterByJson(graph_json);

    // ODescent parameter
    const auto& odescent = json["odescent"];
    this->odescent_param_ = std::make_shared<ODescentParameter>();
    this->odescent_param_->FromJson(odescent);

    // Low precision
    if (this->use_quantization_) {
        low_precision_graph_flatten_codes_param_ = std::make_shared<FlattenDataCellParameter>();
        low_precision_graph_flatten_codes_param_->FromJson(json["quantization_codes"]);
    }

    this->bucket_file_ = build_param_json["bucket_file"];
}

JsonType
UGraphParameter::ToJson() {
    JsonType json;
    json["type"] = "pagraph";
    json["graph"] = graph_param_->ToJson();
    json["odescent"] = odescent_param_->ToJson();
    json["base_codes"] = graph_flatten_codes_param_->ToJson();
    return json;
}

}  // namespace vsag
