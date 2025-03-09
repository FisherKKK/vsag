
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

#include "pag_zparameters.h"

#include "inner_string_params.h"

namespace vsag {


static const std::string PAGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_PAGRAPH}",
        "PA_PARAM": {
        },
        "{PAGRAPH_GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_PARAM_MAX_DEGREE}": 32,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100
        },
        "{PGRAPH_GRAPH_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "use_quantization": false
            }
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_THREAD_COUNT}": 100,
            "sample_rate": 0.2,
            "start_decay_rate": 0.8,
            "capacity": 32,
            "num_iter": 2,
            "replicas": 4,
            "fine_radius_rate": 0.5,
            "coarse_radius_rate": 0.5,
        }
    })";

void
PAGraphParameter::FromJson(const JsonType& json) {
    // Build parameter
    const auto &build_param_json = json[BUILD_PARAMS_KEY];
    this->sample_rate_ = build_param_json["sample_rate"];
    this->start_decay_rate_ = build_param_json["start_decay_rate"];
    this->capacity_ = build_param_json["capacity"];
    this->num_iter_ = build_param_json["num_iter"];
    this->replicas_ = build_param_json["replicas"];
    this->fine_radius_rate_ = build_param_json["fine_radius_rate"];
    this->coarse_radius_rate_ = build_param_json["coarse_radius_rate"];

    // Graph flatten codes parameter
    const auto &graph_codes_json = json["base_codes"];
    this->graph_flatten_codes_param_ = std::make_shared<FlattenDataCellParameter>();
    this->graph_flatten_codes_param_->FromJson(graph_codes_json);

    // Graph parameter
    const auto &graph_json = json["graph"];
    this->graph_param_ = GraphInterfaceParameter::GetGraphParameterByJson(graph_json);

    // ODescent parameter
    const auto &odescent = json["odescent"];
    this->odescent_param_ = std::make_shared<ODescentParameter>();
    this->odescent_param_->FromJson(odescent);
}

JsonType
PAGraphParameter::ToJson() {
    JsonType json;
    json["type"] = "pagraph";
    json["num_iter"] = num_iter_;
    json["replicas"] = replicas_;
    json["capacity"] = capacity_;
    json["sample_rate"] = sample_rate_;
    json["start_decay_rate"] = start_decay_rate_;
    json["fine_radius_rate"] = fine_radius_rate_;
    json["coarse_radius_rate"] = coarse_radius_rate_;
    json["ef"] = ef_;

    json["graph"] = graph_param_->ToJson();
    json["odescent"] = odescent_param_->ToJson();
    json["base_codes"] = graph_flatten_codes_param_->ToJson();
}



}
