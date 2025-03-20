
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

#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>

#include "../eval/eval_dataset.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

using namespace nlohmann;
using namespace spdlog;
using namespace vsag;
using namespace vsag::eval;

std::unordered_set<int64_t>
get_intersection(const int64_t* neighbors,
                 const int64_t* ground_truth,
                 size_t recall_num,
                 size_t top_k) {
    std::unordered_set<int64_t> neighbors_set(neighbors, neighbors + recall_num);
    std::unordered_set<int64_t> intersection;
    for (size_t i = 0; i < top_k; ++i) {
        if (i < top_k && neighbors_set.count(ground_truth[i])) {
            intersection.insert(ground_truth[i]);
        }
    }
    return intersection;
}

int
main(int argc, char** argv) {
    vsag::init();



    // /******************* Prepare Base Dataset *****************/
    // int64_t num_vectors = 1000;
    // int64_t dim = 128;
    // std::vector<int64_t> ids(num_vectors);
    // std::vector<float> datas(num_vectors * dim);
    // std::mt19937 rng(47);
    // std::uniform_real_distribution<float> distrib_real;
    // for (int64_t i = 0; i < num_vectors; ++i) {
    //     ids[i] = i;
    // }
    // for (int64_t i = 0; i < dim * num_vectors; ++i) {
    //     datas[i] = distrib_real(rng);
    // }
    // auto base = vsag::Dataset::Make();
    // base->NumElements(num_vectors)
    //     ->Dim(dim)
    //     ->Ids(ids.data())
    //     ->Float32Vectors(datas.data())
    //     ->Owner(false);
    //
    // /******************* Create HGraph Index *****************/
    std::string pagraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 256,
        "pagraph": {}
    }
    )";
    // vsag::Engine engine;
    // auto index = engine.CreateIndex("pagraph", pagraph_build_parameters).value();
    //
    // /******************* Build PAGraph Index *****************/
    // if (auto build_result = index->Build(base); build_result.has_value()) {
    //     std::cout << "After Build(), Index PAGraph contains: " << index->GetNumElements()
    //               << std::endl;
    // } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
    //     std::cerr << "Failed to build index: internalError" << std::endl;
    //     exit(-1);
    // }
    //
    // /******************* Prepare Query Dataset *****************/
    // std::vector<float> query_vector(dim);
    // for (int64_t i = 0; i < dim; ++i) {
    //     query_vector[i] = distrib_real(rng);
    // }
    // auto query = vsag::Dataset::Make();
    // query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);
    //
    // /******************* KnnSearch For PAGraph Index *****************/
    auto pagraph_search_parameters = R"(
    {
        "pagraph": {
            "ef_search": 1000,
            "nprobe": 600
        }
    }
    )";
    // int64_t topk = 10;
    // auto result = index->KnnSearch(query, topk, pagraph_search_parameters).value();
    //
    // /******************* Print Search Result *****************/
    // std::cout << "results: " << std::endl;
    // for (int64_t i = 0; i < result->GetDim(); ++i) {
    //     std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    // }
    //
    /****************** Serialize and Deserialize PAGraph ****************/
    std::string dataset_path = "/data/dataset/security-256-ip.hdf5";
    std::filesystem::path dir("/data/index/test_pag_security256/");
        std::map<std::string, size_t> file_sizes;
        std::ifstream infile(dir / "pag_security256_meta.data");
        std::string filename;
        size_t size;
        while (infile >> filename >> size) {
            file_sizes[filename] = size;
        }
        infile.close();

        auto index = vsag::Factory::CreateIndex("pagraph", pagraph_build_parameters).value();
        vsag::ReaderSet reader_set;
        for (const auto& single_file : file_sizes) {
            const std::string& key = single_file.first;
            size = single_file.second;
            std::filesystem::path file_path(key);
            std::filesystem::path full_path = dir / file_path;
            auto reader = vsag::Factory::CreateLocalFileReader(full_path.string(), 0, size);
            reader_set.Set(key, reader);
        }

        index->Deserialize(reader_set);
        unsigned long long memoryUsage = 0;
        std::ifstream statFileAfter("/proc/self/status");
        if (statFileAfter.is_open()) {
            std::string line;
            while (std::getline(statFileAfter, line)) {
                if (line.substr(0, 6) == "VmRSS:") {
                    std::string value = line.substr(6);
                    memoryUsage = std::stoull(value) * 1024;
                    break;
                }
            }
            statFileAfter.close();
        }

        auto eval_dataset = EvalDataset::Load(dataset_path);

        // search
        auto search_start = std::chrono::steady_clock::now();
        int64_t correct = 0;
        int64_t total = eval_dataset->GetNumberOfQuery();
        spdlog::debug("total: " + std::to_string(total));
        std::vector<DatasetPtr> results;
        int i = 998;
            auto query = Dataset::Make();
            query->NumElements(1)->Dim(eval_dataset->GetDim())->Owner(false);

            if (eval_dataset->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                query->Float32Vectors((const float*)eval_dataset->GetOneTest(i));
            } else if (eval_dataset->GetTestDataType() == vsag::DATATYPE_INT8) {
                query->Int8Vectors((const int8_t*)eval_dataset->GetOneTest(i));
            }
            auto filter = [&eval_dataset, i](int64_t base_id) {
                return not eval_dataset->IsMatch(i, base_id);
            };

            auto result = index->KnnSearch(query, 100, pagraph_search_parameters, filter);

            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            results.emplace_back(result.value());

        auto search_finish = std::chrono::steady_clock::now();

        int top_k = 100;
        // calculate recall
            // k@k
            int64_t* neighbors = eval_dataset->GetNeighbors(i);
            const int64_t* ground_truth = results[i]->GetIds();
            auto hit_result = get_intersection(neighbors, ground_truth, top_k, top_k);
            correct += hit_result.size();

    return 0;
}
