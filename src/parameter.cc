// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "src/parameter.h"

#include <dirent.h>
#include <sys/stat.h>
#include <ctime>
#include <iomanip>
#include <vector>

namespace paddle {
namespace tape {

void ParameterCollection::InitParameter(ParameterHandle param,
                                        const std::string &initializer,
                                        const framework::AttributeMap &attrs) {
  if (initializer == "fill_constant") {
    // fill_constant is an Operator instead of OperatorWithKernel
    RunOperator(initializer, {}, {{"Out", {param}}}, attrs);
  } else {
    RunOperatorWithKernel(initializer, {}, {{"Out", {param}}}, attrs);
  }
}

ParameterHandle ParameterCollection::AddParameter(
    const std::string &name,
    const std::string &initializer,
    const framework::AttributeMap &attrs) {
  ParameterHandle param(new Parameter(name));
  InitParameter(param, initializer, attrs);
  params_[param->Name()] = param;

  return param;
}

std::vector<ParameterHandle> ParameterCollection::OptimizableParameters() {
  std::vector<ParameterHandle> rval;
  for (auto &pair : params_) {
    if (no_grad_name_.find(pair.first) == no_grad_name_.end()) {
      rval.emplace_back(pair.second);
    }
  }
  return rval;
}

constexpr char kTensorFileSuffix[] = ".pd";

void ParameterCollection::SaveAllParameters(std::string directory_name) {
  if (directory_name.empty()) {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
    directory_name = "/tmp/" + ss.str() + "/";
  }
  VLOG(3) << directory_name;
  PADDLE_ENFORCE_EQ(directory_name.back(), '/');
  PADDLE_ENFORCE_EQ(mkdir(directory_name.c_str(), 0775),
                    0,
                    "directory %s already exists",
                    directory_name);
  for (auto &pair : params_) {
    auto &param = pair.second;
    VLOG(10) << pair.first;
    VLOG(10) << param->Name();
    std::string file_path = directory_name + param->Name() + kTensorFileSuffix;
    RunOperator("save", {{"X", {param}}}, {}, {{"file_path", file_path}});
  }
}

// borrowed from
// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

ParameterCollection::ParameterCollection(std::string directory_name) {
  PADDLE_ENFORCE_EQ(directory_name.back(), '/');
  DIR *dir = opendir(directory_name.c_str());
  PADDLE_ENFORCE_NOT_NULL(dir);
  struct dirent *ent;
  while ((ent = readdir(dir)) != nullptr) {
    std::string filename = ent->d_name;
    if (ends_with(filename, kTensorFileSuffix)) {
      std::string param_name(
          filename, 0, filename.size() - strlen(kTensorFileSuffix));
      ParameterHandle param(new Parameter(param_name, Variable::Suffix::NONE));
      RunOperator("load",
                  {},
                  {{"Out", {param}}},
                  {{"file_path", directory_name + filename}});
      LOG(INFO) << filename;
      LOG(INFO) << param_name;
      LOG(INFO) << param->Name();
      // TODO(tonyyang-svail) batchnorm
      params_[param->Name()] = param;
    }
  }
}

ParameterCollection &GlobalParameterCollection() {
  static ParameterCollection pc;
  return pc;
}

std::vector<ParameterHandle> OptimizableParameters() {
  return GlobalParameterCollection().OptimizableParameters();
}

}  // namespace tape
}  // namespace paddle
