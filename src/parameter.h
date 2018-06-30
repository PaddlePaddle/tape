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
#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "src/tape.h"

namespace paddle {
namespace tape {

class Parameter;
using ParameterHandle = std::shared_ptr<Parameter>;

class Parameter : public Variable {
 public:
  explicit Parameter(const std::string &name,
                     Variable::Suffix suffix = Variable::Suffix::COUNT)
      : Variable(name, suffix) {}
};

class ParameterCollection {
 public:
  ParameterCollection() {}

  // Load Parameter from a directory
  explicit ParameterCollection(std::string directory_name);

  ParameterHandle AddParameter(const std::string &name,
                               const std::string &initializer,
                               const framework::AttributeMap &attrs);

  void MarkNoGrad(const std::string &name) { no_grad_name_.insert(name); }

  // All parameters excluding no_grad_name_
  std::vector<ParameterHandle> OptimizableParameters();

  void SaveAllParameters(std::string directory_name = "");

 private:
  void InitParameter(ParameterHandle param,
                     const std::string &initializer,
                     const framework::AttributeMap &attrs);

  std::set<std::string> no_grad_name_;
  std::map<std::string, ParameterHandle> params_;
};

ParameterCollection &GlobalParameterCollection();
std::vector<ParameterHandle> OptimizableParameters();

}  // namespace tape
}  // namespace paddle
