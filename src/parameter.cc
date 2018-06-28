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
  optimizable_params_.emplace_back(param);

  return param;
}

ParameterHandle ParameterCollection::AddBNParameter(
    const std::string &name,
    const std::string &initializer,
    const framework::AttributeMap &attrs) {
  ParameterHandle param(new Parameter(name));
  InitParameter(param, initializer, attrs);
  batch_norm_params_.emplace_back(param);

  return param;
}

std::vector<ParameterHandle> ParameterCollection::OptimizableParameters() {
  return optimizable_params_;
}

ParameterCollection &ParameterCollectionInstance() {
  static ParameterCollection pc;
  return pc;
}

std::vector<ParameterHandle> OptimizableParameters() {
  return ParameterCollectionInstance().OptimizableParameters();
}

}  // namespace tape
}  // namespace paddle
