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

#include <memory>
#include <string>
#include <vector>

#include "src/tape.h"

namespace paddle {
namespace tape {

class Parameter;
using ParameterHandle = std::shared_ptr<Parameter>;

class Parameter : public Variable {
 public:
  explicit Parameter(std::string name) : Variable(name) {}
};

class ParameterCollection {
 public:
  ParameterHandle AddParameter(const std::string &name,
                               const std::string &initializer,
                               const framework::AttributeMap &attrs);

  // batch norm parameter are special since it is not updated by optimizer
  ParameterHandle AddBNParameter(const std::string &name,
                                 const std::string &initializer,
                                 const framework::AttributeMap &attrs);

  // All parameters excluding batch norm parameters
  std::vector<ParameterHandle> OptimizableParameters();

 private:
  void InitParameter(ParameterHandle param,
                     const std::string &initializer,
                     const framework::AttributeMap &attrs);

  std::vector<ParameterHandle> optimizable_params_;
  std::vector<ParameterHandle> batch_norm_params_;
};

ParameterCollection &GlobalParameterCollection();
std::vector<ParameterHandle> OptimizableParameters();

}  // namespace tape
}  // namespace paddle
