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

#include <string>
#include <vector>

#include "src/parameter.h"
#include "src/tape.h"

namespace paddle {
namespace tape {

class Optimizer {
 public:
  virtual void Step() = 0;

  virtual void Update(ParameterHandle param) = 0;

  virtual ~Optimizer() {}
};

class SGD : public Optimizer {
 public:
  explicit SGD(float learning_rate);

  void Step() override;

  void Update(ParameterHandle) override;

 private:
  VariableHandle learning_rate_;
};

class Adam : public Optimizer {
 public:
  explicit Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f);

  void Step() override;

  void Update(ParameterHandle param) override;

 private:
  VariableHandle learning_rate_;
  VariableHandle beta1_pow_var_;
  VariableHandle beta2_pow_var_;
  float beta1_;
  float beta2_;
  float beta1_pow_{1.0f};
  float beta2_pow_{1.0f};
};

void BackwardAndUpdate(VariableHandle target, Optimizer *optimizer);

}  // namespace tape
}  // namespace paddle
