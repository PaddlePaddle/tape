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
  explicit SGD(float learning_rate) : learning_rate_(new Variable("sgd")) {
    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = learning_rate;
    RunOperator("fill_constant", {}, {{"Out", {learning_rate_}}}, attrs);
  }

  void Step() override {
    for (auto &w : OptimizableParameters()) {
      LOG(INFO) << "updating " << w->Name();
      Update(w);
    }
  }

  void Update(ParameterHandle param) override {
    PADDLE_ENFORCE(get_global_tape().HasBeenBackwarded(),
                   "optimization must happen after the backward");
    RunOperatorWithKernel("sgd",
                          {{"Param", {param}},
                           {"LearningRate", {learning_rate_}},
                           {"Grad", {param->Grad()}}},
                          {{"ParamOut", {param}}},
                          {});
  }

 private:
  VariableHandle learning_rate_;
};

class Adam : public Optimizer {
 public:
  explicit Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f)
      : learning_rate_(new Variable("adam")),
        beta1_pow_var_(new Variable("adam")),
        beta2_pow_var_(new Variable("adam")),
        beta1_(beta1),
        beta2_(beta2) {
    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = learning_rate;
    RunOperator("fill_constant", {}, {{"Out", {learning_rate_}}}, attrs);
  }

  void Step() override {
    // Update optimizer parameters
    beta1_pow_ *= beta1_;
    beta2_pow_ *= beta2_;
    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = beta1_pow_;
    RunOperator("fill_constant", {}, {{"Out", {beta1_pow_var_}}}, attrs);
    attrs["value"] = beta2_pow_;
    RunOperator("fill_constant", {}, {{"Out", {beta2_pow_var_}}}, attrs);
    // Update model parameters
    for (auto &w : OptimizableParameters()) {
      LOG(INFO) << "updating " << w->Name();
      Update(w);
    }
  }

  void Update(ParameterHandle param) override {
    PADDLE_ENFORCE(get_global_tape().HasBeenBackwarded(),
                   "optimization must happen after the backward");
    auto *hyperparams = param->MutableHyperParams("adam");
    // initialize states if they haven't been created
    if (hyperparams->empty()) {
      framework::AttributeMap attrs;
      attrs["shape"] = paddle::framework::vectorize2int(
          param->Get<paddle::framework::LoDTensor>().dims());
      attrs["value"] = 0.0f;
      ParameterHandle moment1(new Parameter("adam"));
      ParameterHandle moment2(new Parameter("adam"));
      RunOperator("fill_constant", {}, {{"Out", {moment1}}}, attrs);
      RunOperator("fill_constant", {}, {{"Out", {moment2}}}, attrs);

      hyperparams->emplace_back(moment1);
      hyperparams->emplace_back(moment2);
    }

    PADDLE_ENFORCE_EQ(
        hyperparams->size(), 2, "Adam should have two hyperparameters");
    auto moment1 = hyperparams->at(0);
    auto moment2 = hyperparams->at(1);

    RunOperatorWithKernel("adam",
                          {{"Param", {param}},
                           {"LearningRate", {learning_rate_}},
                           {"Grad", {param->Grad()}},
                           {"Moment1", {moment1}},
                           {"Moment2", {moment2}},
                           {"Beta1Pow", {beta1_pow_var_}},
                           {"Beta2Pow", {beta2_pow_var_}}},
                          {{"ParamOut", {param}},
                           {"Moment1Out", {moment1}},
                           {"Moment2Out", {moment2}}},
                          {{"beta1", beta1_}, {"beta2", beta2_}});
  }

 private:
  VariableHandle learning_rate_;
  VariableHandle beta1_pow_var_;
  VariableHandle beta2_pow_var_;
  float beta1_;
  float beta2_;
  float beta1_pow_{1.0f};
  float beta2_pow_{1.0f};
};

void BackwardAndUpdate(VariableHandle target, Optimizer *optimizer) {
  get_global_tape().Backward(target);
  optimizer->Step();
}

}  // namespace tape
}  // namespace paddle
