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

#include "paddle/fluid/framework/operator.h"  // framework::kGradVarSuffix
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace tape {

class Variable;
using VariableHandle = std::shared_ptr<Variable>;

std::ostream& operator<<(std::ostream&, const Variable&);

/*
 * Combination of
 *     framework::VarDesc desc_;
 *     framework::Variable var_;
 */
class Variable {
 public:
  explicit Variable(const std::string pre_fix)
      : name_(pre_fix + std::to_string(count())) {}

  Variable(const std::string pre_fix, bool is_grad)
      : name_(pre_fix + (is_grad ? framework::kGradVarSuffix
                                 : std::to_string(count()))) {}

  ~Variable() { VLOG(10) << "Deleting " << Name(); }

  VariableHandle Grad() {
    if (grad_.expired()) {
      VariableHandle new_grad(new Variable(name_, true));
      grad_ = new_grad;
      return new_grad;
    } else {
      return VariableHandle(grad_);
    }
  }

  bool GradExist() { return !grad_.expired(); }

  // Evaluate a variable by running Forward() on the global tape
  const Variable& Value();

  // TODO(tonyyang-svail): No need to expose name
  std::string Name() const { return name_; }

  const framework::Variable& Var() const { return var_; }
  framework::Variable* MutableVar() { return &var_; }

  template <typename T>
  const T& Get() const {
    return var_.Get<T>();
  }

  template <typename T>
  T* GetMutable() {
    return var_.GetMutable<T>();
  }

  std::vector<VariableHandle>* MutableHyperParams(
      const std::string& optimizer) {
    PADDLE_ENFORCE(hyperparams_.find(optimizer) != hyperparams_.end(),
                   "%s optimizer is not supported",
                   optimizer);
    return &hyperparams_[optimizer];
  }

 private:
  int count() {
    static int counter = 0;
    return counter++;
  }

  std::string name_;
  framework::Variable var_;

  // Not own
  std::weak_ptr<Variable> grad_;

  // Optimizer hyperparameters
  std::unordered_map<std::string, std::vector<VariableHandle>> hyperparams_{
      {"adam", {}}};
};

}  // namespace tape
}  // namespace paddle
