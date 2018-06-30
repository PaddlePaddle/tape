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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/operator.h"  // framework::kGradVarSuffix
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace tape {

class Variable;
using VariableHandle = std::shared_ptr<Variable>;
using VariableHandleMap = std::map<std::string, std::vector<VariableHandle>>;

std::ostream &operator<<(std::ostream &, const Variable &);

class Variable {
 public:
  explicit Variable(const std::string pre_fix)
      : name_(pre_fix + std::to_string(count())) {}

  Variable(const std::string pre_fix, bool is_grad)
      : name_(pre_fix + (is_grad ? framework::kGradVarSuffix
                                 : std::to_string(count()))) {}

  ~Variable() { VLOG(10) << "Deleting " << Name(); }

  bool GradExist() { return !grad_.expired(); }

  VariableHandle Grad() {
    if (grad_.expired()) {
      VariableHandle new_grad(new Variable(name_, true));
      grad_ = new_grad;
      return new_grad;
    } else {
      return VariableHandle(grad_);
    }
  }

  // Evaluate a variable by running Forward() on the global tape
  const Variable &Value();

  // Evaluate and make a copy of Variable data on CPU
  VariableHandle FetchValue();

  // TODO(tonyyang-svail): No need to expose name
  std::string Name() const { return name_; }

  const framework::Variable &Var() const { return var_; }
  framework::Variable *MutableVar() { return &var_; }

  template <typename T>
  const T &Get() const {
    return var_.Get<T>();
  }

  template <typename T>
  T *GetMutable() {
    return var_.GetMutable<T>();
  }

  // TODO(tonyyang-svail): move MutableHyperParams to parameter.h
  std::vector<VariableHandle> *MutableHyperParams(
      const std::string &optimizer) {
    PADDLE_ENFORCE(hyperparams_.find(optimizer) != hyperparams_.end(),
                   "%s optimizer is not supported",
                   optimizer);
    return &hyperparams_[optimizer];
  }

 private:
  int64_t count() {
    static int64_t counter = 0;
    return counter++;
  }

  std::string name_;
  framework::Variable var_;

  // Not own
  std::weak_ptr<Variable> grad_;

  // Optimizer hyperparameters
  // TODO(tonyyang-svail): move hyperparameters to parameter.h
  std::unordered_map<std::string, std::vector<VariableHandle>> hyperparams_{
      {"adam", {}}};
};

struct OpHandle {
  OpHandle(const std::string &type,
           const VariableHandleMap &in_vars,
           const VariableHandleMap &out_vars,
           const framework::AttributeMap &attrs)
      : type_(type), inputs_(in_vars), outputs_(out_vars), attrs_(attrs) {}

  std::string type_;
  VariableHandleMap inputs_;
  VariableHandleMap outputs_;
  framework::AttributeMap attrs_;
};

class Tape {
 public:
  Tape() : place_(platform::CPUPlace()) {}
  explicit Tape(const platform::Place &place) : place_(place) {}

  void AddOp(const std::string &type,
             const VariableHandleMap &in_vars,
             const VariableHandleMap &out_vars,
             const framework::AttributeMap &attrs);
  void Forward();
  void Backward(VariableHandle target);

  bool HasBeenBackwarded() { return has_been_backwarded_; }

  std::string GraphVizString(bool with_backward = true);

  const platform::Place &Place() { return place_; }

 private:
  /*
   * Only used in backward
   *
   * Construct vhm based on name2var, variable_name_map
   *
   * During the construction, record duplicated gradient and
   * uninitialzied gradient.
   */
  void DescMapToVarMap(
      const std::unordered_map<std::string, VariableHandle> &name2var,
      const framework::VariableNameMap &variable_name_map,
      VariableHandleMap *vhm,
      std::vector<std::pair<VariableHandle, VariableHandle>> *dup_grad,
      std::vector<std::pair<VariableHandle, VariableHandle>> *init_grad,
      bool is_output);

  bool has_been_backwarded_ = false;
  size_t current_position_ = 0;
  platform::Place place_;

  std::vector<OpHandle> ops_;
  std::shared_ptr<Tape> backward_tape_;
};

void RunOperator(const std::string &type,
                 const VariableHandleMap &in_vars,
                 const VariableHandleMap &out_vars,
                 const framework::AttributeMap &attrs);
void RunOperatorWithKernel(const std::string &type,
                           const VariableHandleMap &in_vars,
                           const VariableHandleMap &out_vars,
                           const framework::AttributeMap &attrs);

Tape &get_global_tape();

void reset_global_tape(const platform::Place &place = platform::CPUPlace());
}  // namespace tape
}  // namespace paddle
