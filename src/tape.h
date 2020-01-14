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
  explicit Variable(const std::string &pre_fix);

  enum Suffix { COUNT, GRAD, NONE };
  Variable(const std::string &pre_fix, Suffix suffix);

  ~Variable() { VLOG(10) << "Deleting " << Name(); }

  bool GradExist() { return !grad_.expired(); }

  // Return the gradient of a Variable
  VariableHandle Grad();

  // Evaluate a variable by running Forward() on the global tape
  const Variable &Value();

  // Evaluate and make a copy of Variable data on CPU
  VariableHandle FetchValue();

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

 private:
  std::string name_;
  framework::Variable var_;

  // Not own
  std::weak_ptr<Variable> grad_;
};

struct OpRecord {
  OpRecord(const std::string &type,
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

  std::string GraphVizString(bool print_backward = true);

  const platform::Place &Place() { return place_; }

 private:
  bool has_been_backwarded_ = false;
  size_t current_position_ = 0;
  platform::Place place_;

  std::vector<OpRecord> ops_;
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
