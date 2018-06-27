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
#include <utility>
#include <vector>

#include "src/variable.h"

namespace paddle {
namespace tape {

using VariableHandleMap = std::map<std::string, std::vector<VariableHandle>>;

framework::OpDesc CreateOpDesc(const std::string &type,
                               const VariableHandleMap &in_vars,
                               const VariableHandleMap &out_vars,
                               const framework::AttributeMap &attrs);

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

// Temporary Scope for Operator::Run()
class ScopeWrapper : public framework::Scope {
 public:
  ScopeWrapper(const VariableHandleMap &in_vars,
               const VariableHandleMap &out_vars) {
    for (auto &v : in_vars) {
      for (auto &vv : v.second) {
        if (!vars_.count(vv->Name())) {
          vars_[vv->Name()].reset(vv->MutableVar());
        }
      }
    }
    for (auto &v : out_vars) {
      for (auto &vv : v.second) {
        if (!vars_.count(vv->Name())) {
          vars_[vv->Name()].reset(vv->MutableVar());
        }
      }
    }
  }

  ~ScopeWrapper() {
    for (auto &pair : vars_) {
      pair.second.release();
    }
  }
};

class Tape {
 public:
  void AddOp(const std::string &type,
             const VariableHandleMap &in_vars,
             VariableHandleMap out_vars,
             const framework::AttributeMap &attrs);
  void Forward();
  void Backward(VariableHandle target);

  bool HasBeenBackwarded() { return has_been_backwarded_; }

  std::string GraphVizString(bool with_backward = true);

 private:
  /*
   * Construct vhm based on name2var, variable_name_map
   *
   * During the construction, record duplicated gradient and
   * uninitialzied gradient.
   */
  using std::vector;
  using std::pair;
  void DescMapToVarMap(
      const std::unordered_map<std::string, VariableHandle> &name2var,
      const framework::VariableNameMap &variable_name_map,
      VariableHandleMap *vhm,
      vector<pair<VariableHandle, VariableHandle>> *duplicated_grad,
      vector<pair<VariableHandle, VariableHandle>> *uninitialized_grad,
      bool is_output);

  bool has_been_backwarded_ = false;
  size_t current_position_ = 0;

  std::vector<OpHandle> ops_;
  std::shared_ptr<Tape> backward_tape_;
};

Tape &get_global_tape();

void reset_global_tape();
}  // namespace tape
}  // namespace paddle
