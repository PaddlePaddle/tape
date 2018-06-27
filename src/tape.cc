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

#include "src/tape.h"

#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace tape {

using std::map;
using std::pair;
using std::unordered_map;
using std::string;
using std::unique_ptr;
using std::vector;

// borrowed from
// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
inline bool ends_with(string const &value, string const &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

string to_string(const string &type,
                 const VariableHandleMap &in_vars,
                 const VariableHandleMap &out_vars,
                 const framework::AttributeMap &attrs) {
  std::stringstream ss;
  ss << type << " ";
  for (auto &param_name : in_vars) {
    for (auto &var : param_name.second) {
      ss << param_name.first << ":(" << var->Name() << ") ";
    }
  }
  for (auto &param_name : out_vars) {
    for (auto &var : param_name.second) {
      ss << param_name.first << ":(" << var->Name() << ") ";
    }
  }
  return ss.str();
}

framework::OpDesc CreateOpDesc(const string &type,
                               const VariableHandleMap &in_vars,
                               const VariableHandleMap &out_vars,
                               const framework::AttributeMap &attrs) {
  framework::VariableNameMap inputs;
  for (auto &param_name : in_vars) {
    for (auto &var : param_name.second) {
      inputs[param_name.first].emplace_back(var->Name());
    }
  }
  framework::VariableNameMap outputs;
  for (auto &param_name : out_vars) {
    for (auto &var : param_name.second) {
      outputs[param_name.first].emplace_back(var->Name());
    }
  }
  framework::OpDesc op_desc(type, inputs, outputs, attrs);
  op_desc.CheckAttrs();
  return op_desc;
}

void InferShapeAndVarType(const string &type,
                          const VariableHandleMap &in_vars,
                          VariableHandleMap *out_vars,
                          const framework::AttributeMap &attrs) {
  // Tape only supports LoDTensor
  for (auto &param2var : *out_vars) {
    for (auto &var : param2var.second) {
      var->GetMutable<framework::LoDTensor>();
    }
  }

  framework::OpDesc op_desc = CreateOpDesc(type, in_vars, *out_vars, attrs);
  ScopeWrapper scope(in_vars, *out_vars);

  // Tape only supports OperatorWithKernel
  auto op = framework::OpRegistry::CreateOp(op_desc);
  auto *op_with_kernel =
      dynamic_cast<framework::OperatorWithKernel *>(op.get());
  PADDLE_ENFORCE_NOT_NULL(op_with_kernel, "%s doesn't have kernel", type);
  paddle::framework::RuntimeInferShapeContext infer_shape_ctx(*op_with_kernel,
                                                              scope);
  op_with_kernel->InferShape(&infer_shape_ctx);
}

void Tape::AddOp(const string &type,
                 const VariableHandleMap &in_vars,
                 VariableHandleMap out_vars,
                 const framework::AttributeMap &attrs) {
  PADDLE_ENFORCE(!has_been_backwarded_);
  LOG(INFO) << "AddOp " << to_string(type, in_vars, out_vars, attrs);
  InferShapeAndVarType(type, in_vars, &out_vars, attrs);
  tape_.emplace_back(type, in_vars, out_vars, attrs);
}

void Tape::Forward() {
  VLOG(3) << "Starting forward -------------------------";
  while (current_position_ < tape_.size()) {
    PADDLE_ENFORCE(!has_been_backwarded_);
    OpHandle &op = tape_[current_position_];
    framework::OpDesc op_desc =
        CreateOpDesc(op.type_, op.inputs_, op.outputs_, op.attrs_);
    ScopeWrapper scope(op.inputs_, op.outputs_);
    framework::OpRegistry::CreateOp(op_desc)->Run(scope, platform::CPUPlace());
    current_position_++;
  }

  VLOG(3) << "Finishing forward -------------------------";
}

void Tape::DescMapToVarMap(
    const unordered_map<string, VariableHandle> &name2var,
    const framework::VariableNameMap &variable_name_map,
    VariableHandleMap *vhm,
    vector<pair<VariableHandle, VariableHandle>> *duplicated_grad,
    bool is_output) {
  for (auto &p2a : variable_name_map) {
    for (auto &arg : p2a.second) {
      auto &param = p2a.first;
      if (name2var.count(arg)) {
        (*vhm)[param].push_back(name2var.at(arg));
      } else {
        PADDLE_ENFORCE(
            ends_with(arg, framework::kGradVarSuffix),
            "Backward can only add gradient variable. %s not end with %s",
            arg,
            framework::kGradVarSuffix);
        string name =
            arg.substr(0, arg.size() - strlen(framework::kGradVarSuffix));
        PADDLE_ENFORCE(name2var.count(name), "%s not found", name);
        if (is_output &&
            name2var.at(name)->GradExist()) {  // Sum duplicated grad
          VariableHandle temp_grad(new Variable(
              name + framework::kGradVarSuffix + framework::kTempVarName));
          // we want sum duplicated grad to be in-place
          // since sum_op uses X[0] == Out to determine inplace
          // we assign name2var[name]->Grad to be the first element
          duplicated_grad->emplace_back(name2var.at(name)->Grad(), temp_grad);
          (*vhm)[param].emplace_back(temp_grad);
        } else if (!is_output &&
                   !name2var.at(name)
                        ->GradExist()) {  // zero initialize empty grad
          auto var = name2var.at(name);
          backward_tape_->AddOp(
              "fill_zeros_like", {{"X", {var}}}, {{"Out", {var->Grad()}}}, {});
          (*vhm)[param].push_back(var->Grad());
        } else {
          (*vhm)[param].push_back(name2var.at(name)->Grad());
        }
      }
    }
  }
}

void Tape::Backward(VariableHandle target) {
  PADDLE_ENFORCE(!has_been_backwarded_, "A tape can only backward once.");

  Forward();

  // TODO(tonyyang-svail): check output of last op is target
  backward_tape_.reset(new Tape());

  backward_tape_->AddOp(
      "fill_ones_like", {{"X", {target}}}, {{"Out", {target->Grad()}}}, {});

  for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
    framework::OpDesc op_desc =
        CreateOpDesc(it->type_, it->inputs_, it->outputs_, it->attrs_);
    unordered_map<string, string> grad_to_var;
    vector<unique_ptr<framework::OpDesc>> grad_op_descs =
        framework::OpInfoMap::Instance()
            .Get(op_desc.Type())
            .GradOpMaker()(op_desc, {}, &grad_to_var, {});

    for (auto &op_grad_desc : grad_op_descs) {
      unordered_map<string, VariableHandle> name2var;
      for (auto &param2vars : it->inputs_) {
        for (auto &a : param2vars.second) {
          name2var[a->Name()] = a;
        }
      }
      for (auto &param2vars : it->outputs_) {
        for (auto &a : param2vars.second) {
          name2var[a->Name()] = a;
        }
      }

      vector<pair<VariableHandle, VariableHandle>>
          duplicated_grad;  // {grad, grad@temp}
      VariableHandleMap in_vars, out_vars;
      DescMapToVarMap(
          name2var, op_grad_desc->Inputs(), &in_vars, &duplicated_grad, false);
      DescMapToVarMap(
          name2var, op_grad_desc->Outputs(), &out_vars, &duplicated_grad, true);

      backward_tape_->AddOp(
          op_grad_desc->Type(), in_vars, out_vars, op_grad_desc->GetAttrMap());
      for (auto &pair : duplicated_grad) {
        backward_tape_->AddOp("sum",
                              {{"X", {pair.first, pair.second}}},
                              {{"Out", {pair.first}}},
                              {});
      }
    }
  }

  backward_tape_->Forward();
  has_been_backwarded_ = true;
}

Tape &get_global_tape() {
  static Tape T;
  return T;
}

void reset_global_tape() { get_global_tape() = Tape(); }
}  // namespace tape
}  // namespace paddle
