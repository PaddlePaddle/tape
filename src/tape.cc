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

// borrowed from
// https://stackoverflow.com/questions/874134/find-if-string-ends-with-another-string-in-c
inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::ostream &operator<<(std::ostream &os, const framework::VarDesc &var_desc) {
  os << var_desc.Name();
  os << "[" << var_desc.GetType() << "]";
  os << "[" << var_desc.GetDataType() << "]";
  os << "{";
  for (auto &i : var_desc.GetShape()) {
    os << i << ",";
  }
  os << "}";
  return os;
}

std::string to_string(const std::string &type,
                      const VariableHandleMap &in_vars,
                      const VariableHandleMap &out_vars,
                      const framework::AttributeMap &attrs) {
  std::stringstream ss;
  ss << type << " ";
  for (auto &param_name : in_vars) {
    for (auto &var : param_name.second) {
      ss << param_name.first << ":(" << var << ") ";
    }
  }
  for (auto &param_name : out_vars) {
    for (auto &var : param_name.second) {
      ss << param_name.first << ":(" << var << ") ";
    }
  }
  return ss.str();
}

framework::OpDesc CreateOpDesc(const std::string &type,
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

void InferShapeAndVarType(const std::string &type,
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

void Tape::AddOp(const std::string &type,
                 const VariableHandleMap &in_vars,
                 VariableHandleMap out_vars,
                 const framework::AttributeMap &attrs) {
  InferShapeAndVarType(type, in_vars, &out_vars, attrs);
  tape_.emplace_back(type, in_vars, out_vars, attrs);
  Forward();
}

void Tape::Forward() {
  VLOG(3) << "Starting forward -------------------------";
  PADDLE_ENFORCE(!has_been_backwarded_);
  while (current_position_ < tape_.size()) {
    OpHandle &op = tape_[current_position_];
    framework::OpDesc op_desc =
        CreateOpDesc(op.type_, op.inputs_, op.outputs_, op.attrs_);
    ScopeWrapper scope(op.inputs_, op.outputs_);
    framework::OpRegistry::CreateOp(op_desc)->Run(scope, platform::CPUPlace());
    current_position_++;
  }

  VLOG(3) << "Finishing forward -------------------------";
}

void Tape::Backward(VariableHandle target) {
  PADDLE_ENFORCE(!has_been_backwarded_);

  Forward();

  // TODO(tonyyang-svail): check output of last op is target
  backward_tape_.reset(new Tape());

  // FIXME(tonyyang-svail): Need to infer_data_type
  backward_tape_->AddOp(
      "fill_ones_like", {{"X", {target}}}, {{"Out", {target->Grad()}}}, {});

  for (auto it = tape_.rbegin(); it != tape_.rend(); ++it) {
    framework::OpDesc op_desc =
        CreateOpDesc(it->type_, it->inputs_, it->outputs_, it->attrs_);
    std::unordered_map<std::string, std::string> grad_to_var;
    std::vector<std::unique_ptr<framework::OpDesc>> grad_op_descs =
        framework::OpInfoMap::Instance()
            .Get(op_desc.Type())
            .GradOpMaker()(op_desc, {}, &grad_to_var, {});

    for (auto &op_desc : grad_op_descs) {
      std::unordered_map<std::string, VariableHandle> name2var;
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

      VariableHandleMap in_vars;
      VariableHandleMap out_vars;
      std::map<const framework::VariableNameMap *, VariableHandleMap *>
          loop_over{{&op_desc->Inputs(), &in_vars},
                    {&op_desc->Outputs(), &out_vars}};
      for (auto &each : loop_over) {
        auto &vmp = *each.first;
        auto &vhm = *each.second;
        for (auto &p2a : vmp) {
          for (auto &argu : p2a.second) {
            if (name2var.count(argu)) {
              vhm[p2a.first].push_back(name2var[argu]);
            } else {
              PADDLE_ENFORCE(ends_with(argu, framework::kGradVarSuffix),
                             argu.c_str());
              std::string name = argu.substr(
                  0, argu.size() - std::strlen(framework::kGradVarSuffix));
              PADDLE_ENFORCE(name2var.count(name), name.c_str());
              vhm[p2a.first].push_back(name2var[name]->Grad());
            }
          }
        }
      }

      backward_tape_->AddOp(
          op_desc->Type(), in_vars, out_vars, op_desc->GetAttrMap());
    }

    // TODO(tonyyang-svail): how to fill empty grad?
    // TODO(tonyyang-svail): Sum var grad is necessary
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
