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

#include <cmath>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/type_defs.h"
#include "src/tape.h"
#include "src/variable.h"

namespace paddle {
namespace tape {

class Function {};

class RandomSeed {
 public:
  static int GetRandomSeed() { return 0; }
};

class Fill {
 public:
  Fill(const std::string &initializer, const framework::AttributeMap &attrs)
      : initializer_(initializer), attrs_(attrs) {}

  VariableHandle operator()() {
    if (initializer_ == "fill_constant") {
      PADDLE_THROW(
          "fill_constant is not supported, since it is not of type "
          "OperatorWithKernel");
    }

    VariableHandle var(new Variable(initializer_));
    get_global_tape().AddOp(initializer_, {}, {{"Out", {var}}}, attrs_);
    return var;
  }

 private:
  const std::string initializer_;
  const framework::AttributeMap attrs_;
};

void init_params(VariableHandle v,
                 const std::string &initializer,
                 const framework::AttributeMap &attrs) {
  if (initializer == "fill_constant") {
    // fill_constant is not OperatorWithKernel, so we can't add it to the tape
    framework::OpDesc op_desc =
        CreateOpDesc(initializer, {}, {{"Out", {v}}}, attrs);
    ScopeWrapper scope({}, {{"Out", {v}}});
    framework::OpRegistry::CreateOp(op_desc)->Run(scope, platform::CPUPlace());
  } else {
    Tape init_tape;
    init_tape.AddOp(initializer, {}, {{"Out", {v}}}, attrs);
    init_tape.Forward();
  }
}

class Linear {
 public:
  Linear(int in_dim, int out_dim, const std::string &act)
      : w_(new Variable("LinearWeight")),
        b_(new Variable("LinearBias")),
        act_(act) {
    // Use Xavier to initialize Weight
    float limit = sqrt(6.0 / static_cast<float>(in_dim + out_dim));
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{in_dim, out_dim};
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["min"] = -limit;
    attrs["max"] = limit;
    attrs["seed"] = RandomSeed::GetRandomSeed();
    init_params(w_, "uniform_random", attrs);

    // Use fill zero to initialize Bias
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{out_dim};
    attrs["value"] = 0.0f;
    init_params(b_, "fill_constant", attrs);
  }

  VariableHandle operator()(VariableHandle input,
                            const framework::AttributeMap &mul_op_attrs = {},
                            const framework::AttributeMap &add_op_attrs = {}) {
    VariableHandle pre_bias(new Variable("linear"));
    get_global_tape().AddOp("mul",
                            {{"X", {input}}, {"Y", {w_}}},
                            {{"Out", {pre_bias}}},
                            mul_op_attrs);
    VariableHandle pre_act(new Variable("linear"));
    get_global_tape().AddOp("elementwise_add",
                            {{"X", {pre_bias}}, {"Y", {b_}}},
                            {{"Out", {pre_act}}},
                            add_op_attrs);
    VariableHandle post_act(new Variable("linear"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<VariableHandle> Params() { return {w_, b_}; }

 private:
  VariableHandle w_;
  VariableHandle b_;
  std::string act_;
};

class Convolution2D {
 public:
  Convolution2D(int c_in, int c_out, int f, const std::string &act)
      : w_(new Variable("ConvolutionWeight")),
        b_(new Variable("ConvolutionBias")),
        act_(act) {
    // Use Xavier to initialize Weight
    float fan_in = c_in * f * f, fan_out = c_out * f * f;
    float limit = sqrt(6.0 / (fan_in + fan_out));
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{c_out, c_in, f, f};
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["min"] = -limit;
    attrs["max"] = limit;
    attrs["seed"] = RandomSeed::GetRandomSeed();
    init_params(w_, "uniform_random", attrs);

    // Use fill zero to initialize Bias
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{c_out};
    attrs["value"] = 0.0f;
    init_params(b_, "fill_constant", attrs);
  }

  VariableHandle operator()(
      VariableHandle input,
      const framework::AttributeMap &conv_op_attrs = {},
      const framework::AttributeMap &add_op_attrs = {{"axis", 1}}) {
    VariableHandle pre_bias(new Variable("conv"));
    get_global_tape().AddOp("conv2d",
                            {{"Input", {input}}, {"Filter", {w_}}},
                            {{"Output", {pre_bias}}},
                            conv_op_attrs);
    VariableHandle pre_act(new Variable("conv"));
    get_global_tape().AddOp("elementwise_add",
                            {{"X", {pre_bias}}, {"Y", {b_}}},
                            {{"Out", {pre_act}}},
                            add_op_attrs);
    VariableHandle post_act(new Variable("conv"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<VariableHandle> Params() { return {w_, b_}; }

 private:
  VariableHandle w_;
  VariableHandle b_;
  std::string act_;
};

class SGD {
 public:
  explicit SGD(float learning_rate) : learning_rate_(new Variable("sgd")) {
    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{1};
    attrs["value"] = learning_rate;
    init_params(learning_rate_, initializer, attrs);
  }

  void Update(VariableHandle input) {
    PADDLE_ENFORCE(get_global_tape().HasBeenBackwarded(),
                   "optimization must happen after the backward");
    Tape temp_tape;
    temp_tape.AddOp("sgd",
                    {{"Param", {input}},
                     {"LearningRate", {learning_rate_}},
                     {"Grad", {input->Grad()}}},
                    {{"ParamOut", {input}}},
                    {});
    temp_tape.Forward();
  }

 private:
  VariableHandle learning_rate_;
};

class BatchNorm {
 public:
  BatchNorm(int channel_in, const std::string &act)
      : scale_(new Variable("BatchNormScale")),
        bias_(new Variable("BatchNormBias")),
        mean_(new Variable("BatchNormMean")),
        variance_(new Variable("BatchNormVariance")),
        act_(act) {
    // Use fill one to initialize scale and variance
    framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{channel_in};
    attrs["value"] = 1.0f;
    init_params(scale_, "fill_constant", attrs);
    init_params(variance_, "fill_constant", attrs);

    // Use fill zero to initialize bias and mean
    attrs["value"] = 0.0f;
    init_params(bias_, "fill_constant", attrs);
    init_params(mean_, "fill_constant", attrs);
  }

  VariableHandle operator()(VariableHandle x,
                            const framework::AttributeMap &attrs = {}) {
    VariableHandle pre_act(new Variable("batch_norm"));
    VariableHandle tmp_mean(new Variable("tmp_mean"));
    VariableHandle tmp_var(new Variable("tmp_var"));
    get_global_tape().AddOp("batch_norm",
                            {{"X", {x}},
                             {"Scale", {scale_}},
                             {"Bias", {bias_}},
                             {"Mean", {mean_}},
                             {"Variance", {variance_}}},
                            {{"Y", {pre_act}},
                             {"MeanOut", {mean_}},
                             {"VarianceOut", {variance_}},
                             {"SavedMean", {tmp_mean}},
                             {"SavedVariance", {tmp_var}}},
                            attrs);

    VariableHandle post_act(new Variable("batch_norm"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  // Only scale and bias need to be updated by SGD
  std::vector<VariableHandle> Params() { return {scale_, bias_}; }

 private:
  VariableHandle scale_;
  VariableHandle bias_;
  VariableHandle mean_;
  VariableHandle variance_;
  std::string act_;
};

VariableHandle pool2d(VariableHandle x,
                      const framework::AttributeMap &attrs = {}) {
  VariableHandle out(new Variable("pool2d"));
  get_global_tape().AddOp("pool2d", {{"X", {x}}}, {{"Out", {out}}}, attrs);
  return out;
}

VariableHandle dropout(VariableHandle x,
                       const framework::AttributeMap &attrs = {}) {
  VariableHandle out(new Variable("dropout"));
  VariableHandle mask(new Variable("mask"));
  get_global_tape().AddOp(
      "dropout", {{"X", {x}}}, {{"Out", {out}}, {"Mask", {mask}}}, attrs);
  return out;
}

VariableHandle mean(VariableHandle x) {
  VariableHandle out(new Variable("mean"));
  get_global_tape().AddOp("mean", {{"X", {x}}}, {{"Out", {out}}}, {});
  return out;
}

VariableHandle relu(VariableHandle x) {
  VariableHandle out(new Variable("relu"));
  get_global_tape().AddOp("relu", {{"X", {x}}}, {{"Out", {out}}}, {});
  return out;
}

VariableHandle softmax(VariableHandle x) {
  VariableHandle out(new Variable("softmax"));
  get_global_tape().AddOp("softmax", {{"X", {x}}}, {{"Out", {out}}}, {});
  return out;
}

VariableHandle cross_entropy(VariableHandle x, VariableHandle label) {
  VariableHandle out(new Variable("cross_entropy"));
  get_global_tape().AddOp(
      "cross_entropy", {{"X", {x}}, {"Label", {label}}}, {{"Y", {out}}}, {});
  return out;
}

VariableHandle CreateRecordioFileReader(std::string filename,
                                        std::vector<int> shape_concat,
                                        std::vector<int> ranks,
                                        std::vector<int> lod_levels) {
  VariableHandle reader(new paddle::tape::Variable("reader"));

  framework::OpDesc op_desc = CreateOpDesc("create_recordio_file_reader",
                                           {},
                                           {{"Out", {reader}}},
                                           {{"filename", filename},
                                            {"shape_concat", shape_concat},
                                            {"ranks", ranks},
                                            {"lod_levels", lod_levels}});
  ScopeWrapper scope({}, {{"Out", {reader}}});
  framework::OpRegistry::CreateOp(op_desc)->Run(scope, platform::CPUPlace());

  return reader;
}

std::vector<VariableHandle> ReadNext(VariableHandle reader) {
  PADDLE_ENFORCE(reader->Var().IsType<framework::ReaderHolder>());

  paddle::framework::LoDTensorArray data_holder;
  reader->GetMutable<paddle::framework::ReaderHolder>()->ReadNext(&data_holder);
  if (data_holder.empty()) {
    reader->GetMutable<paddle::framework::ReaderHolder>()->ReInit();
    reader->GetMutable<paddle::framework::ReaderHolder>()->ReadNext(
        &data_holder);
  }
  PADDLE_ENFORCE(!data_holder.empty(), "Error reading file.");

  std::vector<VariableHandle> rval;
  for (size_t i = 0; i < data_holder.size(); ++i) {
    rval.emplace_back(new Variable("data" + std::to_string(i)));
    auto *lod_tensor = rval.back()->GetMutable<framework::LoDTensor>();
    lod_tensor->ShareDataWith(data_holder[i]);
    lod_tensor->set_lod(data_holder[i].lod());
  }

  return rval;
}

}  // namespace tape
}  // namespace paddle
