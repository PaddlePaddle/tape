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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/place.h"

#include "src/parameter.h"
#include "src/tape.h"

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

class Linear {
 public:
  Linear(std::vector<int> in_dims, int out_dim, const std::string &act = "")
      : act_(act) {
    // Use Xavier to initialize Weight
    framework::AttributeMap attrs;
    attrs["dtype"] = framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["seed"] = RandomSeed::GetRandomSeed();
    for (int in_dim : in_dims) {
      float limit = sqrt(6.0 / static_cast<float>(in_dim + out_dim));
      attrs["min"] = -limit;
      attrs["max"] = limit;
      attrs["shape"] = std::vector<int>{in_dim, out_dim};
      weights_.emplace_back(GlobalParameterCollection().AddParameter(
          "LinearWeight", "uniform_random", attrs));
    }

    // Use fill zero to initialize Bias
    attrs["dtype"] = framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{out_dim};
    attrs["value"] = 0.0f;
    b_ = GlobalParameterCollection().AddParameter(
        "LinearBias", "fill_constant", attrs);
  }

  Linear(const std::vector<ParameterHandle> &params, const std::string &act)
      : act_(act), b_(params.back()) {
    for (int i = 0; i < params.size() - 1; ++i) {
      weights_.emplace_back(params[i]);
    }
  }

  VariableHandle operator()(std::vector<VariableHandle> inputs,
                            const framework::AttributeMap &mul_op_attrs = {},
                            const framework::AttributeMap &add_op_attrs = {}) {
    PADDLE_ENFORCE_EQ(inputs.size(), weights_.size());
    std::vector<VariableHandle> sums;
    for (int i = 0; i < inputs.size(); ++i) {
      VariableHandle pre_bias(new Variable("linear"));
      get_global_tape().AddOp("mul",
                              {{"X", {inputs[i]}}, {"Y", {weights_[i]}}},
                              {{"Out", {pre_bias}}},
                              mul_op_attrs);
      sums.emplace_back(pre_bias);
    }

    VariableHandle pre_bias(new Variable("linear"));
    if (inputs.size() > 1) {
      get_global_tape().AddOp("sum", {{"X", sums}}, {{"Out", {pre_bias}}}, {});
    } else {
      pre_bias = sums[0];
    }

    VariableHandle pre_act(new Variable("linear"));
    get_global_tape().AddOp("elementwise_add",
                            {{"X", {pre_bias}}, {"Y", {b_}}},
                            {{"Out", {pre_act}}},
                            add_op_attrs);
    if (act_.empty()) {
      return pre_act;
    }
    VariableHandle post_act(new Variable("linear"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<std::string> ParamNames() {
    std::vector<std::string> res;
    for (auto w : weights_) {
      res.emplace_back(w->Name());
    }
    res.emplace_back(b_->Name());
    return res;
  }

  std::string ActName() { return act_; }

 private:
  std::vector<ParameterHandle> weights_;
  ParameterHandle b_;
  std::string act_;
};

class Convolution2D {
 public:
  Convolution2D(int c_in, int c_out, int f_size, const std::string &act = "")
      : act_(act) {
    // Use Xavier to initialize Weight
    float fan_in = c_in * f_size * f_size, fan_out = c_out * f_size * f_size;
    float limit = sqrt(6.0 / (fan_in + fan_out));
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{c_out, c_in, f_size, f_size};
    attrs["min"] = -limit;
    attrs["max"] = limit;
    attrs["seed"] = RandomSeed::GetRandomSeed();
    w_ = GlobalParameterCollection().AddParameter(
        "ConvolutionWeight", "uniform_random", attrs);

    // Use fill zero to initialize Bias
    attrs["shape"] = std::vector<int>{c_out};
    attrs["value"] = 0.0f;
    b_ = GlobalParameterCollection().AddParameter(
        "ConvolutionBias", "fill_constant", attrs);
  }

  Convolution2D(const std::vector<ParameterHandle> &params,
                const std::string &act)
      : act_(act), w_(params[0]), b_(params[1]) {}

  VariableHandle operator()(
      VariableHandle input,
      const framework::AttributeMap &conv_op_attrs = {{"paddings",
                                                       std::vector<int>{1, 1}},
                                                      {"use_cudnn", true}},
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
    if (act_.empty()) {
      return pre_act;
    }
    VariableHandle post_act(new Variable("conv"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<std::string> ParamNames() { return {w_->Name(), b_->Name()}; }

  std::string ActName() { return act_; }

 private:
  ParameterHandle w_;
  ParameterHandle b_;
  std::string act_;
};

class BatchNorm {
 public:
  explicit BatchNorm(int channel_in, const std::string &act = "") : act_(act) {
    // Use fill one to initialize scale and variance
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{channel_in};
    attrs["value"] = 1.0f;
    scale_ = GlobalParameterCollection().AddParameter(
        "BatchNormScale", "fill_constant", attrs);
    variance_ = GlobalParameterCollection().AddParameter(
        "BatchNormVariance", "fill_constant", attrs);

    // Use fill zero to initialize bias and mean
    attrs["value"] = 0.0f;
    bias_ = GlobalParameterCollection().AddParameter(
        "BatchNormBias", "fill_constant", attrs);
    mean_ = GlobalParameterCollection().AddParameter(
        "BatchNormMean", "fill_constant", attrs);

    GlobalParameterCollection().MarkNoGrad(variance_->Name());
    GlobalParameterCollection().MarkNoGrad(mean_->Name());
  }

  BatchNorm(const std::vector<ParameterHandle> &params, const std::string &act)
      : act_(act),
        scale_(params[0]),
        bias_(params[1]),
        mean_(params[2]),
        variance_(params[3]) {}

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
    if (act_.empty()) {
      return pre_act;
    }
    VariableHandle post_act(new Variable("batch_norm"));
    get_global_tape().AddOp(
        act_, {{"X", {pre_act}}}, {{"Out", {post_act}}}, {});
    return post_act;
  }

  std::vector<std::string> ParamNames() {
    return {scale_->Name(), bias_->Name(), mean_->Name(), variance_->Name()};
  }

  std::string ActName() { return act_; }

 private:
  ParameterHandle scale_;
  ParameterHandle bias_;
  ParameterHandle mean_;
  ParameterHandle variance_;
  std::string act_;
};

class Embedding {
 public:
  Embedding(int in_dim, int out_dim) {
    // Use Xavier to initialize lookup table
    float limit = sqrt(6.0 / static_cast<float>(in_dim + out_dim));
    framework::AttributeMap attrs;
    attrs["shape"] = std::vector<int>{in_dim, out_dim};
    attrs["dtype"] = framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["min"] = -limit;
    attrs["max"] = limit;
    attrs["seed"] = RandomSeed::GetRandomSeed();
    w_ = GlobalParameterCollection().AddParameter(
        "lookup_table", "uniform_random", attrs);
  }

  VariableHandle operator()(VariableHandle input) {
    VariableHandle output(new Variable("embedding"));
    get_global_tape().AddOp(
        "lookup_table",
        {{"Ids", {input}}, {"W", {w_}}},
        {{"Out", {output}}},
        {{"is_sparse", false}, {"is_distributed", false}, {"padding_idx", -2}});
    return output;
  }

 private:
  ParameterHandle w_;
};

// Calculate the top k accuracy of the prediction against the label
VariableHandle accuracy(VariableHandle prediction,
                        VariableHandle label,
                        int k = 1) {
  // Use top_k op to get top k prediction class labels
  VariableHandle topk_values(new Variable("accuracy"));
  VariableHandle topk_indices(new Variable("accuracy"));
  get_global_tape().AddOp("top_k",
                          {{"X", {prediction}}},
                          {{"Out", {topk_values}}, {"Indices", {topk_indices}}},
                          {{"k", k}});

  VariableHandle acc_out(new Variable("accuracy"));
  VariableHandle correct(new Variable("accuracy"));
  VariableHandle total(new Variable("accuracy"));
  get_global_tape().AddOp(
      "accuracy",
      {{"Out", {topk_values}}, {"Indices", {topk_indices}}, {"Label", {label}}},
      {{"Accuracy", {acc_out}}, {"Correct", {correct}}, {"Total", {total}}},
      {});
  return acc_out;
}

VariableHandle pool2d(VariableHandle x,
                      const framework::AttributeMap &attrs = {
                          {"strides", std::vector<int>{2, 2}},
                          {"use_cudnn", true}}) {
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

VariableHandle sigmoid(VariableHandle x) {
  VariableHandle out(new Variable("sigmoid"));
  get_global_tape().AddOp("sigmoid", {{"X", {x}}}, {{"Out", {out}}}, {});
  return out;
}

VariableHandle tanh(VariableHandle x) {
  VariableHandle out(new Variable("tanh"));
  get_global_tape().AddOp("tanh", {{"X", {x}}}, {{"Out", {out}}}, {});
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

VariableHandle add(VariableHandle x, VariableHandle y) {
  VariableHandle out(new Variable("add"));
  get_global_tape().AddOp(
      "elementwise_add", {{"X", {x}}, {"Y", {y}}}, {{"Out", {out}}}, {});
  return out;
}

VariableHandle var_from_vector(std::vector<int> input) {
  VariableHandle out(new Variable("from_vector"));
  int *out_data = out->GetMutable<framework::LoDTensor>()->mutable_data<int>(
      framework::make_ddim({static_cast<int64_t>(input.size())}),
      paddle::platform::CPUPlace());
  std::memcpy(out_data, input.data(), input.size() * sizeof(int));
  return out;
}

VariableHandle gather(VariableHandle x, std::vector<int> idx) {
  VariableHandle out(new Variable("gather"));
  get_global_tape().AddOp("gather",
                          {{"X", {x}}, {"Index", {var_from_vector(idx)}}},
                          {{"Out", {out}}},
                          {});
  return out;
}

std::vector<int> GetSeqLens(VariableHandle x) {
  std::vector<int> lens;
  auto lod = x->Get<paddle::framework::LoDTensor>().lod();
  PADDLE_ENFORCE_GT(lod.size(), 0);
  PADDLE_ENFORCE_GT(lod[0].size(), 1);
  for (int i = 0; i < lod[0].size() - 1; ++i) {
    lens.push_back(lod[0][i + 1] - lod[0][i]);
  }
  return lens;
}

VariableHandle ReorderAndPad(VariableHandle x, int padding_idx = -2) {
  VariableHandle out(new Variable("reorder"));
  auto input_tensor = x->Get<paddle::framework::LoDTensor>();
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(input_tensor.place()));

  auto dim = input_tensor.dims();
  auto lod = input_tensor.lod()[0];

  size_t batch_size = lod.size() - 1;
  size_t max_seq_len = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    max_seq_len = std::max(max_seq_len, lod[i + 1] - lod[i]);
  }

  int64_t length = static_cast<int64_t>(batch_size * max_seq_len);
  auto out_dim = paddle::framework::make_ddim({length, 1});

  auto *output_tensor = out->GetMutable<paddle::framework::LoDTensor>();
  int64_t *out_data =
      output_tensor->mutable_data<int64_t>(out_dim, input_tensor.place());
  const int64_t *in_data = input_tensor.data<int64_t>();

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      if (lod[i] + j < lod[i + 1]) {
        out_data[batch_size * j + i] = in_data[lod[i] + j];
      } else {
        out_data[batch_size * j + i] = -2;
      }
    }
  }

  return out;
}

VariableHandle reshape(VariableHandle x, std::vector<int> shape, bool inplace) {
  VariableHandle out(new Variable("reshape"));
  get_global_tape().AddOp("reshape",
                          {{"X", {x}}},
                          {{"Out", {out}}},
                          {{"shape", shape}, {"inplace", inplace}});
  return out;
}

VariableHandle concat(std::vector<VariableHandle> inputs, int axis = 0) {
  VariableHandle out(new Variable("concat"));
  get_global_tape().AddOp(
      "concat", {{"X", inputs}}, {{"Out", {out}}}, {{"axis", axis}});
  return out;
}

std::vector<VariableHandle> split(VariableHandle x) {
  // We need to known the runtime shape of x
  x->Value();
  int time_steps =
      static_cast<int>(x->Get<paddle::framework::LoDTensor>().dims()[0]);
  std::vector<VariableHandle> outputs;
  for (int i = 0; i < time_steps; ++i) {
    outputs.push_back(VariableHandle(new Variable("split")));
  }
  get_global_tape().AddOp(
      "split", {{"X", {x}}}, {{"Out", outputs}}, {{"num", time_steps}});
  //  for (auto out : outputs) {
  //    LOG(INFO) << out->Name();
  //    LOG(INFO) << out->Value();
  //  }

  return outputs;
}

VariableHandle fill_constant(std::vector<int> shape, float val = 0.0f) {
  VariableHandle output(new Variable("fill_constant"));
  paddle::framework::AttributeMap attrs;
  attrs["shape"] = shape;
  attrs["value"] = val;
  RunOperator("fill_constant", {}, {{"Out", {output}}}, attrs);
  return output;
}

std::vector<VariableHandle> lstm_step(VariableHandle x, VariableHandle c_prev) {
  VariableHandle cell_state(new Variable("lstm_step"));
  VariableHandle hidden_state(new Variable("lstm_step"));

  get_global_tape().AddOp("lstm_unit",
                          {{"X", {x}}, {"C_prev", {c_prev}}},
                          {{"C", {cell_state}}, {"H", {hidden_state}}},
                          {});
  return {cell_state, hidden_state};
}

VariableHandle CreateRecordioFileReader(std::string filename,
                                        std::vector<int> shape_concat,
                                        std::vector<int> ranks,
                                        std::vector<int> lod_levels) {
  std::ifstream infile(filename);
  PADDLE_ENFORCE(
      infile.good(),
      "%s doesn't exist; have you run the corresponding create_recordio.py?",
      filename);

  VariableHandle reader(new paddle::tape::Variable("reader"));

  RunOperator("create_recordio_file_reader",
              {},
              {{"Out", {reader}}},
              {{"filename", filename},
               {"shape_concat", shape_concat},
               {"ranks", ranks},
               {"lod_levels", lod_levels}});

  return reader;
}

VariableHandle CreateBatchReader(VariableHandle reader, int batch_size) {
  VariableHandle batch_reader(new paddle::tape::Variable("reader"));

  RunOperator("create_batch_reader",
              {{"UnderlyingReader", {reader}}},
              {{"Out", {batch_reader}}},
              {{"batch_size", batch_size}});

  return batch_reader;
}

VariableHandle CreateDoubleBufferReader(VariableHandle reader) {
  VariableHandle db_reader(new paddle::tape::Variable("reader"));

  RunOperator("create_double_buffer_reader",
              {{"UnderlyingReader", {reader}}},
              {{"Out", {db_reader}}},
              {});

  return db_reader;
}

void ResetReader(VariableHandle reader) {
  PADDLE_ENFORCE(reader->Var().IsType<framework::ReaderHolder>());
  reader->GetMutable<framework::ReaderHolder>()->ReInit();
}

std::vector<VariableHandle> ReadNext(VariableHandle reader, bool repeat) {
  PADDLE_ENFORCE(reader->Var().IsType<framework::ReaderHolder>());

  framework::LoDTensorArray data_holder;
  reader->GetMutable<framework::ReaderHolder>()->ReadNext(&data_holder);
  if (data_holder.empty()) {
    VLOG(5) << "ReInit reader";
    reader->GetMutable<framework::ReaderHolder>()->ReInit();
    reader->GetMutable<framework::ReaderHolder>()->ReadNext(&data_holder);
    PADDLE_ENFORCE(!data_holder.empty(), "Error reading file.");
    if (!repeat) {
      // FIXME(kexinzhao): Doing ReInit here will cause error when using
      // double_buffer reader
      // reader->GetMutable<framework::ReaderHolder>()->ReInit();
      return {};
    }
  }

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
