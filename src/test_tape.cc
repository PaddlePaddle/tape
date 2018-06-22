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

#include "gtest/gtest.h"
#include "src/function.h"

using paddle::tape::VariableHandle;
using paddle::tape::Variable;
using paddle::tape::Linear;
using paddle::tape::Convolution2D;
using paddle::tape::Mean;
using paddle::tape::SGD;
using paddle::tape::Fill;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;

TEST(Tape, TestReader) {
  VariableHandle data_label(new paddle::tape::Variable("data_label"));
  VariableHandle reader = CreateRecordioFileReader(
      "/tape/src/data/mnist.recordio", {32, 1, 28, 28, 32, 1}, {4, 2}, {0, 0});
  ReadNext(reader, data_label);
  LOG(INFO) << *data_label;
}

TEST(Tape, TestRelu) {
  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["shape"] = std::vector<int>{10};
  attrs["seed"] = 123;
  Fill filler(initializer, attrs);

  VariableHandle input(new Variable("input"));
  filler(input);
  auto loss = relu(input);
  LOG(INFO) << input->Value();
  LOG(INFO) << loss->Value();
}

TEST(Tape, TestConv) {
  Convolution2D conv1(3, 16, 3, "relu");
  Convolution2D conv2(16, 1, 3, "relu");
  Mean mean;

  SGD sgd(0.001);

  std::string initializer = "fill_constant";
  paddle::framework::AttributeMap attrs;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["shape"] = std::vector<int>{32, 3, 8, 8};
  attrs["value"] = 1.0f;
  Fill filler(initializer, attrs);

  for (int i = 0; i < 2; ++i) {
    reset_global_tape();

    VariableHandle input(new Variable("input"));
    filler(input);

    auto loss = mean(conv2(conv1(input)));

    get_global_tape().Backward(loss);

    for (auto w : conv1.Params()) {
      sgd.Update(w);
    }
    for (auto w : conv2.Params()) {
      sgd.Update(w);
    }
  }
}

TEST(Tape, TestMLP) {
  Linear linear1(3, 3, "relu");
  Linear linear2(3, 3, "relu");
  Mean mean;

  SGD sgd(0.001);

  std::string initializer = "fill_constant";
  paddle::framework::AttributeMap attrs;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["shape"] = std::vector<int>{3, 3};
  attrs["value"] = 1.0f;
  Fill filler(initializer, attrs);

  for (int i = 0; i < 2; ++i) {
    reset_global_tape();

    VariableHandle input(new Variable("input"));
    filler(input);

    auto loss = mean(linear2(linear1(input)));
    LOG(INFO) << loss->Value();

    get_global_tape().Backward(loss);

    for (auto w : linear1.Params()) {
      sgd.Update(w);
    }
    for (auto w : linear2.Params()) {
      sgd.Update(w);
    }
  }
}

int main(int argc, char **argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
