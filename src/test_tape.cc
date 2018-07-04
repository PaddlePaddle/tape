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
#include "src/optimizer.h"

using paddle::tape::VariableHandle;
using paddle::tape::Variable;
using paddle::tape::Linear;
using paddle::tape::Convolution2D;
using paddle::tape::SGD;
using paddle::tape::Adam;
using paddle::tape::Fill;
using paddle::tape::BatchNorm;
using paddle::tape::dropout;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;
using paddle::tape::BackwardAndUpdate;
using paddle::framework::LoDTensor;
using paddle::tape::ParameterCollection;
using paddle::tape::ParameterHandle;
using paddle::tape::GlobalParameterCollection;

template <typename T>
bool EnforceClose(ParameterHandle p1, ParameterHandle p2, T epsilon) {
  auto& t1 = p1->Get<LoDTensor>();
  auto& t2 = p2->Get<LoDTensor>();

  PADDLE_ENFORCE(t1.numel() == t2.numel());
  for (int i = 0; i < t1.numel(); ++i) {
    T d1 = t1.data<T>()[i], d2 = t2.data<T>()[i];
    PADDLE_ENFORCE(d1 - d2 <= epsilon);
    PADDLE_ENFORCE(d2 - d1 <= epsilon);
  }
}

TEST(Tape, TestDropout) {
  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{3, 3};
  Fill filler(initializer, attrs);

  reset_global_tape();

  auto input = filler();
  auto loss = dropout(input);
  LOG(INFO) << input->Value();
  LOG(INFO) << loss->Value();

  get_global_tape().Backward(loss);
  LOG(INFO) << input->Grad()->Value();
}

TEST(Tape, TestPool2d) {
  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{1, 1, 3, 3};
  Fill filler(initializer, attrs);

  reset_global_tape();

  auto input = filler();
  auto loss = pool2d(input);
  LOG(INFO) << input->Value();
  LOG(INFO) << loss->Value();

  get_global_tape().Backward(loss);
  LOG(INFO) << input->Grad()->Value();
}

TEST(Tape, TestBatchNorm) {
  BatchNorm bn(4, "relu");

  Adam adam(0.001);

  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{32, 4, 8, 8};
  Fill filler(initializer, attrs);

  for (int i = 0; i < 5; ++i) {
    reset_global_tape();

    auto input = filler();
    auto loss = bn(input);

    LOG(INFO) << loss->Value();
    BackwardAndUpdate(loss, &adam);
  }
}

TEST(Tape, TestGraph) {
  Convolution2D conv1(3, 16, 3, "relu");
  Convolution2D conv2(16, 1, 3, "relu");

  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{32, 3, 8, 8};
  Fill filler(initializer, attrs);

  reset_global_tape();

  auto input = filler();
  auto loss = mean(conv2(conv1(input)));
  get_global_tape().Backward(loss);

  LOG(INFO) << get_global_tape().GraphVizString(false);
}

TEST(Tape, TestConv) {
  Convolution2D conv1(3, 16, 3, "relu");
  Convolution2D conv2(16, 1, 3, "relu");

  Adam adam(0.001);

  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{32, 3, 8, 8};
  Fill filler(initializer, attrs);

  for (int i = 0; i < 2; ++i) {
    reset_global_tape();

    auto input = filler();
    auto loss = mean(conv2(conv1(input)));
    LOG(INFO) << loss->Value();

    BackwardAndUpdate(loss, &adam);
  }
}

TEST(Tape, TestMLP) {
  Linear linear1(3, 3, "relu");
  Linear linear2(3, 3, "relu");

  SGD sgd(0.001);

  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{3, 3};
  Fill filler(initializer, attrs);

  for (int i = 0; i < 2; ++i) {
    reset_global_tape();

    auto input = filler();
    auto loss = mean(linear2(linear1(input)));
    LOG(INFO) << loss->Value();

    BackwardAndUpdate(loss, &sgd);
  }
}

TEST(Tape, TestSaveLoadLayer) {
  std::string file_path = "/tmp/test_layer_save_load/";
  Linear linear1(3, 6, "relu");
  Linear linear2(6, 3);

  auto& global_pc = GlobalParameterCollection();
  global_pc.SaveAllParameters(file_path);

  ParameterCollection loaded_pc(file_path);
  PADDLE_ENFORCE_EQ(global_pc.OptimizableParameters().size(),
                    loaded_pc.OptimizableParameters().size());

  Linear loaded_linear1(loaded_pc.LookUp(linear1.ParamNames()),
                        linear1.ActName());
  Linear loaded_linear2(loaded_pc.LookUp(linear2.ParamNames()),
                        linear2.ActName());

  for (size_t i = 0; i < global_pc.OptimizableParameters().size(); ++i) {
    auto old_param = global_pc.OptimizableParameters()[i];
    auto new_param = loaded_pc.OptimizableParameters()[i];
    EnforceClose<float>(old_param, new_param, 0.0001);
    PADDLE_ENFORCE_NE(old_param.get(),
                      new_param.get(),
                      "Loaded parameter should be in different memory");
  }

  PADDLE_ENFORCE_EQ(linear1.ActName(), loaded_linear1.ActName());
  PADDLE_ENFORCE_EQ(linear2.ActName(), loaded_linear2.ActName());

  PADDLE_ENFORCE_EQ(system(std::string("rm -r " + file_path).c_str()), 0);
}

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
