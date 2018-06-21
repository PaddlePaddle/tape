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
#include "paddle/fluid/framework/reader.h"
#include "src/function.h"

using paddle::tape::VariableHandle;
using paddle::tape::Variable;
using paddle::tape::Linear;
using paddle::tape::Mean;
using paddle::tape::SGD;
using paddle::tape::Fill;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;

VariableHandle create_recordio_file_reader(std::string filename,
                                           std::vector<int> shape_concat,
                                           std::vector<int> ranks,
                                           std::vector<int> lod_levels) {
  VariableHandle reader(new paddle::tape::Variable("reader"));
  reader->MutableDesc()->SetType(paddle::framework::proto::VarType::READER);

  paddle::tape::Tape init_tape;
  init_tape.AddOp("create_recordio_file_reader",
                  {},
                  {{"Out", {reader}}},
                  {{"filename", filename},
                   {"shape_concat", shape_concat},
                   {"ranks", ranks},
                   {"lod_levels", lod_levels}});
  init_tape.Forward();

  return reader;
}

TEST(Tape, TestReader) {
  VariableHandle data_label(new paddle::tape::Variable("data_label"));
  VariableHandle reader = create_recordio_file_reader(
      "./data/mnist.recordio", {-1, 784, -1, 1}, {2, 2}, {0, 0});

  reader->MutableVar()->GetMutable<paddle::framework::ReaderHolder>()->ReadNext(
      data_label->MutableVar()
          ->GetMutable<paddle::framework::LoDTensorArray>());
}

/*
TEST(Tape, TestMLP) {
  LOG(INFO) << "TestMLP";
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
    LOG(INFO) << loss->value();

    get_global_tape().Backward(loss);

    for (auto w : linear1.Params()) {
      sgd.Update(w);
    }
    for (auto w : linear2.Params()) {
      sgd.Update(w);
    }
  }
}
*/

int main(int argc, char **argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
