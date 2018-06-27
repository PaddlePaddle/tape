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

#include <fstream>

#include "gtest/gtest.h"
#include "src/function.h"

using paddle::tape::Variable;
using paddle::tape::VariableHandle;

using paddle::tape::Fill;
using paddle::tape::Linear;
using paddle::tape::SGD;
using paddle::tape::concat;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;

using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;

using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;

VariableHandle fill_constant(std::vector<int> dims, float value) {
  VariableHandle out(new Variable("out"));

  paddle::framework::AttributeMap attrs;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_INT64;
  attrs["shape"] = dims;
  attrs["value"] = value;
  paddle::framework::OpDesc op_desc =
      paddle::tape::CreateOpDesc("fill_constant", {}, {{"Out", {out}}}, attrs);
  paddle::tape::ScopeWrapper scope({}, {{"Out", {out}}});
  paddle::framework::OpRegistry::CreateOp(op_desc)->Run(
      scope, paddle::platform::CPUPlace());

  return out;
}

VariableHandle get_real_fake_labels(int batch_size) {
  auto real_label = fill_constant({batch_size, 1}, 1.0);
  auto fake_label = fill_constant({batch_size, 1}, 0.0);

  VariableHandle out(new Variable("out"));
  out->GetMutable<paddle::framework::LoDTensor>();
  paddle::framework::OpDesc op_desc = paddle::tape::CreateOpDesc(
      "concat", {{"X", {real_label, fake_label}}}, {{"Out", {out}}}, {});
  paddle::tape::ScopeWrapper scope({{"X", {real_label, fake_label}}},
                                   {{"Out", {out}}});
  paddle::framework::OpRegistry::CreateOp(op_desc)->Run(
      scope, paddle::platform::CPUPlace());

  return out;
}

bool is_file_exist(const std::string& fileName) {
  std::ifstream infile(fileName);
  return infile.good();
}

TEST(FCGAN, TestCPU) {
  std::string filename = "/tmp/mnist.recordio";
  PADDLE_ENFORCE(is_file_exist(filename),
                 "file doesn't exist; have you run create_mnist_recordio.py");
  auto reader =
      CreateRecordioFileReader(filename, {32, 784, 32, 1}, {2, 2}, {0, 0});

  // Discriminator
  Linear d_linear1(784, 200, "relu");
  Linear d_linear2(200, 2, "relu");

  // Generator
  Linear g_linear1(100, 200, "relu");
  Linear g_linear2(200, 784, "relu");

  SGD sgd(0.001);

  std::string initializer = "uniform_random";
  paddle::framework::AttributeMap attrs;
  attrs["min"] = -1.0f;
  attrs["max"] = 1.0f;
  attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
  attrs["seed"] = 123;
  attrs["shape"] = std::vector<int>{32, 100};
  Fill noise_generator(initializer, attrs);

  for (int i = 0; i < 10; ++i) {
    // train Discriminator
    {
      reset_global_tape();

      // generate fake image
      auto noise = noise_generator();
      auto fake_image = g_linear2(g_linear1(noise));

      auto data_label = ReadNext(reader);
      auto real_image = data_label[0];

      auto image = concat({real_image, fake_image});
      auto label = get_real_fake_labels(32);  // [1, ..., 1, 0, ..., 0]

      auto loss = cross_entropy(softmax(d_linear2(d_linear1(image))), label);
      auto mean_loss = mean(loss);
      LOG(INFO) << "D " << mean_loss->Value();

      get_global_tape().Backward(mean_loss);

      for (auto w : d_linear1.Params()) {
        sgd.Update(w);
      }
      for (auto w : d_linear2.Params()) {
        sgd.Update(w);
      }
    }

    // train Generator
    {
      reset_global_tape();

      // generate fake image
      auto noise = noise_generator();
      auto label = fill_constant({32, 1}, 1);
      auto fake_image = g_linear2(g_linear1(noise));

      auto loss =
          cross_entropy(softmax(d_linear2(d_linear1(fake_image))), label);
      auto mean_loss = mean(loss);
      LOG(INFO) << "G " << mean_loss->Value();

      get_global_tape().Backward(mean_loss);

      for (auto w : g_linear1.Params()) {
        sgd.Update(w);
      }
      for (auto w : g_linear2.Params()) {
        sgd.Update(w);
      }
    }
  }
}

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
