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

using paddle::tape::Linear;
using paddle::tape::SGD;
using paddle::tape::Adam;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;

using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;

bool is_file_exist(const std::string& fileName) {
  std::ifstream infile(fileName);
  return infile.good();
}

TEST(Mnist, TestCPU) {
  std::string filename = "/tmp/mnist.recordio";
  PADDLE_ENFORCE(is_file_exist(filename),
                 "file doesn't exist; have you run create_mnist_recordio.py");
  auto reader = CreateRecordioFileReader(
      filename, {32, 1, 28, 28, 32, 1}, {4, 2}, {0, 0});

  Linear linear1(784, 200, "relu");
  Linear linear2(200, 200, "relu");
  Linear linear3(200, 10, "relu");
  Adam adam(0.001);

  int print_step = 100;
  float avg_loss = 0.0;

  for (int i = 0; i < 1000; ++i) {
    reset_global_tape();
    auto data_label = ReadNext(reader);
    auto data = data_label[0];
    auto label = data_label[1];

    auto predict = softmax(linear3(linear2(linear1(data))));
    auto loss = mean(cross_entropy(predict, label));

    avg_loss +=
        loss->Value().Get<paddle::framework::LoDTensor>().data<float>()[0];
    if ((i + 1) % print_step == 0) {
      LOG(INFO) << avg_loss / print_step;
      avg_loss = 0;
    }

    get_global_tape().Backward(loss);

    for (auto w : linear1.Params()) {
      adam.Update(w);
    }
    for (auto w : linear2.Params()) {
      adam.Update(w);
    }
    for (auto w : linear3.Params()) {
      adam.Update(w);
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
