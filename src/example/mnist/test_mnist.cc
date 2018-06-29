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

#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "src/function.h"

using paddle::tape::VariableHandle;
using paddle::tape::Linear;
using paddle::tape::SGD;
using paddle::tape::Adam;
using paddle::tape::accuracy;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::OptimizableParameters;

using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;

TEST(Mnist, TestCPU) {
  std::string filename1 = "/tmp/mnist_train.recordio";
  std::string filename2 = "/tmp/mnist_test.recordio";
  auto train_reader = CreateRecordioFileReader(
      filename1, {32, 1, 28, 28, 32, 1}, {4, 2}, {0, 0});
  auto test_reader = CreateRecordioFileReader(
      filename2, {32, 1, 28, 28, 32, 1}, {4, 2}, {0, 0});

  Linear linear1(784, 200, "tanh");
  Linear linear2(200, 200, "tanh");
  Linear linear3(200, 10, "softmax");
  Adam adam(0.001);

  auto forward = [&](VariableHandle input) -> VariableHandle {
    return linear3(linear2(linear1(input)));
  };

  int total_steps = 10000;
  int print_step = 100;
  float threshold = 0.90f;

  for (int i = 0; i < total_steps; ++i) {
    reset_global_tape();
    auto data_label = ReadNext(train_reader, true);
    auto data = data_label[0];
    auto label = data_label[1];

    auto predict = forward(data);
    auto loss = mean(cross_entropy(predict, label));
    auto precision = accuracy(predict, label);

    get_global_tape().Backward(loss);

    for (auto w : OptimizableParameters()) {
      adam.Update(w);
    }

    // Every time certain amount of batches have been processed,
    // we test the average loss and accuracy on the test data set,
    // we stop training when the accuracy hit some threshold
    if ((i + 1) % print_step == 0) {
      std::vector<float> losses;
      std::vector<float> accuracies;

      while (true) {
        reset_global_tape();

        auto data_label = ReadNext(test_reader, false);
        if (data_label.empty()) {
          break;
        }

        auto data = data_label[0];
        auto label = data_label[1];

        auto predict = forward(data);
        auto loss = mean(cross_entropy(predict, label));
        auto precision = accuracy(predict, label);

        get_global_tape().Forward();

        losses.push_back(
            loss->Get<paddle::framework::LoDTensor>().data<float>()[0]);
        accuracies.push_back(
            precision->Get<paddle::framework::LoDTensor>().data<float>()[0]);
      }

      float avg_loss =
          std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
      float avg_accu =
          std::accumulate(accuracies.begin(), accuracies.end(), 0.0f) /
          accuracies.size();

      LOG(INFO) << "Pass #" << (i + 1) / print_step
                << ", test set evaluation result: Avg loss is " << avg_loss
                << ", Avg accuracy is " << avg_accu;

      if (avg_accu >= threshold) {
        LOG(INFO) << "Meets target accuracy, stop training";
        break;
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
