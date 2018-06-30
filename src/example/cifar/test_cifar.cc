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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "src/function.h"

using paddle::tape::VariableHandle;
using paddle::tape::Linear;
using Conv2D = paddle::tape::Convolution2D;
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

TEST(Cifar, TestCPU) {
  std::string filename1 = "/tmp/cifar10_train_64_CUDAPlace(1).recordio";
  std::string filename2 = "/tmp/cifar10_test_64_CUDAPlace(1).recordio";
  auto train_reader = CreateRecordioFileReader(
      filename1, {128, 3, 32, 32, 32, 1}, {4, 2}, {0, 0});
  auto test_reader = CreateRecordioFileReader(
      filename2, {128, 3, 32, 32, 32, 1}, {4, 2}, {0, 0});

  // input 3x32x32
  // after conv1_1   64x32x32
  // after conv1_2   64x32x32
  // after pool1     64x16x16
  Conv2D conv1_1(3, 64, 3);
  Conv2D conv1_2(64, 64, 3);

  // after conv2_1   128x16x16
  // after conv2_2   128x16x16
  // after pool2     128x8x8
  Conv2D conv2_1(64, 128, 3);
  Conv2D conv2_2(128, 128, 3);

  // after conv3_1   256x8x8
  // after conv3_2   256x8x8
  // after conv3_3   256x8x8
  // after pool3     256x4x4
  Conv2D conv3_1(128, 256, 3);
  Conv2D conv3_2(256, 256, 3);
  Conv2D conv3_3(256, 256, 3);

  // after conv4_1   512x4x4
  // after conv4_2   512x4x4
  // after conv4_3   512x4x4
  // after pool4     512x2x2
  Conv2D conv4_1(256, 512, 3);
  Conv2D conv4_2(512, 512, 3);
  Conv2D conv4_3(512, 512, 3);

  // after conv5_1   512x2x2
  // after conv5_2   512x2x2
  // after conv5_3   512x2x2
  // after pool5     512x1x1
  Conv2D conv5_1(512, 512, 3);
  Conv2D conv5_2(512, 512, 3);
  Conv2D conv5_3(512, 512, 3);

  // Input dim 512x1x1 = 512
  Linear fc1(512, 512, "relu");
  Linear fc2(512, 512, "relu");
  Linear fc3(512, 10, "softmax");

  SGD adam(0.001);

  auto vgg16_forward = [&](VariableHandle input) -> VariableHandle {
    auto pool1 = pool2d(conv1_2(conv1_1(input)));
    auto pool2 = pool2d(conv2_2(conv2_1(pool1)));
    auto pool3 = pool2d(conv3_3(conv3_2(conv3_1(pool2))));
    auto pool4 = pool2d(conv4_3(conv4_2(conv4_1(pool3))));
    auto pool5 = pool2d(conv5_3(conv5_2(conv5_1(pool4))));
    return fc3(fc2(fc1(pool5)));
  };

  int total_steps = 10000;
  int test_steps = 1000;
  int print_step = 5;
  float threshold = 0.6f;

  //  auto place = paddle::platform::CPUPlace();
  auto place = paddle::platform::CUDAPlace(1);
  for (int i = 0; i < total_steps; ++i) {
    LOG(INFO) << "Train step #" << i;

    reset_global_tape(place);
    auto data_label = ReadNext(train_reader, true);
    auto data = data_label[0];
    auto label = data_label[1];

    auto predict = vgg16_forward(data);
    auto loss = mean(cross_entropy(predict, label));
    auto precision = accuracy(predict, label);

    LOG(INFO) << "Before forward";
    LOG(INFO) << loss->Value();
    LOG(INFO) << "After forward";

    LOG(INFO) << "Before backward";
    get_global_tape().Backward(loss);
    LOG(INFO) << "After backward";

    LOG(INFO) << "Before optimizer";
    // Update all parameters
    for (auto w : OptimizableParameters()) {
      adam.Update(w);
    }
    LOG(INFO) << "After optimizer";

    // Every time certain amount of batches have been processed,
    // we test the average loss and accuracy on the test data set,
    // we stop training when the accuracy hit some threshold
    if ((i + 1) % print_step == 0) {
      std::vector<float> losses;
      std::vector<float> accuracies;

      for (int i = 0; i < test_steps; ++i) {
        LOG(INFO) << "Test step #" << i;

        reset_global_tape(place);

        auto data_label = ReadNext(test_reader, false);
        if (data_label.empty()) {
          LOG(INFO) << "full test set has been traversed";
          break;
        }

        auto data = data_label[0];
        auto label = data_label[1];

        auto predict = vgg16_forward(data);
        auto loss = mean(cross_entropy(predict, label));
        auto precision = accuracy(predict, label);

        get_global_tape().Forward();

        losses.push_back(loss->Value()
                             .CopyToCPU()
                             ->Get<paddle::framework::LoDTensor>()
                             .data<float>()[0]);
        accuracies.push_back(precision->Value()
                                 .CopyToCPU()
                                 ->Get<paddle::framework::LoDTensor>()
                                 .data<float>()[0]);
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
  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    places.emplace_back(paddle::platform::CUDAPlace(i));
  }
  LOG(INFO) << "DeviceCount " << count;
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
