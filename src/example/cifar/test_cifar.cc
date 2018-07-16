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

#include <chrono>  // NOLINT
#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "src/function.h"
#include "src/optimizer.h"

using paddle::tape::VariableHandle;
using paddle::tape::Linear;
using Conv2D = paddle::tape::Convolution2D;
using paddle::tape::BatchNorm;
using paddle::tape::SGD;
using paddle::tape::Adam;
using paddle::tape::accuracy;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::OptimizableParameters;
using paddle::tape::BackwardAndUpdate;
using paddle::tape::ParameterCollection;
using paddle::tape::ParameterHandle;
using paddle::tape::GlobalParameterCollection;

using paddle::tape::CreateRecordioFileReader;
using paddle::tape::ReadNext;

TEST(Cifar, TestGPU) {
  // auto place = paddle::platform::CPUPlace();
  auto place = paddle::platform::CUDAPlace(0);
  reset_global_tape(place);

  const int batch_size = 128;
  LOG(INFO) << "Batch size is " << batch_size << std::endl;
  std::string save_model_path = "/tmp/cifar_model/";
  std::string filename1 =
      "/tmp/cifar10_train_" + std::to_string(batch_size) + "_CPUPlace.recordio";
  std::string filename2 =
      "/tmp/cifar10_test_" + std::to_string(batch_size) + "_CPUPlace.recordio";
  auto train_reader = CreateRecordioFileReader(
      filename1, {batch_size, 3, 32, 32, batch_size, 1}, {4, 2}, {0, 0});
  auto test_reader = CreateRecordioFileReader(
      filename2, {batch_size, 3, 32, 32, batch_size, 1}, {4, 2}, {0, 0});

  // input 3x32x32
  // after conv1_1   64x32x32
  // after conv1_2   64x32x32
  // after pool1     64x16x16
  Conv2D conv1_1(3, 64, 3);
  BatchNorm bn1_1(64, "relu");
  Conv2D conv1_2(64, 64, 3);
  BatchNorm bn1_2(64, "relu");

  // after conv2_1   128x16x16
  // after conv2_2   128x16x16
  // after pool2     128x8x8
  Conv2D conv2_1(64, 128, 3);
  BatchNorm bn2_1(128, "relu");
  Conv2D conv2_2(128, 128, 3);
  BatchNorm bn2_2(128, "relu");

  // after conv3_1   256x8x8
  // after conv3_2   256x8x8
  // after conv3_3   256x8x8
  // after pool3     256x4x4
  Conv2D conv3_1(128, 256, 3);
  BatchNorm bn3_1(256, "relu");
  Conv2D conv3_2(256, 256, 3);
  BatchNorm bn3_2(256, "relu");
  Conv2D conv3_3(256, 256, 3);
  BatchNorm bn3_3(256, "relu");

  // after conv4_1   512x4x4
  // after conv4_2   512x4x4
  // after conv4_3   512x4x4
  // after pool4     512x2x2
  Conv2D conv4_1(256, 512, 3);
  BatchNorm bn4_1(512, "relu");
  Conv2D conv4_2(512, 512, 3);
  BatchNorm bn4_2(512, "relu");
  Conv2D conv4_3(512, 512, 3);
  BatchNorm bn4_3(512, "relu");

  // after conv5_1   512x2x2
  // after conv5_2   512x2x2
  // after conv5_3   512x2x2
  // after pool5     512x1x1
  Conv2D conv5_1(512, 512, 3);
  BatchNorm bn5_1(512, "relu");
  Conv2D conv5_2(512, 512, 3);
  BatchNorm bn5_2(512, "relu");
  Conv2D conv5_3(512, 512, 3);
  BatchNorm bn5_3(512, "relu");

  // Input dim 512x1x1 = 512
  Linear fc1(512, 512);
  BatchNorm bn6(512, "relu");
  Linear fc2(512, 512);
  Linear fc3(512, 10, "softmax");

  Adam adam(0.001);

  auto vgg16_forward = [&](VariableHandle input,
                           bool is_test) -> VariableHandle {
    // Set attribute for batchnorm and dropout op
    paddle::framework::AttributeMap bn_attrs, d_attrs;
    bn_attrs["is_test"] = is_test;
    d_attrs["is_test"] = is_test;
    d_attrs["dropout_prob"] = 0.3f;
    auto temp1 = dropout(bn1_1(conv1_1(input), bn_attrs), d_attrs);
    auto pool1 = pool2d(bn1_2(conv1_2(temp1), bn_attrs));

    d_attrs["dropout_prob"] = 0.4f;
    auto temp2 = dropout(bn2_1(conv2_1(pool1), bn_attrs), d_attrs);
    auto pool2 = pool2d(bn2_2(conv2_2(temp2), bn_attrs));

    auto temp3_1 = dropout(bn3_1(conv3_1(pool2), bn_attrs), d_attrs);
    auto temp3_2 = dropout(bn3_2(conv3_2(temp3_1), bn_attrs), d_attrs);
    auto pool3 = pool2d(bn3_3(conv3_3(temp3_2), bn_attrs));

    auto temp4_1 = dropout(bn4_1(conv4_1(pool3), bn_attrs), d_attrs);
    auto temp4_2 = dropout(bn4_2(conv4_2(temp4_1), bn_attrs), d_attrs);
    auto pool4 = pool2d(bn4_3(conv4_3(temp4_2), bn_attrs));

    auto temp5_1 = dropout(bn5_1(conv5_1(pool4), bn_attrs), d_attrs);
    auto temp5_2 = dropout(bn5_2(conv5_2(temp5_1), bn_attrs), d_attrs);
    auto pool5 = pool2d(bn5_3(conv5_3(temp5_2), bn_attrs));

    d_attrs["dropout_prob"] = 0.5f;
    auto temp6 = bn6(fc1(dropout(pool5, d_attrs)), bn_attrs);
    return fc3(fc2(dropout(temp6, d_attrs)));
  };

  int total_steps = 10000;
  int test_steps = 1000;
  int print_step = 2000;
  float threshold = 0.88f;
  int iter_num = 1050;
  int skip_batch_num = 50;

  auto start = std::chrono::system_clock::now();
  int num_samples = 0;
  // Training
  for (int i = 0; i < total_steps; ++i) {
    LOG(INFO) << "Train step #" << i;

    if (i == iter_num) {
      break;
    }

    if (i == skip_batch_num) {
      start = std::chrono::system_clock::now();
      num_samples = 0;
    }

    reset_global_tape(place);
    auto data_label = ReadNext(train_reader, true);
    auto data = data_label[0];
    auto label = data_label[1];

    auto predict = vgg16_forward(data, false);
    auto loss = mean(cross_entropy(predict, label));
    auto precision = accuracy(predict, label);

    BackwardAndUpdate(loss, &adam);

    num_samples += batch_size;

    // Every time certain amount of batches have been processed,
    // we test the average loss and accuracy on the test data set,
    // we stop training when the accuracy hit some threshold
    if ((i + 1) % print_step == 0) {
      std::vector<float> losses;
      std::vector<float> accuracies;

      LOG(INFO) << "Start testing";
      for (int i = 0; i < test_steps; ++i) {
        reset_global_tape(place);

        auto data_label = ReadNext(test_reader, false);
        if (data_label.empty()) {
          LOG(INFO) << "Full test set has been traversed";
          break;
        }

        auto data = data_label[0];
        auto label = data_label[1];

        auto predict = vgg16_forward(data, true);
        auto loss = mean(cross_entropy(predict, label));
        auto precision = accuracy(predict, label);

        get_global_tape().Forward();

        losses.push_back(loss->FetchValue()
                             ->Get<paddle::framework::LoDTensor>()
                             .data<float>()[0]);
        accuracies.push_back(precision->FetchValue()
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
        LOG(INFO) << "Meets target accuracy, stop training and save parameters";
        GlobalParameterCollection().SaveAllParameters(save_model_path);
        break;
      }
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  LOG(INFO) << "Total wall clock time for iteration num " << (iter_num + 1)
            << " is " << elapsed_time.count() << " seconds" << std::endl;
  LOG(INFO) << "Total samples: " << num_samples
            << "; Throughput: " << num_samples / elapsed_time.count()
            << std::endl;

  return;

  // Inference using test set
  LOG(INFO) << "Start inferencing and load parameters";
  ParameterCollection loaded_pc(save_model_path);

  // Reconstruct layers by loading the saved parameters
  Conv2D inf_conv1_1(loaded_pc.LookUp(conv1_1.ParamNames()), conv1_1.ActName());
  BatchNorm inf_bn1_1(loaded_pc.LookUp(bn1_1.ParamNames()), bn1_1.ActName());
  Conv2D inf_conv1_2(loaded_pc.LookUp(conv1_2.ParamNames()), conv1_2.ActName());
  BatchNorm inf_bn1_2(loaded_pc.LookUp(bn1_2.ParamNames()), bn1_2.ActName());

  Conv2D inf_conv2_1(loaded_pc.LookUp(conv2_1.ParamNames()), conv2_1.ActName());
  BatchNorm inf_bn2_1(loaded_pc.LookUp(bn2_1.ParamNames()), bn2_1.ActName());
  Conv2D inf_conv2_2(loaded_pc.LookUp(conv2_2.ParamNames()), conv2_2.ActName());
  BatchNorm inf_bn2_2(loaded_pc.LookUp(bn2_2.ParamNames()), bn2_2.ActName());

  Conv2D inf_conv3_1(loaded_pc.LookUp(conv3_1.ParamNames()), conv3_1.ActName());
  BatchNorm inf_bn3_1(loaded_pc.LookUp(bn3_1.ParamNames()), bn3_1.ActName());
  Conv2D inf_conv3_2(loaded_pc.LookUp(conv3_2.ParamNames()), conv3_2.ActName());
  BatchNorm inf_bn3_2(loaded_pc.LookUp(bn3_2.ParamNames()), bn3_2.ActName());
  Conv2D inf_conv3_3(loaded_pc.LookUp(conv3_3.ParamNames()), conv3_3.ActName());
  BatchNorm inf_bn3_3(loaded_pc.LookUp(bn3_3.ParamNames()), bn3_3.ActName());

  Conv2D inf_conv4_1(loaded_pc.LookUp(conv4_1.ParamNames()), conv4_1.ActName());
  BatchNorm inf_bn4_1(loaded_pc.LookUp(bn4_1.ParamNames()), bn4_1.ActName());
  Conv2D inf_conv4_2(loaded_pc.LookUp(conv4_2.ParamNames()), conv4_2.ActName());
  BatchNorm inf_bn4_2(loaded_pc.LookUp(bn4_2.ParamNames()), bn4_2.ActName());
  Conv2D inf_conv4_3(loaded_pc.LookUp(conv4_3.ParamNames()), conv4_3.ActName());
  BatchNorm inf_bn4_3(loaded_pc.LookUp(bn4_3.ParamNames()), bn4_3.ActName());

  Conv2D inf_conv5_1(loaded_pc.LookUp(conv5_1.ParamNames()), conv5_1.ActName());
  BatchNorm inf_bn5_1(loaded_pc.LookUp(bn5_1.ParamNames()), bn5_1.ActName());
  Conv2D inf_conv5_2(loaded_pc.LookUp(conv5_2.ParamNames()), conv5_2.ActName());
  BatchNorm inf_bn5_2(loaded_pc.LookUp(bn5_2.ParamNames()), bn5_2.ActName());
  Conv2D inf_conv5_3(loaded_pc.LookUp(conv5_3.ParamNames()), conv5_3.ActName());
  BatchNorm inf_bn5_3(loaded_pc.LookUp(bn5_3.ParamNames()), bn5_3.ActName());

  Linear inf_fc1(loaded_pc.LookUp(fc1.ParamNames()), fc1.ActName());
  BatchNorm inf_bn6(loaded_pc.LookUp(bn6.ParamNames()), bn6.ActName());
  Linear inf_fc2(loaded_pc.LookUp(fc2.ParamNames()), fc2.ActName());
  Linear inf_fc3(loaded_pc.LookUp(fc3.ParamNames()), fc3.ActName());

  auto vgg16_inference = [&](VariableHandle input,
                             bool is_test) -> VariableHandle {
    // Set attribute for batchnorm and dropout op
    paddle::framework::AttributeMap bn_attrs, d_attrs;
    bn_attrs["is_test"] = is_test;
    d_attrs["is_test"] = is_test;
    d_attrs["dropout_prob"] = 0.3f;
    auto temp1 = dropout(inf_bn1_1(inf_conv1_1(input), bn_attrs), d_attrs);
    auto pool1 = pool2d(inf_bn1_2(inf_conv1_2(temp1), bn_attrs));

    d_attrs["dropout_prob"] = 0.4f;
    auto temp2 = dropout(inf_bn2_1(inf_conv2_1(pool1), bn_attrs), d_attrs);
    auto pool2 = pool2d(inf_bn2_2(inf_conv2_2(temp2), bn_attrs));

    auto temp3_1 = dropout(inf_bn3_1(inf_conv3_1(pool2), bn_attrs), d_attrs);
    auto temp3_2 = dropout(inf_bn3_2(inf_conv3_2(temp3_1), bn_attrs), d_attrs);
    auto pool3 = pool2d(inf_bn3_3(inf_conv3_3(temp3_2), bn_attrs));

    auto temp4_1 = dropout(inf_bn4_1(inf_conv4_1(pool3), bn_attrs), d_attrs);
    auto temp4_2 = dropout(inf_bn4_2(inf_conv4_2(temp4_1), bn_attrs), d_attrs);
    auto pool4 = pool2d(inf_bn4_3(inf_conv4_3(temp4_2), bn_attrs));

    auto temp5_1 = dropout(inf_bn5_1(inf_conv5_1(pool4), bn_attrs), d_attrs);
    auto temp5_2 = dropout(inf_bn5_2(inf_conv5_2(temp5_1), bn_attrs), d_attrs);
    auto pool5 = pool2d(inf_bn5_3(inf_conv5_3(temp5_2), bn_attrs));

    d_attrs["dropout_prob"] = 0.5f;
    auto temp6 = inf_bn6(inf_fc1(dropout(pool5, d_attrs)), bn_attrs);
    return inf_fc3(inf_fc2(dropout(temp6, d_attrs)));
  };

  std::vector<float> losses;
  std::vector<float> accuracies;

  while (true) {
    reset_global_tape(place);

    auto data_label = ReadNext(test_reader, false);
    if (data_label.empty()) {
      break;
    }

    auto data = data_label[0];
    auto label = data_label[1];

    auto predict = vgg16_inference(data, true);
    auto loss = mean(cross_entropy(predict, label));
    auto precision = accuracy(predict, label);

    get_global_tape().Forward();

    losses.push_back(loss->FetchValue()
                         ->Get<paddle::framework::LoDTensor>()
                         .data<float>()[0]);
    accuracies.push_back(precision->FetchValue()
                             ->Get<paddle::framework::LoDTensor>()
                             .data<float>()[0]);
  }

  float avg_loss =
      std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
  float avg_accu = std::accumulate(accuracies.begin(), accuracies.end(), 0.0f) /
                   accuracies.size();

  LOG(INFO) << "Inference on test set result: Avg loss is " << avg_loss
            << ", Avg accuracy is " << avg_accu;

  PADDLE_ENFORCE_EQ(system(std::string("rm -r " + save_model_path).c_str()), 0);
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
