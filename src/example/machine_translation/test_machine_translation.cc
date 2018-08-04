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
#include "paddle/fluid/platform/place.h"
#include "src/function.h"
#include "src/optimizer.h"

using paddle::tape::VariableHandle;
using paddle::tape::Variable;
using paddle::tape::Linear;
using Conv2D = paddle::tape::Convolution2D;
using paddle::tape::BatchNorm;
using paddle::tape::Embedding;
using paddle::tape::SGD;
using paddle::tape::Adam;
using paddle::tape::accuracy;
using paddle::tape::mean;
using paddle::tape::softmax;
using paddle::tape::cross_entropy;
using paddle::tape::reshape;
using paddle::tape::split;
using paddle::tape::concat;
using paddle::tape::fill_constant;
using paddle::tape::lstm_step;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::OptimizableParameters;
using paddle::tape::BackwardAndUpdate;
using paddle::tape::ParameterCollection;
using paddle::tape::ParameterHandle;
using paddle::tape::GlobalParameterCollection;
using paddle::tape::RunOperator;

using paddle::tape::ReorderAndPad;
using paddle::tape::CreateRecordioFileReader;
using paddle::tape::CreateBatchReader;
using paddle::tape::CreateDoubleBufferReader;
using paddle::tape::ReadNext;
using paddle::tape::ResetReader;

TEST(NMT, TestTrainCPU) {
  auto place = paddle::platform::CPUPlace();
  reset_global_tape(place);

  const int batch_size = 2;
  const int word_dim = 32;
  const int dict_size = 30000;
  const int hidden_size = 32;
  LOG(INFO) << "Batch size is " << batch_size << std::endl;

  std::string save_model_path = "/tmp/NMT_model/";
  std::string train_file = "/tmp/wmt14_train.recordio";
  auto train_reader = CreateRecordioFileReader(
      train_file, {-1, 1, -1, 1, -1, 1}, {2, 2, 2}, {1, 1, 1});
  train_reader = CreateBatchReader(train_reader, batch_size);

  auto nmt_train = ReadNext(train_reader, true);
  LOG(INFO) << nmt_train.size() << std::endl;
  LOG(INFO) << nmt_train[0]->Value() << std::endl;
  LOG(INFO) << nmt_train[1]->Value() << std::endl;
  LOG(INFO) << nmt_train[2]->Value() << std::endl;

  // If look up table find this word idx, if will insert a zero vector in the
  // result
  // also it will bypass this idx when doing backward calculation
  int padding_idx = -2;
  auto src_idx = ReorderAndPad(nmt_train[0], padding_idx);
  auto trg_idx = ReorderAndPad(nmt_train[1], padding_idx);
  auto trg_next_idx = ReorderAndPad(nmt_train[2], padding_idx);

  LOG(INFO) << "After reorder";
  LOG(INFO) << src_idx->Value() << std::endl;
  LOG(INFO) << trg_idx->Value() << std::endl;
  LOG(INFO) << trg_next_idx->Value() << std::endl;

  Embedding embed1(dict_size, word_dim);
  auto output = embed1(src_idx);
  LOG(INFO) << output->Value();

  auto reshaped = reshape(output, {-1, batch_size, word_dim}, true);
  LOG(INFO) << reshaped->Value();
  auto temp = split(reshaped);
  std::vector<VariableHandle> steps;
  for (auto in : temp) {
    steps.emplace_back(reshape(in, {batch_size, word_dim}, true));
  }

  // input is of shape (1, batch_size, word_dim)
  Linear fc1({word_dim, hidden_size}, 4 * hidden_size);
  std::vector<VariableHandle> cell_states;
  std::vector<VariableHandle> output_states;

  VariableHandle init_cell = fill_constant({batch_size, hidden_size}, 0.0f);
  VariableHandle init_hidden = fill_constant({batch_size, hidden_size}, 0.0f);

  cell_states.emplace_back(init_cell);
  output_states.emplace_back(init_hidden);

  for (auto step : steps) {
    auto lstm_step_input = fc1({step, output_states.back()});
    std::vector<VariableHandle> outputs =
        lstm_step(lstm_step_input, cell_states.back());
    cell_states.emplace_back(outputs[0]);
    output_states.emplace_back(outputs[1]);
  }

  get_global_tape().Forward();
  LOG(INFO) << "cell states size " << cell_states.size() << "output states size"
            << output_states.size();

  for (int i = 0; i < cell_states.size(); ++i) {
    LOG(INFO) << cell_states[i]->Value();
    LOG(INFO) << output_states[i]->Value();
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
