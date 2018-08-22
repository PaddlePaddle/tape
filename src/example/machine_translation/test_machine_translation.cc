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
using paddle::tape::gather;
using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;
using paddle::tape::OptimizableParameters;
using paddle::tape::BackwardAndUpdate;
using paddle::tape::ParameterCollection;
using paddle::tape::ParameterHandle;
using paddle::tape::GlobalParameterCollection;
using paddle::tape::RunOperator;

using paddle::tape::ReorderAndPad;
using paddle::tape::GetSeqLens;
using paddle::tape::CreateRecordioFileReader;
using paddle::tape::CreateBatchReader;
using paddle::tape::CreateDoubleBufferReader;
using paddle::tape::ReadNext;
using paddle::tape::ResetReader;

enum RnnType { LSTM, GRU, VANILLA };

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
  std::string test_file = "/tmp/wmt14_test.recordio";
  auto train_reader = CreateRecordioFileReader(
      train_file, {-1, 1, -1, 1, -1, 1}, {2, 2, 2}, {1, 1, 1});
  auto test_reader = CreateRecordioFileReader(
      test_file, {-1, 1, -1, 1, -1, 1}, {2, 2, 2}, {1, 1, 1});
  train_reader = CreateBatchReader(train_reader, batch_size);
  test_reader = CreateBatchReader(test_reader, batch_size);

  // Used as lookup table for both source and target word idx.
  Embedding embed(dict_size, word_dim);
  // Output of this fc serves as the input to the lstm unit update in encoder.
  Linear encoder_fc({word_dim, hidden_size}, 4 * hidden_size);
  // decoder_train
  Linear decoder_fc({word_dim, hidden_size}, 4 * hidden_size);
  Linear decoder_softmax({hidden_size}, dict_size, "softmax");
  // Linear decoder_fc1({word_dim, hidden_size}, hidden_size, "tanh");
  // Linear decoder_fc2({hidden_size}, dict_size, "softmax");

  Adam adam(0.001);

  auto encoder = [&](VariableHandle input) -> VariableHandle {
    // Get seq lens info for both src idx and target idx
    std::vector<int> seq_lens = GetSeqLens(input);
    // If the word idx is equal to the padding idx, the look up table op will
    // insert
    // a zero vector in the result. Also it will bypass this idx when doing
    // backprop.
    int padding_idx = -2;
    // Reorder the batch of word idx sequence to the shape of [max_seq_len *
    // batch_size, 1]
    // where the same time step of different batches are placed together
    // followed by
    // the batch of word idx in the next time step, padding padding_idx where
    // needed.
    auto src_idx = ReorderAndPad(input, padding_idx);

    auto src_vec = embed(src_idx);
    // reshape src_vec to [max_seq_len, batch_size, word_dim]
    // then split to max_seq_len number of time step vector
    auto steps = split(reshape(src_vec, {-1, batch_size, word_dim}, true));
    std::vector<VariableHandle> src_steps;
    for (auto step : steps) {
      // each step is of shape [1, batch_size, word_dim]
      src_steps.emplace_back(reshape(step, {batch_size, word_dim}, true));
    }

    std::vector<VariableHandle> cell_states;
    std::vector<VariableHandle> output_states;

    VariableHandle init_cell = fill_constant({batch_size, hidden_size}, 0.0f);
    VariableHandle init_hidden = fill_constant({batch_size, hidden_size}, 0.0f);

    cell_states.emplace_back(init_cell);
    output_states.emplace_back(init_hidden);

    for (auto step : src_steps) {
      auto step_input = encoder_fc({step, output_states.back()});
      std::vector<VariableHandle> outputs =
          lstm_step(step_input, cell_states.back());
      cell_states.emplace_back(outputs[0]);
      output_states.emplace_back(outputs[1]);
    }

    std::vector<VariableHandle> encoder_output;
    for (int i = 0; i < batch_size; ++i) {
      encoder_output.emplace_back(gather(output_states[seq_lens[i]], {i}));
    }

    return concat(encoder_output);
  };

  auto decoder_train = [&](VariableHandle context,
                           VariableHandle input) -> VariableHandle {
    std::vector<int> seq_lens = GetSeqLens(input);
    auto tgt_idx = ReorderAndPad(input, -2);

    std::vector<VariableHandle> decoder_states;
    std::vector<VariableHandle> decoder_hidden;
    std::vector<VariableHandle> decoder_outputs;
    decoder_states.emplace_back(context);
    decoder_hidden.emplace_back(fill_constant({batch_size, hidden_size}, 0.0f));

    auto tgt_vec = embed(tgt_idx);
    auto steps = split(reshape(tgt_vec, {-1, batch_size, word_dim}, true));
    std::vector<VariableHandle> decoder_steps;
    for (auto step : steps) {
      decoder_steps.emplace_back(reshape(step, {batch_size, word_dim}, true));
    }

    for (auto step : decoder_steps) {
      auto step_input = decoder_fc({step, decoder_hidden.back()});
      std::vector<VariableHandle> outputs =
          lstm_step(step_input, decoder_states.back());
      decoder_states.emplace_back(outputs[0]);
      decoder_hidden.emplace_back(outputs[1]);
      decoder_outputs.emplace_back(decoder_softmax({outputs[1]}));
      //  auto current_state = decoder_fc1({step, decoder_states.back()});
      //  decoder_states.emplace_back(current_state);
      //  decoder_outputs.emplace_back(decoder_fc2({current_state}));
    }

    std::vector<VariableHandle> decoder_result;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_lens[i]; ++j) {
        decoder_result.emplace_back(gather(decoder_outputs[j], {i}));
      }
    }

    return concat(decoder_result);
  };

  int total_steps = 50000;
  int print_steps = 200;
  int test_steps = 100;
  float threshold = 6.6f;
  bool model_saved = false;

  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < total_steps; ++i) {
    // LOG(INFO) << "Train step #" << i;

    reset_global_tape(place);
    auto nmt_train = ReadNext(train_reader, true);
    // LOG(INFO) << nmt_train[0]->Value();

    auto encoder_out = encoder(nmt_train[0]);
    LOG(INFO) << encoder_out->Value();
    auto rnn_out = decoder_train(encoder_out, nmt_train[1]);
    LOG(INFO) << rnn_out->Value();

    PADDLE_ENFORCE(false);

    auto trg_next_idx = nmt_train[2];
    VariableHandle cost = cross_entropy(rnn_out, trg_next_idx);
    VariableHandle loss = mean(cost);
    // LOG(INFO) << loss->Value();

    BackwardAndUpdate(loss, &adam);

    if (i % print_steps == 0) {
      std::vector<float> losses;
      ResetReader(test_reader);

      LOG(INFO) << "starting testing";
      for (int i = 0; i < test_steps; ++i) {
        reset_global_tape(place);

        auto nmt_test = ReadNext(test_reader, true);
        // LOG(INFO) << nmt_test[0]->Value();
        auto test_encoder_out = encoder(nmt_test[0]);
        auto test_rnn_out = decoder_train(test_encoder_out, nmt_test[1]);

        VariableHandle loss = mean(cross_entropy(test_rnn_out, nmt_test[2]));
        get_global_tape().Forward();

        losses.push_back(loss->FetchValue()
                             ->Get<paddle::framework::LoDTensor>()
                             .data<float>()[0]);
      }

      float avg_loss =
          std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
      LOG(INFO) << "Batch #" << i << ", test set avg loss is " << avg_loss;

      if (avg_loss < threshold) {
        LOG(INFO) << "Meets target avg loss, stop training and save parameters";
        GlobalParameterCollection().SaveAllParameters(save_model_path);
        model_saved = true;
        break;
      }
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  LOG(INFO) << "Total wall clock time is " << elapsed_time.count()
            << " seconds";

  if (!model_saved) {
    return;
  }

  // Inference using test set
  LOG(INFO) << "Start inferencing and load parameters";
  ParameterCollection loaded_pc(save_model_path);

  // Reconstruct layers by loading the saved parameters
  Embedding inf_embed(loaded_pc.LookUp(embed.ParamNames()));
  Linear inf_encoder_fc(loaded_pc.LookUp(encoder_fc.ParamNames()),
                        encoder_fc.ActName());
  Linear inf_decoder_fc(loaded_pc.LookUp(decoder_fc.ParamNames()),
                        decoder_fc.ActName());
  Linear inf_decoder_softmax(loaded_pc.LookUp(decoder_softmax.ParamNames()),
                             decoder_softmax.ActName());

  auto test_encoder = [&](VariableHandle input) -> VariableHandle {
    std::vector<int> seq_lens = GetSeqLens(input);
    auto src_idx = ReorderAndPad(input, -2);
    auto src_vec = inf_embed(src_idx);
    auto steps = split(reshape(src_vec, {-1, batch_size, word_dim}, true));
    std::vector<VariableHandle> src_steps;
    for (auto step : steps) {
      src_steps.emplace_back(reshape(step, {batch_size, word_dim}, true));
    }

    std::vector<VariableHandle> cell_states;
    std::vector<VariableHandle> output_states;

    VariableHandle init_cell = fill_constant({batch_size, hidden_size}, 0.0f);
    VariableHandle init_hidden = fill_constant({batch_size, hidden_size}, 0.0f);

    cell_states.emplace_back(init_cell);
    output_states.emplace_back(init_hidden);

    for (auto step : src_steps) {
      auto step_input = inf_encoder_fc({step, output_states.back()});
      std::vector<VariableHandle> outputs =
          lstm_step(step_input, cell_states.back());
      cell_states.emplace_back(outputs[0]);
      output_states.emplace_back(outputs[1]);
    }

    std::vector<VariableHandle> encoder_output;
    for (int i = 0; i < batch_size; ++i) {
      encoder_output.emplace_back(gather(output_states[seq_lens[i]], {i}));
    }

    return concat(encoder_output);
  };

  auto test_decoder = [&](VariableHandle context,
                          VariableHandle input) -> VariableHandle {
    std::vector<int> seq_lens = GetSeqLens(input);
    auto tgt_idx = ReorderAndPad(input, -2);

    std::vector<VariableHandle> decoder_states;
    std::vector<VariableHandle> decoder_hidden;
    std::vector<VariableHandle> decoder_outputs;
    decoder_states.emplace_back(context);
    decoder_hidden.emplace_back(fill_constant({batch_size, hidden_size}, 0.0f));

    auto tgt_vec = inf_embed(tgt_idx);
    auto steps = split(reshape(tgt_vec, {-1, batch_size, word_dim}, true));
    std::vector<VariableHandle> decoder_steps;
    for (auto step : steps) {
      decoder_steps.emplace_back(reshape(step, {batch_size, word_dim}, true));
    }

    for (auto step : decoder_steps) {
      auto step_input = inf_decoder_fc({step, decoder_hidden.back()});
      std::vector<VariableHandle> outputs =
          lstm_step(step_input, decoder_states.back());
      decoder_states.emplace_back(outputs[0]);
      decoder_hidden.emplace_back(outputs[1]);
      decoder_outputs.emplace_back(inf_decoder_softmax({outputs[1]}));
    }

    std::vector<VariableHandle> decoder_result;
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_lens[i]; ++j) {
        decoder_result.emplace_back(gather(decoder_outputs[j], {i}));
      }
    }

    return concat(decoder_result);
  };

  std::vector<float> losses;
  std::vector<float> accuracies;
  ResetReader(test_reader);

  for (int i = 0; i < test_steps; ++i) {
    reset_global_tape(place);

    auto nmt_test = ReadNext(test_reader, true);
    auto test_encoder_out = test_encoder(nmt_test[0]);
    auto test_rnn_out = test_decoder(test_encoder_out, nmt_test[1]);

    VariableHandle loss = mean(cross_entropy(test_rnn_out, nmt_test[2]));
    get_global_tape().Forward();

    losses.push_back(loss->FetchValue()
                         ->Get<paddle::framework::LoDTensor>()
                         .data<float>()[0]);
  }

  float avg_loss =
      std::accumulate(losses.begin(), losses.end(), 0.0f) / losses.size();
  LOG(INFO) << "Loaded Model and inference on test set result: Avg loss is "
            << avg_loss;

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
