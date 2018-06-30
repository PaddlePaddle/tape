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

#include <gtest/gtest.h>
#include <vector>

#include "src/parameter.h"

using paddle::tape::ParameterCollection;

TEST(ParameterCollection, TestAddParameter) {
  ParameterCollection pc;
  pc.AddParameter("w", "fill_constant", {{"shape", std::vector<int>{3}}});
  PADDLE_ENFORCE_EQ(pc.OptimizableParameters().size(), 1);

  auto param = pc.OptimizableParameters()[0];
  for (int i = 0; i < 3; ++i) {
    PADDLE_ENFORCE_EQ(
        param->Get<paddle::framework::LoDTensor>().data<float>()[i], 0.0);
  }
}

TEST(ParameterCollection, TestSaveAllParameters) {
  std::string file_path = "/tmp/test_parameter_save/";
  ParameterCollection pc;
  pc.AddParameter("w", "fill_constant", {{"shape", std::vector<int>{3}}});
  pc.SaveAllParameters(file_path);

  ParameterCollection loaded_pc(file_path);
  PADDLE_ENFORCE_EQ(loaded_pc.OptimizableParameters().size(), 1);
  auto param = loaded_pc.OptimizableParameters()[0];
  for (int i = 0; i < 3; ++i) {
    PADDLE_ENFORCE_EQ(
        param->Get<paddle::framework::LoDTensor>().data<float>()[i], 0.0);
  }

  PADDLE_ENFORCE_EQ(system(std::string("rm -r " + file_path).c_str()), 0);
}

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
