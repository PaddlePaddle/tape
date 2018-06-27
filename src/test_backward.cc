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

#include <vector>

#include "gtest/gtest.h"
#include "src/function.h"

using paddle::tape::reset_global_tape;
using paddle::tape::get_global_tape;

using paddle::framework::LoDTensor;

using paddle::tape::add;
using paddle::tape::mean;
using paddle::tape::Variable;
using paddle::tape::VariableHandle;

VariableHandle fill_with_vector(std::vector<float> data) {
  VariableHandle var(new Variable("fill"));
  auto* tensor = var->GetMutable<LoDTensor>();
  tensor->Resize({static_cast<int64_t>(data.size())});
  LOG(INFO) << tensor->dims();
  auto* ptr = tensor->mutable_data<float>(paddle::platform::CPUPlace());
  for (size_t i = 0; i < data.size(); ++i) {
    ptr[i] = data[i];
  }
  return var;
}

/*
 * y = op(x)
 * z = op(x)
 * loss = y + z
 */
TEST(Backward, TestMultipleAssignment) {
  reset_global_tape();

  auto x = fill_with_vector({42});
  auto y = mean(x);
  auto z = mean(x);
  auto loss = add(y, z);

  get_global_tape().Backward(loss);

  LOG(INFO) << x->Value();
  LOG(INFO) << x->Grad()->Value();
  PADDLE_ENFORCE_EQ(x->Grad()->Get<LoDTensor>().data<float>()[0], 2.0);
}

/*
 * loss = x + x
 */
TEST(Backward, TestInplaceSum) {
  reset_global_tape();

  auto x = fill_with_vector({42});
  auto loss = add(x, x);

  get_global_tape().Backward(loss);

  PADDLE_ENFORCE_EQ(x->Grad()->Get<LoDTensor>().data<float>()[0], 2.0);
}

/*
 * y = op(x)  // y@grad is not initialized
 * loss = op(z)
 */
TEST(Backward, TestEmptyGrad) {
  reset_global_tape();
  auto x = fill_with_vector({42});
  auto y = mean(x);

  auto z = fill_with_vector({42});
  auto loss = mean(z);

  get_global_tape().Backward(loss);

  PADDLE_ENFORCE_EQ(x->Grad()->Get<LoDTensor>().data<float>()[0], 0.0);
  PADDLE_ENFORCE_EQ(y->Grad()->Get<LoDTensor>().data<float>()[0], 0.0);
  PADDLE_ENFORCE_EQ(z->Grad()->Get<LoDTensor>().data<float>()[0], 1.0);
}

/*
 * vector<> v
 * for i in dim(x, 0):
 *   y = x.Slice(i)
 *   out = linear(y)
 *   v.push_back(out)
 * loss = v.back()
 */
TEST(Backward, TestForLoop) { reset_global_tape(); }

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
