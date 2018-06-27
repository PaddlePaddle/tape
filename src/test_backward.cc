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
#include "src/function.h"

/*
 * y = op(x)
 * z = op(x)
 * loss = y + z
 */
TEST(Backward, TestMultipleAssignment) {}

/*
 * loss = x + x
 */
TEST(Backward, TestInplaceSum) {}

/*
 * y = op(x)  // y@grad is not initialized
 * loss = op(z)
 */
TEST(Backward, TestEmptyGrad) {}

/*
 * vector<> v
 * for i in dim(x, 0):
 *   y = x.Slice(i)
 *   out = linear(y)
 *   v.push_back(out)
 * loss = v.back()
 */
TEST(Backward, TestForLoop) {}

int main(int argc, char** argv) {
  std::vector<paddle::platform::Place> places;
  places.emplace_back(paddle::platform::CPUPlace());
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
