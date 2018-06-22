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

#include "src/variable.h"

#include <paddle/paddle/fluid/framework/lod_tensor_array.h>
#include "src/tape.h"

namespace paddle {
namespace tape {

std::ostream& operator<<(std::ostream& os, const Variable& var) {
  LOG(INFO) << "Printing " << var.Name();
  if (var.Var().IsType<framework::LoDTensor>()) {
    os << var.Var().Get<framework::LoDTensor>();
  } else if (var.Var().IsType<framework::LoDTensorArray>()) {
    framework::LoDTensorArray array =
        var.Var().Get<framework::LoDTensorArray>();
    for (size_t i = 0; i < array.size(); ++i) {
      os << "Printing lod_tensor #" << i << " in lod_tensor_array "
         << var.Name() << "\n";
      os << array[i] << "\n";
    }
  } else {
    PADDLE_THROW("Variable type is not in [LOD_TENSOR, LOD_TENSOR_ARRAY]");
  }
  return os;
}

const Variable& Variable::Value() {
  get_global_tape().Forward();
  return *this;
}

}  // namespace tape
}  // namespace paddle
