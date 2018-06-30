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

#include "src/tape.h"

#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace tape {

std::ostream& operator<<(std::ostream& os, const Variable& var) {
  LOG(INFO) << "Printing " << var.Name();
  if (var.Var().IsType<framework::LoDTensor>()) {
    os << var.Get<framework::LoDTensor>();
  } else if (var.Var().IsType<framework::LoDTensorArray>()) {
    framework::LoDTensorArray array = var.Get<framework::LoDTensorArray>();
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

VariableHandle Variable::CopyToCPU() const {
  auto place = this->Get<framework::LoDTensor>().place();
  auto context = platform::DeviceContextPool::Instance().Get(place);
  VariableHandle cpu_copy(new Variable("temp"));
  framework::TensorCopy(this->Get<framework::LoDTensor>(),
                        platform::CPUPlace(),
                        *context,
                        cpu_copy->GetMutable<framework::LoDTensor>());
  context->Wait();
  return cpu_copy;
}

const Variable& Variable::Value() {
  get_global_tape().Forward();
  auto place = this->Get<framework::LoDTensor>().place();
  PADDLE_ENFORCE(platform::is_same_place(place, get_global_tape().Place()),
                 "Data place should match tape place");
  if (platform::is_gpu_place(place)) {
    auto context = platform::DeviceContextPool::Instance().Get(place);
    context->Wait();
  }
  return *this;
}

}  // namespace tape
}  // namespace paddle
