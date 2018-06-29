# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif(APPLE)

CheckCompilerCXX11Flag()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Some Paddle Operators uses OMP, e.g.
# https://github.com/PaddlePaddle/Paddle/blob/d00a0436b1c562060b49aa2981be094d78bcbdf5/paddle/fluid/operators/top_k_op.h#L58-L60
find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Include cuda and cudnn
if(WITH_GPU)
    add_definitions(-DPADDLE_WITH_CUDA)
    include_directories(${CUDNN_INCLUDE_DIR})
    include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

include_directories(${CMAKE_SOURCE_DIR}/paddle) # for header files in paddle/
include_directories(${CMAKE_BINARY_DIR}/paddle) # for *.pb.h
include_directories(${CMAKE_BINARY_DIR}/third_party/install/glog/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/install/gflags/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/install/protobuf/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/install/gtest/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/boost/src/extern_boost/boost_1_41_0/) # hack in paddle/cmake/external/boost.cmake
include_directories(${CMAKE_BINARY_DIR}/third_party/eigen3/src/extern_eigen3)
# hack in paddle/cmake/external/eigen.cmake

cc_library(tape SRCS variable.cc tape.cc backward.cc DEPS ${GLOB_OP_LIB})
cc_library(tape_parameter SRCS parameter.cc DEPS tape)
cc_library(tape_function SRCS function.cc DEPS ${GLOB_OP_LIB} tape tape_parameter)

cc_test(test_parameter
        SRCS test_parameter.cc
        DEPS tape_parameter)
cc_test(test_tape
        SRCS test_tape.cc
        DEPS tape tape_function)
cc_test(test_backward
        SRCS test_backward.cc
        DEPS tape tape_function)

add_subdirectory(example)
