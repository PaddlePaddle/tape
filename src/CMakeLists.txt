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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}) # for header files in src/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/paddle) # for header files in paddle/
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../paddle")

# the following lines fix include dependencies
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(${BOOST_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIR})
include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories("${CMAKE_CURRENT_BINARY_DIR}/../third_party/eigen3/src/extern_eigen3")

set (CMAKE_CXX_STANDARD 11)
cc_library(tape_variable SRCS variable.cc DEPS operator proto_desc memory lod_tensor executor init)
cc_library(tape SRCS tape.cc DEPS ${GLOB_OP_LIB} tape_variable)

cc_test(test_tape
        SRCS test_tape.cc
        DEPS tape tape_variable)
