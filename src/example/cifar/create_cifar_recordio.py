#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.fluid as fluid
import paddle.dataset.cifar as cifar


def create_cifar_recordio_files(batch_size=32, place=fluid.CPUPlace()):
    # Convert cifar training set to recordio files
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        reader = paddle.batch(cifar.train10(), batch_size=batch_size)
        feeder = fluid.DataFeeder(
            feed_list=[  # order is image and label
                fluid.layers.data(
                    name='image', shape=[3, 32, 32], dtype='float32'),
                fluid.layers.data(
                    name='label', shape=[1], dtype='int64'),
            ],
            place=place)
        filename = '/tmp/cifar10_train_' + str(batch_size) + "_" + str(
            place) + '.recordio'
        fluid.recordio_writer.convert_reader_to_recordio_file(filename, reader,
                                                              feeder)

    # Convert cifar testing set to recordio files
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        reader = paddle.batch(cifar.test10(), batch_size=batch_size)
        feeder = fluid.DataFeeder(
            feed_list=[  # order is image and label
                fluid.layers.data(
                    name='image', shape=[3, 32, 32], dtype='float32'),
                fluid.layers.data(
                    name='label', shape=[1], dtype='int64'),
            ],
            place=place)
        filename = '/tmp/cifar10_test_' + str(batch_size) + "_" + str(
            place) + '.recordio'
        fluid.recordio_writer.convert_reader_to_recordio_file(filename, reader,
                                                              feeder)


if __name__ == "__main__":
    create_cifar_recordio_files(64, fluid.CUDAPlace(1))
    create_cifar_recordio_files(64, fluid.CUDAPlace(1))