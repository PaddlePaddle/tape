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
import paddle.dataset.wmt14 as wmt14


def create_wmt14_recordio_files():
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        reader = paddle.batch(wmt14.train(30000), batch_size=1)
        feeder = fluid.DataFeeder(
            feed_list=[
                fluid.layers.data(
                    name='src_word_id', shape=[1], dtype='int64', lod_level=1),
                fluid.layers.data(
                    name='target_language_word',
                    shape=[1],
                    dtype='int64',
                    lod_level=1), fluid.layers.data(
                        name='target_language_next_word',
                        shape=[1],
                        dtype='int64',
                        lod_level=1)
            ],
            place=fluid.CPUPlace())
        fluid.recordio_writer.convert_reader_to_recordio_file(
            '/tmp/wmt14_train.recordio', reader, feeder)


if __name__ == "__main__":
    create_wmt14_recordio_files()
