# coding=utf-8
# Copyright 2018 David Mack
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



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from absl import logging
from absl import flags
FLAGS = flags.FLAGS

from compare_gan.metrics import eval_task
import tensorflow as tf
import tensorflow_gan as tfgan
import imageio


flags.DEFINE_string(
    "example_dir", "./examples",
    "Where to save generated image examples")


flags.DEFINE_integer(
    "example_count", 40,
    "How many generated image examples to save")

class SaveExamplesTask(eval_task.EvalTask):
  """Quick and dirty image saver."""

  _LABEL = "save_examples"

  def run_after_session(self, fake_dset, real_dest):

    tf.io.gfile.makedirs(FLAGS.example_dir)
    
    for i in range(min(fake_dset.shape[0], FLAGS.example_count)):
      imageio.imwrite(os.path.join(FLAGS.example_dir, '%03d.png' % i), fake_dset[i])

    return {self._LABEL: FLAGS.example_count}
