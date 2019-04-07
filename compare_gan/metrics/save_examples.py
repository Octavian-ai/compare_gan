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
import math
import numpy as np

flags.DEFINE_string(
    "example_dir", "./examples",
    "Where to save generated image examples")


flags.DEFINE_integer(
    "example_count", 100,
    "How many generated image examples to save")

class SaveExamplesTask():
  """Quick and dirty image saver."""

  _LABEL = "save_examples"




  def merge(self, images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


  def run_after_session(self, fake_dset, real_dest, step, force_label=None):

    tf.io.gfile.makedirs(os.path.join(FLAGS.example_dir, step))
    
    n_images = fake_dset.images.shape[0]

    if force_label is not None: 
      label_str = "force_label_" + str(force_label)
    else:
      label_str = "all_labels"
    
    for i in range(min(n_images, FLAGS.example_count)):
      filename = os.path.join(FLAGS.example_dir, step,  label_str + '_%03d.png' % i)
      with tf.io.gfile.GFile(filename, 'w') as file:
        imageio.imwrite(file, fake_dset.images[i], format='png')

    grid_size = (int(math.sqrt(n_images))+1, int(math.sqrt(n_images)))
    grid = self.merge(fake_dset.images, grid_size)

    filename = os.path.join(FLAGS.example_dir, step + '_' + label_str + '_grid.png')
    with tf.io.gfile.GFile(filename, 'w') as file:
      imageio.imwrite(file, grid, format='png')

    return {self._LABEL: FLAGS.example_count}



