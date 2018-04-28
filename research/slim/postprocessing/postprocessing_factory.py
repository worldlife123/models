# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from postprocessing import identity_postprocessing
from postprocessing import face_mm_postprocessing

slim = tf.contrib.slim


def get_postprocessing(name):
  """Returns postprocessing_fn(input, **kwargs).

  Args:
    name: The name of the postprocessing function.

  Returns:
    postprocessing_fn: A function that postprocesses a single output (pre-batch).
      It has the following signature:
        output = postprocessing_fn(input, ...).

  Raises:
    ValueError: If Postprocessing `name` is not recognized.
  """
  postprocessing_fn_map = {
      'identity': identity_postprocessing,
      
      'cifarnet': identity_postprocessing,
      'inception': identity_postprocessing,
      'inception_v1': identity_postprocessing,
      'inception_v2': identity_postprocessing,
      'inception_v3': identity_postprocessing,
      'inception_v4': identity_postprocessing,
      'inception_resnet_v2': identity_postprocessing,
      'lenet': identity_postprocessing,
      'mobilenet_v1': identity_postprocessing,
      'nasnet_mobile': identity_postprocessing,
      'nasnet_large': identity_postprocessing,
      'resnet_v1_50': identity_postprocessing,
      'resnet_v1_101': identity_postprocessing,
      'resnet_v1_152': identity_postprocessing,
      'resnet_v1_200': identity_postprocessing,
      'resnet_v2_50': identity_postprocessing,
      'resnet_v2_101': identity_postprocessing,
      'resnet_v2_152': identity_postprocessing,
      'resnet_v2_200': identity_postprocessing,
      'vgg': identity_postprocessing,
      'vgg_a': identity_postprocessing,
      'vgg_16': identity_postprocessing,
      'vgg_19': identity_postprocessing,
      
      'face_landmark': identity_postprocessing,
      'face_mm': face_mm_postprocessing,
  }

  if name not in postprocessing_fn_map:
    raise ValueError('Postprocessing name [%s] was not recognized' % name)

  def postprocessing_fn(input, **kwargs):
    return postprocessing_fn_map[name].postprocess(input, **kwargs)

  return postprocessing_fn
