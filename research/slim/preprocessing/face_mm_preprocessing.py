# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]

def draw_landmarks(image, landmarks, scope=None):
  """Draw landmarks on an image. Landmarks should be an 1D-Tensor with format [x1,y1,x2,y2...]

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    landmarks: 1-D Tensor with format [x1,y1,x2,y2...], in which x,y in [0, 1] 
    scope: Optional scope for name_scope.
  Returns:
    4-D Tensor color-distorted image on range [0, 1]
  """
  with tf.name_scope(scope, 'draw_landmarks', [image, landmarks]):
    landmarks_2d = tf.transpose(tf.reshape(landmarks,[-1,2]))
    pt2bboxes = tf.stack([landmarks_2d[0]-0.01, landmarks_2d[1]-0.01, landmarks_2d[0]+0.01, landmarks_2d[1]+0.01], axis=1)
    return tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.expand_dims(pt2bboxes,0), scope)

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def expand_bounding_box_crop_resize(image, height, width, bbox, expand_range=(1.0, 2.0), scope=None):
    with tf.name_scope(scope, 'expand_bounding_box_crop_resize', [image, height, width, bbox]):
        #expand bbox randomly
        bbox_size = tf.gather(bbox, [2,3]) - tf.gather(bbox, [0,1])
        expand_min, expand_max = (expand_range[0]-1)/2, (expand_range[1]-1)/2
        rd_expand = tf.reshape(tf.stack([tf.random_uniform([1], minval=-bbox_size[0]*expand_max, maxval=-bbox_size[0]*expand_min),
                                         tf.random_uniform([1], minval=-bbox_size[1]*expand_max, maxval=-bbox_size[1]*expand_min),
                                         tf.random_uniform([1], minval=bbox_size[0]*expand_min, maxval=bbox_size[0]*expand_max),
                                         tf.random_uniform([1], minval=bbox_size[1]*expand_min, maxval=bbox_size[1]*expand_max)]), [-1])
        expand_bbox = bbox + rd_expand
        cropped_image = tf.image.crop_and_resize(tf.expand_dims(image, 0),
                                                tf.expand_dims(expand_bbox, 0),
                                                [0],
                                                tf.constant([height,width], dtype=tf.int32),
                                                extrapolation_value=0)[-1] #remove batch
        return cropped_image, expand_bbox

def preprocess_for_train(image, height, width, bbox, shape, expression, pose,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 1-D float Tensor of bounding box, the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    shape: 1-D float Tensor of 3DMM shape coefficients.
    expression: 1-D float Tensor of 3DMM expression coefficients.
    pose: 1-D float Tensor of 3DMM pose coefficients.
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.
  Returns:
    3-D float Tensor of distorted image used for training with range [0, 255].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox, shape, expression, pose]):
    assert not (bbox is None or landmarks is None)
    
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      
    #convert bbox and landmarks to [0,1] coordinates
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  tf.expand_dims(tf.expand_dims(bbox, 0), 0))
    if add_image_summaries:
      tf.summary.image('image_with_bounding_boxes', image_with_box)

    bbox.set_shape([4])
    shape.set_shape([199]) 
    expression.set_shape([29])
    pose.set_shape([7])
    #TODO: unify all components
    
    label = tf.concat([shape, expression, pose], axis=0)

    distorted_image, distorted_bbox = expand_bounding_box_crop_resize(image, height, width, bbox)
    
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), tf.expand_dims(tf.expand_dims((distorted_bbox), 0), 0))
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)



    if add_image_summaries:
      tf.summary.image('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 1 or 4 ways to do it.
    num_distort_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=num_distort_cases)

    if add_image_summaries:
      tf.summary.image('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
                       
    #change range to [-1,1]
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)

    return distorted_image, label

def preprocess_for_eval(image, height, width, bbox, shape, expression, pose,
                         fast_mode=True,
                         scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width, bbox, shape, expression, pose]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox.set_shape([4])
    shape.set_shape([199]) 
    expression.set_shape([29])
    pose.set_shape([7])
    #TODO: unify all components
    
    label = tf.concat([shape, expression, pose], axis=0)

    distorted_image, distorted_bbox = expand_bounding_box_crop_resize(image, height, width, bbox)#distorted_bounding_box_crop(image, bbox, min_object_covered=min_object_covered)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), tf.expand_dims(tf.expand_dims((distorted_bbox), 0), 0))

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    
    #change range to [-1,1]
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)

    return distorted_image, label


def preprocess_image(image, height, width, bbox, shape, expression, pose,
                     is_training=False,
                     fast_mode=True,
                     add_image_summaries=True):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.
    add_image_summaries: Enable image summaries.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  if is_training:
    return preprocess_for_train(image, height, width, bbox, shape, expression, pose, fast_mode=fast_mode,
                                add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(image, height, width, bbox, shape, expression, pose)

