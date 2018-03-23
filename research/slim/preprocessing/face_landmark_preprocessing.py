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


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
#    bbox_x = bbox[3]-bbox[1]
#    bbox_y = bbox[2]-bbox[0]
#    expand_ratio = 2.0
#    bbox_begin_expand = tf.constant(expand_ratio/4)
#    bbox_size_expand = tf.constant(expand_ratio/2)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.expand_dims(tf.expand_dims(bbox,0),0),
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

#    bbox_begin = tf.gather(bbox, [0,1]) + tf.stack([tf.random_uniform([1], minval=-bbox_begin_expand*bbox_y, maxval=0), tf.random_uniform([1], minval=-bbox_begin_expand*bbox_x, maxval=0)])
#    bbox_size = tf.stack([bbox_y+tf.random_uniform([1], minval=0, maxval=bbox_size_expand*bbox_y), bbox_x+tf.random_uniform([1], minval=0, maxval=bbox_size_expand*bbox_x)])

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

def expand_bounding_box_crop_resize(image, height, width, bbox, expand_ratio=1.8, scope=None):
	with tf.name_scope(scope, 'expand_bounding_box_crop_resize', [image, height, width, bbox]):
		#expand bbox randomly
		bbox_size = tf.gather(bbox, [2,3]) - tf.gather(bbox, [0,1])
		rd_expand = tf.reshape(tf.stack([tf.random_uniform([1], minval=-bbox_size[0]*expand_ratio/4, maxval=0),
							tf.random_uniform([1], minval=-bbox_size[1]*expand_ratio/4, maxval=0),
							tf.random_uniform([1], minval=0, maxval=bbox_size[0]*expand_ratio/4),
							tf.random_uniform([1], minval=0, maxval=bbox_size[1]*expand_ratio/4)]), [-1])
		expand_bbox = bbox + rd_expand
		cropped_image = tf.image.crop_and_resize(tf.expand_dims(image, 0),
												tf.expand_dims(expand_bbox, 0),
												[0],
												tf.constant([height,width], dtype=tf.int32),
												extrapolation_value=0)[-1] #remove batch
		return cropped_image, expand_bbox

def preprocess_for_train(image, height, width, bbox, landmarks,
                         min_object_covered=0.6,
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
    landmarks: 1-D float Tensor of landmarks' coordinates, the coordinates are arranged
      as [y1, x1, y2, x2, ..., y68, x68].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
    add_image_summaries: Enable image summaries.
  Returns:
    3-D float Tensor of distorted image used for training with range [0, 255].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox, landmarks]):
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

    distorted_image, distorted_bbox = expand_bounding_box_crop_resize(image, height, width, bbox)#distorted_bounding_box_crop(image, bbox, min_object_covered=min_object_covered)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), tf.expand_dims(tf.expand_dims((distorted_bbox), 0), 0))
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    '''num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)'''
    landmarks.set_shape([136]) #TODO: allow non-68 landmarks
    
    #convert landmarks to [0,1]
    #distorted_bbox = tf.reshape(distorted_bbox, [-1])
    #distorted_landmarks = tf.reshape(landmarks,[2,-1])
    bb_begin = tf.tile(tf.gather(distorted_bbox, [0,1]), tf.div(tf.shape(landmarks),tf.constant(2)))
    bb_size = tf.tile(tf.abs(tf.gather(distorted_bbox, [2,3]) - tf.gather(distorted_bbox, [0,1])), tf.div(tf.shape(landmarks),tf.constant(2)))
    distorted_landmarks = (landmarks - bb_begin) / bb_size
    #distorted_landmarks = tf.reshape(distorted_landmarks,[-1])


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
                       
    #draw landmarks
    if add_image_summaries:
      distorted_image_with_lm = draw_landmarks(distorted_image, distorted_landmarks)
      tf.summary.image('final_distorted_image_with_landmarks',distorted_image_with_lm)
    
    #distorted_image = tf.subtract(distorted_image, 0.5)
    #distorted_image = tf.multiply(distorted_image, 2.0)
    
    #change range to [0.0, 255.0] for quantization
    #distorted_image = tf.multiply(distorted_image, 255.0)
    #distorted_landmarks = tf.multiply(distorted_landmarks, 255.0)
    return distorted_image, distorted_landmarks

def preprocess_for_eval(image, height, width, bbox, landmarks,
                         min_object_covered=0.6,
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
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox.set_shape([4])

    distorted_image, distorted_bbox = expand_bounding_box_crop_resize(image, height, width, bbox)#distorted_bounding_box_crop(image, bbox, min_object_covered=min_object_covered)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), tf.expand_dims(tf.expand_dims((distorted_bbox), 0), 0))
    if add_image_summaries:
      tf.summary.image('images_with_distorted_bounding_box',
                       image_with_distorted_box)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    '''num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=num_resize_cases)'''
    landmarks.set_shape([136]) #TODO: allow non-68 landmarks
    
    #convert landmarks to [0,1]
    #distorted_bbox = tf.reshape(distorted_bbox, [-1])
    #distorted_landmarks = tf.reshape(landmarks,[2,-1])
    bb_begin = tf.tile(tf.gather(distorted_bbox, [0,1]), tf.div(tf.shape(landmarks),tf.constant(2)))
    bb_size = tf.tile(tf.abs(tf.gather(distorted_bbox, [2,3]) - tf.gather(distorted_bbox, [0,1])), tf.div(tf.shape(landmarks),tf.constant(2)))
    distorted_landmarks = (landmarks - bb_begin) / bb_size
    #distorted_landmarks = tf.reshape(distorted_landmarks,[-1])

    return distorted_image, distorted_landmarks


def preprocess_image(image, height, width, bbox, landmarks,
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
    return preprocess_for_train(image, height, width, bbox, landmarks, fast_mode,
                                add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(image, height, width, bbox, landmarks)
