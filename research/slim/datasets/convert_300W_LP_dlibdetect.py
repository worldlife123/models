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
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import cv2
import dlib

import tensorflow as tf
import scipy.io as sio

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
#_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# The number of images in the validation set.
#_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 1

# Path to store event files for summary
_PATH_EVENTS = 'events'

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class ImageDrawer(object):
  """Helper class that provides TensorFlow image drawing utilities."""

  def __init__(self):
    # Initializes function that prepares data.
    self._image_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
    self._bbox_data = tf.placeholder(dtype=tf.float32, shape=[4])
    self._lm_data = tf.placeholder(dtype=tf.float32, shape=[136])
    self._draw_bbox = lambda name : tf.summary.image(name, 
                                       tf.image.draw_bounding_boxes(tf.expand_dims(self._image_data, 0),
                                                                    tf.expand_dims(tf.expand_dims(self._bbox_data, 0), 0)))
    landmarks_2d = tf.transpose(tf.reshape(self._lm_data,[-1,2]))
    pt2bboxes = tf.stack([landmarks_2d[0]-0.01, landmarks_2d[1]-0.01, landmarks_2d[0]+0.01, landmarks_2d[1]+0.01], axis=1)
    self._draw_lm = lambda name : tf.summary.image(name, 
                                     tf.image.draw_bounding_boxes(tf.expand_dims(self._image_data, 0), tf.expand_dims(pt2bboxes,0)))

  def draw_bbox(self, sess, image_data, bbox, name="image_with_bounding_box"):
    image = sess.run(self._draw_bbox(name),
                             feed_dict={self._image_data: image_data, self._bbox_data: bbox})
    return image

  def draw_lm(self, sess, image_data, lm, name="image_with_landmarks"):
    image = sess.run(self._draw_lm(name),
                     feed_dict={self._image_data: image_data, self._lm_data: lm})
    #assert len(image.shape) == 3
    #assert image.shape[2] == 3
    return image

def _face_detect(img_filename, ref_bbox):
    img = cv2.imread(img_filename)
    dets = dlib_detector(img, 0)
    for d in dets:
        det_bbox = [float(d.left())/img.shape[1],float(d.top())/img.shape[0],float(d.right())/img.shape[1],float(d.bottom())/img.shape[0]]
        if det_bbox[0]>ref_bbox[0] or det_bbox[1]>ref_bbox[1] or det_bbox[2]<ref_bbox[2] or det_bbox[3]<ref_bbox[3]: continue
        else: return det_bbox
    
    return None

def _get_filenames(dataset_dir, split_name):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
    split_name: The name of the dataset, either 'train' or 'validation'.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  assert split_name in ['train', 'validation']
  folder_dict = {
    'train' : ['LFPW', 'AFW', 'HELEN'],
    'validation' : ['IBUG']
  }

  photo_filenames = []
  label_filenames = []
  for directory in folder_dict[split_name]:
    full_directory = os.path.join(dataset_dir,directory)
    for filename in os.listdir(full_directory):
      if filename[-4:]==".jpg":
        photo_path_woext = os.path.join(full_directory, filename[:-4])
        photo_filenames.append(photo_path_woext)
        label_filenames.append(os.path.join(dataset_dir, "landmarks", directory, filename[:-4] + "_pts.mat"))

  return photo_filenames, label_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'landmark_300W_LP_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _convert_tfexample(image_data, image_format, height, width, facebbox, landmarks_2d, landmarks_3d, params):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': dataset_utils.bytes_feature(image_data),
      'image/format': dataset_utils.bytes_feature(image_format),
      'image/face/bbox': dataset_utils.float_feature(facebbox),
      'image/face/landmark_2d': dataset_utils.float_feature(landmarks_2d),
      'image/face/landmark_3d': dataset_utils.float_feature(landmarks_3d),
      'image/height': dataset_utils.int64_feature(height),
      'image/width': dataset_utils.int64_feature(width),
      'image/face/params/roi': dataset_utils.int64_feature(list(params['roi'].ravel())),
      'image/face/params/pose': dataset_utils.float_feature(list(params['Pose_Para'].ravel())),
      'image/face/params/color': dataset_utils.float_feature(list(params['Color_Para'].ravel())),
      'image/face/params/shape': dataset_utils.float_feature(list(params['Shape_Para'].ravel())),
      'image/face/params/texture': dataset_utils.float_feature(list(params['Tex_Para'].ravel())),
      'image/face/params/expression': dataset_utils.float_feature(list(params['Exp_Para'].ravel())),
      'image/face/params/illumination': dataset_utils.float_feature(list(params['Illum_Para'].ravel())),
  }))

def _convert_dataset(split_name, photo_filenames, label_filenames, dataset_dir, summary_writer=None):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    photo_filenames: A list of absolute paths of jpg images.
    label_filenames: A list of absolute paths of landmark .mat files.
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']
  assert len(photo_filenames)==len(label_filenames)

  num_per_shard = int(math.ceil(len(photo_filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()
    image_drawer = ImageDrawer()

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(photo_filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(photo_filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(photo_filenames[i]+".jpg", 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            
            params = sio.loadmat(photo_filenames[i]+".mat")
            
            mat = sio.loadmat(label_filenames[i])
            pt2d = mat['pts_2d'].T
            pt3d = mat['pts_3d'].T
            #convert to [0,1] and (y,x) for convenience
            new_pt2d = pt2d.copy()
            new_pt2d[0,:] = pt2d[1,:]/height
            new_pt2d[1,:] = pt2d[0,:]/width
            new_pt3d = pt3d.copy()
            new_pt3d[0,:] = pt3d[1,:]/height
            new_pt3d[1,:] = pt3d[0,:]/width
            x_min = min(new_pt2d[1,:])
            y_min = min(new_pt2d[0,:])
            x_max = max(new_pt2d[1,:])
            y_max = max(new_pt2d[0,:])
            
            bbox, landmarks_2d, landmarks_3d = [y_min, x_min, y_max, x_max], list(new_pt2d.T.ravel()), list(new_pt3d.T.ravel())
            
            
            
            if summary_writer:
              image = image_reader.decode_jpeg(sess, image_data)
              image_with_box = image_drawer.draw_bbox(sess, image, bbox)
              image_with_lm = image_drawer.draw_lm(sess, image, landmarks_2d)
              image_with_lm_3d = image_drawer.draw_lm(sess, image, landmarks_3d, "image_with_landmarks_3d")
              summary_writer.add_summary(image_with_box)
              summary_writer.add_summary(image_with_lm)
              summary_writer.add_summary(image_with_lm_3d)

            example = _convert_tfexample(
                image_data, b'jpg', height, width, bbox, landmarks_2d, landmarks_3d, params)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
#  filename = _DATA_URL.split('/')[-1]
#  filepath = os.path.join(dataset_dir, filename)
#  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, _PATH_EVENTS)
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, add_image_summaries=False):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    add_image_summaries: Whether to store events in dataset_dir/events directory
  """
  if add_image_summaries and not tf.gfile.Exists(os.path.join(dataset_dir, _PATH_EVENTS)):
    tf.gfile.MakeDirs(os.path.join(dataset_dir, _PATH_EVENTS))

#  if _dataset_exists(dataset_dir):
#    print('Dataset files already exist. Exiting without re-creating them.')
#    return
  
  writer = None
  if add_image_summaries:
    writer = tf.summary.FileWriter(os.path.join(dataset_dir, _PATH_EVENTS))

  # Divide into train and test:
  #random.seed(_RANDOM_SEED)
  #random.shuffle(photo_filenames)
  training_filenames, training_label_filenames = _get_filenames(dataset_dir, 'train')
  print("%d images for training" % len(training_filenames))
  validation_filenames, validation_label_filenames = _get_filenames(dataset_dir, 'validation')
  print("%d images for validation" % len(validation_filenames))

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, training_label_filenames,
                   dataset_dir, writer)
  _convert_dataset('validation', validation_filenames, validation_label_filenames,
                   dataset_dir, writer)

  # Finally, write the labels file:
  #labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  #dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the 300W_LP dataset!')
  
if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(description='convert 300W_LP dataset to tfrecord')
  parser.add_argument('--dataset_dir', dest='dataset_dir', help='path to dataset',
            default="/home/dff/NewDisk/300W_LP", type=str)
  parser.add_argument('--add_summary', dest='add_summary', help='Record summary',
            default=False, type=bool)
  args = parser.parse_args()
  run(args.dataset_dir, args.add_summary)
