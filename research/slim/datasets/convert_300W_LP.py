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
_NUM_SHARDS = 5


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
        photo_path = os.path.join(full_directory, filename)
        photo_filenames.append(photo_path)
        label_filenames.append(os.path.join(dataset_dir, "landmarks", directory, filename[:-4] + "_pts.mat"))

  return photo_filenames, label_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'landmark_300W_LP_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, photo_filenames, label_filenames, dataset_dir):
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

    with tf.Session('') as sess:

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
            image_data = tf.gfile.FastGFile(photo_filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            mat = sio.loadmat(label_filenames[i])
            pt2d = mat['pts_2d'].T
            x_min = min(pt2d[0,:])
            y_min = min(pt2d[1,:])
            x_max = max(pt2d[0,:])
            y_max = max(pt2d[1,:])

            example = dataset_utils.image_to_tfexample_face_landmark(
                image_data, b'jpg', height, width, [x_min, y_min, x_max, y_max], list(pt2d.ravel()))
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'flower_photos')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  #random.shuffle(photo_filenames)
  training_filenames, training_label_filenames = _get_filenames(dataset_dir, 'train')
  print("%d images for training" % len(training_filenames))
  validation_filenames, validation_label_filenames = _get_filenames(dataset_dir, 'validation')
  print("%d images for validation" % len(validation_filenames))

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, training_label_filenames,
                   dataset_dir)
  _convert_dataset('validation', validation_filenames, validation_label_filenames,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the 300W_LP dataset!')
  
if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(description='convert 300W_LP dataset to tfrecord')
  parser.add_argument('--dataset_dir', dest='dataset_dir', help='path to dataset',
            default="/home/dff/NewDisk/300W_LP", type=str)
  args = parser.parse_args()
  run(args.dataset_dir)
