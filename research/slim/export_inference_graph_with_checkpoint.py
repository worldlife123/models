from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, argparse, time
import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from datasets import dataset_factory
from nets import nets_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')
    
tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_string(
    'checkpoint_dir', '', 'Directory to checkpoint')
    
tf.app.flags.DEFINE_string(
    'model_name_scope', '', ' ')
    
tf.app.flags.DEFINE_string(
    'output_node', '', ' ')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  
  with tf.Graph().as_default() as graph:
    '''dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)'''
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=136,
        is_training=False)
    image_size = FLAGS.image_size or network_fn.default_image_size
    input_img = np.random.randn(1,image_size,image_size,3)
    placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                 shape=[FLAGS.batch_size, image_size,
                                        image_size, 3])
    model, end_points = network_fn(placeholder)
    init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint_dir, slim.get_model_variables(FLAGS.model_name_scope))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      init_fn(sess)
      print(sess.run(model, feed_dict={placeholder: input_img}))
      output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), output_node_names=[FLAGS.output_node])

    with gfile.GFile(FLAGS.output_file, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
      
    with tf.Session(graph=tf.Graph()) as sess:
        importer.import_graph_def(output_graph_def)

        input_tensor = sess.graph.get_tensor_by_name("import/input:0")
        output_tensor = sess.graph.get_tensor_by_name("import/"+FLAGS.output_node+":0")
        output_pb = sess.run(output_tensor, feed_dict={input_tensor:input_img})
        print(output_pb)


if __name__ == '__main__':
  tf.app.run()

