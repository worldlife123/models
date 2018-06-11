#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# This script prepares the various different versions of MobileNet models for
# use in a mobile application. If you don't specify your own trained checkpoint
# file, it will download pretrained checkpoints for ImageNet. You'll also need
# to have a copy of the TensorFlow source code to run some of the commands,
# by default it will be looked for in ./tensorflow, but you can set the
# TENSORFLOW_PATH environment variable before calling the script if your source
# is in a different location.
# The main slim/nets/mobilenet_v1.md description has more details about the
# model, but the main points are that it comes in four size versions, 1.0, 0.75,
# 0.50, and 0.25, which controls the number of parameters and so the file size
# of the model, and the input image size, which can be 224, 192, 160, or 128
# pixels, and affects the amount of computation needed, and the latency.
# Here's an example generating a frozen model from pretrained weights:
#

set -e

print_usage () {
  echo "Creates a frozen mobilenet model suitable for mobile use"
  echo "Usage:"
  echo "$0 <resnet version> <input size> [checkpoint path]"
}

RESNET_V1_VERSION=$1
IMAGE_SIZE=$2
CHECKPOINT=$3

if [[ ${RESNET_V1_VERSION} = "50" ]]; then
   SLIM_NAME=resnet_v1_50
elif [[ ${RESNET_V1_VERSION} = "101" ]]; then
   SLIM_NAME=resnet_v1_101
elif [[ ${RESNET_V1_VERSION} = "152" ]]; then
   SLIM_NAME=resnet_v1_152
elif [[ ${RESNET_V1_VERSION} = "200" ]]; then
   SLIM_NAME=resnet_v1_200
else
  echo "Bad mobilenet version, should be one of 50, 101, 152, 200"
  print_usage
  exit 1
fi

if [[ ${IMAGE_SIZE} -ne "224" ]] && [[ ${IMAGE_SIZE} -ne "192" ]] && [[ ${IMAGE_SIZE} -ne "160" ]] && [[ ${IMAGE_SIZE} -ne "128" ]]; then
  echo "Bad input image size, should be one of 224, 192, 160, or 128"
  print_usage
  exit 1
fi

if [[ ${TENSORFLOW_PATH} -eq "" ]]; then
   TENSORFLOW_PATH=~/tensorflow
fi

if [[ ! -d ${TENSORFLOW_PATH} ]]; then
   echo "TensorFlow source folder not found. You should download the source and then set"
   echo "the TENSORFLOW_PATH environment variable to point to it, like this:"
   echo "export TENSORFLOW_PATH=/my/path/to/tensorflow"
   print_usage
   exit 1
fi

MODEL_FOLDER=/tmp/resnet_v1_${RESNET_V1_VERSION}_${IMAGE_SIZE}_quantize
if [[ -d ${MODEL_FOLDER} ]]; then
  echo "Model folder ${MODEL_FOLDER} already exists! Emptying ${MODEL_FOLDER} ..."
  rm -rf ${MODEL_FOLDER}/*
else
  mkdir ${MODEL_FOLDER}
fi

if [[ ${CHECKPOINT} = "" ]]; then
  echo "Checkpoint not found!"
  exit 1
fi

echo "*******"
echo "Exporting graph architecture to ${MODEL_FOLDER}/unfrozen_graph.pb"
echo "*******"
python export_inference_graph.py \
  --model_name=${SLIM_NAME} --image_size=${IMAGE_SIZE} --logtostderr \
  --output_file=${MODEL_FOLDER}/unfrozen_graph.pb --dataset_name=landmark_300W_LP --dataset_dir=${MODEL_FOLDER}

cd ${TENSORFLOW_PATH}

OUTPUT_NODE_NAMES=${SLIM_NAME}/predictions/Reshape_1

echo "*******"
echo "Freezing graph to ${MODEL_FOLDER}/frozen_graph.pb"
echo "*******"
bazel run tensorflow/python/tools:freeze_graph -- \
  --input_graph=${MODEL_FOLDER}/unfrozen_graph.pb \
  --input_checkpoint=${CHECKPOINT} \
  --input_binary=true --output_graph=${MODEL_FOLDER}/frozen_graph.pb \
  --output_node_names=${OUTPUT_NODE_NAMES}

echo "Quantizing weights to ${MODEL_FOLDER}/quantized_graph.pb"
#bazel run tensorflow/tools/graph_transforms:transform_graph -- \
#  --in_graph=${MODEL_FOLDER}/frozen_graph.pb \
#  --out_graph=${MODEL_FOLDER}/quantized_graph.pb \
#  --inputs=input --outputs=${OUTPUT_NODE_NAMES} \
#  --transforms='fold_constants fold_batch_norms quantize_weights'

echo "*******"
echo "Running label_image using the graph"
echo "*******"
#bazel build tensorflow/examples/label_image:label_image
#bazel-bin/tensorflow/examples/label_image/label_image \
#  --input_layer=input --output_layer=${OUTPUT_NODE_NAMES} \
#  --graph=${MODEL_FOLDER}/quantized_graph.pb --input_mean=-127 --input_std=127 \
#  --image=tensorflow/examples/label_image/data/grace_hopper.jpg \
#  --input_width=${IMAGE_SIZE} --input_height=${IMAGE_SIZE} --labels=${MODEL_FOLDER}/labels.txt

echo "*******"
echo "Saved graphs to ${MODEL_FOLDER}/frozen_graph.pb and ${MODEL_FOLDER}/quantized_graph.pb"
echo "*******"

echo "Exporting tflite model to ${MODEL_FOLDER}/lite_model.tflite"
bazel run tensorflow/contrib/lite/toco:toco -- \
  --input_file=${MODEL_FOLDER}/frozen_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=${MODEL_FOLDER}/lite_model.tflite \
  --inference_type=FLOAT \
  --inference_input_type=FLOAT --input_arrays=input \
  --output_arrays=${OUTPUT_NODE_NAMES} \
  --input_shapes=1,${IMAGE_SIZE},${IMAGE_SIZE},3
  
#TODO: do not work!
echo "Exporting quantized tflite model to ${MODEL_FOLDER}/lite_model_quantized.tflite"
#bazel run tensorflow/contrib/lite/toco:toco -- \
#  --input_file=${MODEL_FOLDER}/quantized_graph.pb \
#  --input_format=TENSORFLOW_GRAPHDEF \
#  --output_format=TFLITE \
#  --output_file=${MODEL_FOLDER}/lite_model_quantized.tflite \
#  --inference_type=QUANTIZED_UINT8 \
#  --inference_input_type=QUANTIZED_UINT8 --input_arrays=input \
#  --output_arrays=${OUTPUT_NODE_NAMES} \
#  --input_shapes=1,${IMAGE_SIZE},${IMAGE_SIZE},3
