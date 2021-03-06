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
  echo "Creates a frozen mobilenet-v2 model suitable for mobile use"
  echo "Usage:"
  echo "$0 <mobilenet version> <input size> [checkpoint path]"
}

MOBILENET_VERSION=$1
IMAGE_SIZE=$2
CHECKPOINT=$3

if [[ ${MOBILENET_VERSION} = "1.0" ]]; then
   SLIM_NAME=mobilenet_v2
elif [[ ${MOBILENET_VERSION} = "0.75" ]]; then
   SLIM_NAME=mobilenet_v2_075
elif [[ ${MOBILENET_VERSION} = "0.50" ]]; then
   SLIM_NAME=mobilenet_v2_050
elif [[ ${MOBILENET_VERSION} = "0.25" ]]; then
   SLIM_NAME=mobilenet_v2_025
else
  echo "Bad mobilenet version, should be one of 1.0, 0.75, 0.50, or 0.25"
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

MODEL_FOLDER=/tmp/mobilenet_v2_${MOBILENET_VERSION}_${IMAGE_SIZE}_quantize
if [[ -d ${MODEL_FOLDER} ]]; then
  echo "Model folder ${MODEL_FOLDER} already exists! Removing ${MODEL_FOLDER} ..."
  rm -rf ${MODEL_FOLDER}/*
else
  mkdir ${MODEL_FOLDER}
fi

if [[ ${CHECKPOINT} = "" ]]; then
  echo "*******"
  echo "Pretrained weights ${CHECKPOINT} not found! "
  echo "*******"
  exit 1
fi

echo "*******"
echo "Exporting graph architecture to ${MODEL_FOLDER}/unfrozen_graph.pb"
echo "*******"
python export_inference_graph.py \
  --model_name=${SLIM_NAME} --dataset_name=landmark_300W_LP --image_size=${IMAGE_SIZE} --logtostderr \
  --output_file=${MODEL_FOLDER}/unfrozen_graph.pb --dataset_dir=${MODEL_FOLDER} --quantize=True #--is_training=True

cd ${TENSORFLOW_PATH}

#bazel run tensorflow/python/tools:import_pb_to_tensorboard -- \
#  --log_dir=${MODEL_FOLDER} \
#  --model_dir=${MODEL_FOLDER}/unfrozen_graph.pb

OUTPUT_NODE_NAMES=MobilenetV2/Logits/SpatialSqueeze #MobilenetV2/Predictions/Reshape_1

echo "*******"
echo "Freezing graph to ${MODEL_FOLDER}/frozen_graph.pb"
echo "*******"
bazel run tensorflow/python/tools:freeze_graph -- \
  --input_graph=${MODEL_FOLDER}/unfrozen_graph.pb \
  --input_checkpoint=${CHECKPOINT} \
  --input_binary=true --output_graph=${MODEL_FOLDER}/frozen_graph.pb \
  --output_node_names=${OUTPUT_NODE_NAMES}
  
#bazel run tensorflow/tools/graph_transforms:transform_graph -- \
#  --in_graph=${MODEL_FOLDER}/frozen_graph.pb \
#  --out_graph=${MODEL_FOLDER}/folded_graph.pb \
#  --inputs=input --outputs=${OUTPUT_NODE_NAMES} \
#  --transforms='obfuscate_names'
  
#bazel run tensorflow/python/tools:import_pb_to_tensorboard -- \
#  --log_dir=${MODEL_FOLDER} \
#  --model_dir=${MODEL_FOLDER}/folded_graph.pb

echo "*******"
echo "Exporting quantized tflite model to ${MODEL_FOLDER}/lite_model_quantized.tflite"
echo "*******"
bazel run tensorflow/contrib/lite/toco:toco -- \
  --input_file=${MODEL_FOLDER}/frozen_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=${MODEL_FOLDER}/lite_model_quantized.tflite \
  --dump_graphviz=${MODEL_FOLDER} \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=${OUTPUT_NODE_NAMES} \
  --input_shapes=1,224,224,3 \
  --std_values=127.5 --mean_values=127.5 \
#  --default_ranges_min=-70 \
#  --default_ranges_max=70 \

