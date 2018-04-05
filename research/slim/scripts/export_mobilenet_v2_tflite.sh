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
  echo "$0 <mobilenet version> <input size> [input .pb path]"
}

MOBILENET_VERSION=$1
IMAGE_SIZE=$2
INPUT_PB=$3

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

MODEL_SCOPE=MobilenetV2
OUTPUT_NODE_NAMES=MobilenetV2/Logits/Reshape #MobilenetV2/Predictions/Reshape_1

if [[ ${IMAGE_SIZE} -ne "224" ]] && [[ ${IMAGE_SIZE} -ne "192" ]] && [[ ${IMAGE_SIZE} -ne "160" ]] && [[ ${IMAGE_SIZE} -ne "128" ]]; then
  echo "Bad input image size, should be one of 224, 192, 160, or 128"
  print_usage
  exit 1
fi

MODEL_FOLDER=/tmp/mobilenet_v2_${MOBILENET_VERSION}_${IMAGE_SIZE}

cd ${TENSORFLOW_PATH}

echo "Exporting tflite model to ${MODEL_FOLDER}/lite_model.tflite"
bazel run tensorflow/contrib/lite/toco:toco -- \
  --input_file=${MODEL_FOLDER}/${INPUT_PB} \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=${MODEL_FOLDER}/lite_model.tflite \
  --inference_type=FLOAT \
  --inference_input_type=FLOAT --input_arrays=input \
  --output_arrays=${OUTPUT_NODE_NAMES} \
  --input_shapes=1,224,224,3
