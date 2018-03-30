# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/mobilenet_v2_training_2/ft

# Where the training (fine-tuned) checkpoint and logs will be saved to.
EVAL_DIR=/tmp/mobilenet_v2_eval/

# Where the dataset is saved to.
DATASET_DIR=/home/dff/NewDisk/300VW

# Fine-tune only the new layers for 2000 steps.
python eval_landmark_regressor.py \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=landmark_300VW \
  --dataset_split_name=validation_1 \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2 \
  --preprocessing_name face_landmark \
  --batch_size=1 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
