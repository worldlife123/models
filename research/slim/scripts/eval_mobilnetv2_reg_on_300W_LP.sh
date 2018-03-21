# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/tmp/mobilenet_v2_training/

# Where the training (fine-tuned) checkpoint and logs will be saved to.
EVAL_DIR=/tmp/mobilenet_v2_eval/

# Where the dataset is saved to.
DATASET_DIR=/home/dff/NewDisk/300W_LP

# Fine-tune only the new layers for 2000 steps.
python eval_landmark_regressor.py \
  --eval_dir=${EVAL_DIR} \
  --dataset_name=landmark_300W_LP \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2 \
  --preprocessing_name face_landmark \
  --batch_size=1 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=MobilenetV2/Logits \
#  --weight_decay=0.00004
