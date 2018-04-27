# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=checkpoints/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/mobilenet_v1_training_quantize_2/


# Where the dataset is saved to.
DATASET_DIR=/home/dff/NewDisk/300W_LP

# Fine-tune only the new layers for 2000 steps.
python train_landmark_regressor.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=landmark_300W_LP \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v1 \
  --preprocessing_name face_landmark \
  --max_number_of_steps=200000 \
  --batch_size=16 \
  --learning_rate=0.00001 \
  --learning_rate_decay_type=fixed \
  --weight_decay=0.00004 \
  --save_interval_secs=1800 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --quantize=True \
  --quantize_delay=100000 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --ignore_missing_vars=True
#  --end_learning_rate=0.00001 \

