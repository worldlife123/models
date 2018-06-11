# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=checkpoints/resnet_v1_50/resnet_v1_50.ckpt

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/resnet_v1_50_training/

# Where the dataset is saved to.
DATASET_DIR=/home/dff/NewDisk/300W_LP

# Fine-tune only the new layers for 2000 steps.
python train_image_regressor.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=landmark_300W_LP \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=resnet_v1_50 \
  --preprocessing_name=face_landmark \
  --max_number_of_steps=150000 \
  --batch_size=16 \
  --learning_rate=0.00001 \
  --learning_rate_decay_type=fixed \
  --weight_decay=0.00004 \
  --save_interval_secs=1800 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=adam \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}\
  --checkpoint_exclude_scopes=resnet_v1_50/logits \
#  --end_learning_rate=0.00001 \

