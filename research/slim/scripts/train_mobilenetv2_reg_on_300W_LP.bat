rem Where the pre-trained InceptionV1 checkpoint is saved to.
SET PRETRAINED_CHECKPOINT_DIR=checkpoints\mobilenetv2_sgd\model.ckpt-1450000

rem  Where the training (fine-tuned) checkpoint and logs will be saved to.
SET TRAIN_DIR=%TEMP%\mobilenet_v2_training\

rem  Where the dataset is saved to.
SET DATASET_DIR=G:\300W_LP

rem  Fine-tune only the new layers for 2000 steps.
python train_image_regressor.py ^
  --train_dir=%TRAIN_DIR% ^
  --dataset_name=landmark_300W_LP ^
  --dataset_split_name=train ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=mobilenet_v2 ^
  --preprocessing_name face_landmark ^
  --checkpoint_path=%PRETRAINED_CHECKPOINT_DIR% ^
  --checkpoint_exclude_scopes=MobilenetV2/Logits ^
  --max_number_of_steps=1000000 ^
  --batch_size=16 ^
  --learning_rate=0.00001 ^
  --learning_rate_decay_type=fixed ^
  --save_interval_secs=1800 ^
  --save_summaries_secs=60 ^
  --log_every_n_steps=100 ^
  --optimizer=adam ^
  --clone_on_cpu=True ^
rem  --weight_decay=0.00004

rem pause
