# 1st Place Solution of Kaggle Happywhale Competition
This is the charmq's part of the Preferred Dolphin's solution.

## Dataset
Please place competition data under `happywhale_data/`.
```
$ ls happywhale_data
backfin_test_charm.csv    fullbody_train.csv     test2.csv          train.csv
backfin_train_charm.csv   individual_id.npy      test_backfin.csv   train_images
fullbody_test_charm.csv   pseudo_labels          test_images        yolov5_test.csv
fullbody_test.csv         sample_submission.csv  train2.csv         yolov5_train.csv
fullbody_train_charm.csv  species.npy            train_backfin.csv
```

## Reproducing the winning score
We ensembled many models in the final submission, but first we will explain how to reproduce the training of a single model using pseudo label `round2.csv`.

### Step 1: Training
This is an example command to train efficientnet-b7 with all competition training data and pseudo labels.
```
python -m run.train \
  dataset.phase=all \
  model.base_model=tf_efficientnet_b7 \
  dataset.pseudo_label_filename=round2.csv \
  out_dir=results/efficientnet_b7_pl_round2
```

### Step 2: Inference
After training is done, inference needs to be performed on the four combinations of train/test and fullbody bbox/fullbody_charm bbox.
```
python -m run.train \
  dataset.phase=test \
  dataset.bbox=fb \
  test_model=$(find results/efficientnet_b7_pl_round2 -name model_weights.pth) \
  out_dir=results/efficientnet_b7_pl_round2_bbox_fb_test

python -m run.train \
  dataset.phase=valid \
  dataset.bbox=fb \
  test_model=$(find results/efficientnet_b7_pl_round2 -name model_weights.pth) \
  out_dir=results/efficientnet_b7_pl_round2_bbox_fb_train

python -m run.train \
  dataset.phase=test \
  dataset.bbox=fb_charm \
  test_model=$(find results/efficientnet_b7_pl_round2 -name model_weights.pth) \
  out_dir=results/efficientnet_b7_pl_round2_bbox_fb_charm_test

python -m run.train \
  dataset.phase=valid \
  dataset.bbox=fb_charm \
  test_model=$(find results/efficientnet_b7_pl_round2 -name model_weights.pth) \
  out_dir=results/efficientnet_b7_pl_round2_bbox_fb_charm_train
```

After the inference is done, put the result files in one place for the ensemble.
```
cp results/efficientnet_b7_pl_round2_bbox_fb_test/test_results/test_results.npz \
  results/efficientnet_b7_pl_round2/test_fullbody_results.npz

cp results/efficientnet_b7_pl_round2_bbox_fb_train/test_results/test_results.npz \
  results/efficientnet_b7_pl_round2/train_fullbody_results.npz

cp results/efficientnet_b7_pl_round2_bbox_fb_charm_test/test_results/test_results.npz \
  results/efficientnet_b7_pl_round2/test_fullbody_charm_results.npz

cp results/efficientnet_b7_pl_round2_bbox_fb_charm_train/test_results/test_results.npz \
  results/efficientnet_b7_pl_round2/test_fullbody_charm_results.npz
```

### Cross validation
The following commands can be used to perform cross validation. It is not recommended to use the pseudo label as it is, as it will cause a leak.
```
for FOLD in {0..4}; do
python -m run.train \
  dataset.num_folds=5 \
  dataset.test_fold=$FOLD \
  model.base_model=tf_efficientnet_b7 \
  out_dir=results/efficientnet_b7_$FOLD; done
```

### Example for ensemble
Various models can be trained and ensembled by changing the architecture, image size, and bbox mix ratio, as in the following command.
```
python -m run.train \
  dataset.phase=all \
  model.base_model=tf_efficientnetv2_l \
  preprocessing.h_resize_to=1024 \
  preprocessing.w_resize_to=1024 \
  training.batch_size=16 \
  training.epoch=20 \
  training.num_gpus=8 \
  optimizer.lr=1e-4 \
  optimizer.lr_head=1e-3 \
  dataset.p_fb=0.6 \
  dataset.p_fb_charm=0.1 \
  dataset.p_backfin=0.05 \
  dataset.p_backfin_charm=0.05 \
  dataset.p_detic=0.05 \
  dataset.p_yolo=0.05 \
  dataset.p_none=0.1 \
  dataset.pseudo_label_filename=round1.csv \
  dataset.pseudo_label_conf=0.5 \
  training.resume_from=(path to checkpoint) \
  model.restore_path=(path to weight) \
  out_dir=results/efficientnetv2_l_pl_round1
```
Inference commands are similar to step2.
```
python -m run.train \
  dataset.phase=test \
  dataset.bbox=fb \
  model.base_model=tf_efficientnetv2_l \
  preprocessing.h_resize_to=1024 \
  preprocessing.w_resize_to=1024 \
  test_model=(path to weight) \
  # test_model=$(find results/efficientnetv2_l_pl_round1 -name model_weights.pth) \
  out_dir=results/efficientnetv2_l_pl_round1_bbox_fb_test

...
```

## Links
- For an overview of our key ideas and detailed explanation, please also refer to [1st Place Solution](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/320192) in Kaggle discussion.
- My teammate [knshnb's repository](https://github.com/knshnb/kaggle-happywhale-1st-place)
