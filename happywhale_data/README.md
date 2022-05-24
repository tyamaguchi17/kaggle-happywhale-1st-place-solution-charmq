# happywhale_data
Competition dataset (train.csv/sample_submission.csv/train_images/test_images) should be placed in this directory.

The description of each file is as follows

## fullbody_train.csv/fullbody_test.csv
Downloaded from [Jan Bre's dataset](https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations).

## fullbody_train_charm.csv/fullbody_test_charm.csv
Made from [Jan Bre's notebook](https://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5) with DIM=1024, EPOCH=50.
We used fullbody_annotations.csv from [Jan Bre's dataset](https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations) as training data.

## train_backfin.csv/test_backfin.csv
Downloaded from output of [Jan Bre's notebook](https://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5) (copy of train.csv/test.csv).

## backfin_train_charm.csv/backfin_test_charm.csv
Made from [Jan Bre's notebook](https://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5) with DIM=1024, EPOCH=50.

## train2.csv/test2.csv
Downloaded from [phalanx's dataset](https://www.kaggle.com/datasets/phalanx/whale2-cropped-dataset). Bboxes of Detic.

## yolov5_train.csv/yolov5_test.csv
Downloaded from output of [Awsaf's notebook](https://www.kaggle.com/code/awsaf49/happywhale-cropped-dataset-yolov5) (copy of train.csv/test.csv).

## species.npy/individual_id.npy
Arrays of label encoders used in charmq's pipeline.
