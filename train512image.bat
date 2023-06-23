@echo off

set DATA_FLAGS=--data_dir F:\yang_Projects\Datasets\Task1\pelvis --batch_size 2
set TRAIN_FLAGS=--image_size 256 --class_cond False

call F:\yang_Environments\torch\venv\Scripts\activate

python image_train.py %DATA_FLAGS% %TRAIN_FLAGS% 