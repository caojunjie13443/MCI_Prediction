#! /bin/bash

cd /home/junjie/MCI_Prediction
PWD=/home/junjie/MCI_Prediction
pretrained_model="resnet101"
test_id=("1683" "1777" "1828")

export CUDA_VISIBLE_DEVICES=0

for id in "${test_id[@]}"; do
    echo "Running $id..."
    python $PWD/train_audio_pretrained_model.py \
    --lang_id $id \
    --pretrained_model $pretrained_model
done
