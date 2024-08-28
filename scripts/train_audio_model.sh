#! /bin/bash
cd /home/junjie/MCI_Prediction
PWD=/home/junjie/MCI_Prediction

export CUDA_VISIBLE_DEVICES=1

python $PWD/train_audio_pretrained_model.py
