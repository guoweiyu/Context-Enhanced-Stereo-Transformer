#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 800\
                --batch_size 1\
                --checkpoint middlebury_ft\
                --num_workers 2\
                --dataset middlebury2014\
                --dataset_directory /data/middlebury2014_full\
                --ft\
                --resume /home/guoweiyu/stereo-transformer/run/sceneflow/pretrain/experiment_5/epoch_17_model.pth.tar