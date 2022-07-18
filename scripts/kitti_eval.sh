#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3
python main.py  --batch_size 1\
                --checkpoint meddlebury_noft\
                --num_workers 2\
                --eval\
                --px_error_threshold 3\
                --num_attn_layers 6\
                --dataset sintel\
                --dataset_directory /data/Sintel/train\
                --resume /home/guoweiyu/stereo-transformer/run/sceneflow/pretrain/experiment_5/epoch_17_model.pth.tar
