#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1
python main.py  --epochs 25\
                --batch_size 1\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 4\
                --dataset sceneflow\
                --nheads 4\
                --num_attn_layers 6\
                --dataset_directory /yourdatapath\
                
