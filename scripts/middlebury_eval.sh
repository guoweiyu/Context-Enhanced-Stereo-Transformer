#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3
python main.py  --batch_size 1\
                --checkpoint meddlebury_noft\
                --num_workers 2\
                --eval\
                --px_error_threshold 3\
                --num_attn_layers 6\
                --dataset middlebury2014_test\
                --dataset_directory /data/MiddEval3/trainingQ/\
                --resume /your/model/path
