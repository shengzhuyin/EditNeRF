#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
python3 -u ./walk_learning.py --config configs/photoshapes/config.txt >color_walk.log 2>&1
