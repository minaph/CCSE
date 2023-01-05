#!/bin/bash
conda activate ccse38
cd ~/Projects/CCSE
PYTHONPATH=$PYTHONPATH:./ python -u scripts/train_instance.py --config config/instance_segmentation/mask_rcnn_R_50_FPN_3x_kaiti_small.yaml