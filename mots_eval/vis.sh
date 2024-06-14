#!/bin/bash
conda activate deformable_detr

python mots_vis/visualize_mots.py path/to/tracking_results path/to/images/ path/to/vis_output/ path/to/train.seqmap
