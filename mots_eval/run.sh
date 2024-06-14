#!/bin/bash
conda activate seg-train
python mots_eval/eval.py path/to/tracking_results path/to/gt_folder path/to/train.seqmap
