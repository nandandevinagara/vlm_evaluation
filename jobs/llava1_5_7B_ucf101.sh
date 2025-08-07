#!/bin/bash

# execute from vlm_evaluation folder, all other paths are managed
# sbatch -p accelerated --gres=gpu:1 -t 12:00:00 --mem=200000 jobs/llava1_5_7B_ucf101.sh

source /hkfs/work/workspace/scratch/st_st189656-myspace/myenv/bin/activate
echo "environment activated"

cd pipeline/
python pipeline.py --model llava1_5_7B --dataset ucf101
echo"pipeline is started"

