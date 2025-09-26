#!/bin/bash
#SBATCH --account=oymak0
#SBATCH --partition=gpu_mig40
#SBATCH --time=00-24:30:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=128GB



pushd /scratch/oymak_root/oymak0/xychang/water-master
export PYTHONPATH=$(pwd):$PYTHONPATH  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

python MyProject/train_unified.py -c MyProject/config_transformer.json -d 0 --model informer #slurm-33091711


