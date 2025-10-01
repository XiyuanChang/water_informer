#!/bin/bash
#SBATCH --account=oymak_owned1
#SBATCH --partition=gpu_mig40
#SBATCH --time=00-32:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=128GB



pushd /scratch/oymak_root/oymak0/xychang/water-master
export PYTHONPATH=$(pwd):$PYTHONPATH  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

#python water-master/train_unified.py -c MyProject/config_transformer.json -d 0 --model informer 
python water-master/train_unified.py -c MyProject/config_transformer.json -d 0 --model informer -r unified_ablation_informer/baseline_dropout_0.1_665/models/informer/0929_144935/checkpoint-epoch70.pth 

#python MyProject/train_lstm.py -c MyProject/config_LSTM.json -d 0  


#test checkpoint (informer)
#python water-master/lstm_predict.py 
#python water-master/lstm_predict_informer_e70.py 
