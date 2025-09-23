#!/bin/bash

# Loop through the configurations
for i in $(seq 1 16); do
  echo "Running experiment $i"
  # Calculate the GPU device to use based on the batch number
  # Calculate the GPU device to use based on the batch number
  # Map the i values to the desired GPU indices (2, 3, 6, 7)
  case $(( (i - 1) % 4 )) in
    0) gpu_device=4;;
    1) gpu_device=1;;
    2) gpu_device=2;;
    3) gpu_device=3;;
  esac

  # Run the command with the assigned GPU device
  CUDA_VISIBLE_DEVICES=$gpu_device python train.py -c ablation_small_donet/config_ablation_$i.json &

  # Wait for 4 commands to be running before starting the next batch
  if (( (i - 1) % 4 == 3 )); then
    wait
  fi
done

# Wait for all remaining commands to finish
wait