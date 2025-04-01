#!/bin/bash

exp_types=("accuracy" "scale")
export PYTHONPATH=$PWD
for exp_type in "${exp_types[@]}"
do
    echo "Running experiment type: $exp_type"
    python3 experiments.py --exp_type "$exp_type" &
done

wait

echo "All experiments completed"